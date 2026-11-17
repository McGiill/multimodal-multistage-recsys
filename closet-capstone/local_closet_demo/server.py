"""Local-only closet demo. Read existing vectors; never load or train a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import time
from collections import defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
from runtime_paths import config_path, media_path

HERE = Path(__file__).resolve().parent
SLOT_LABEL = {"top": "상의", "bottom": "하의", "outer": "아우터", "dress": "원피스", "shoes": "신발"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class Catalog:
    def __init__(self, config_file=None):
        config_file = config_file or config_path()
        started = time.perf_counter()
        self.cfg = read(config_file)
        base = Path(config_file).resolve().parent
        out = (base / self.cfg["outfits_dir"]).resolve()
        prod = (base / self.cfg["products_dir"]).resolve()
        protocol = read(out / "protocol.json")
        provenance = read(base / self.cfg["provenance_file"])
        encoding = read(prod / "encoding_time_fashion_clip.json")
        # Refuse incompatible feature spaces instead of silently mixing encoders.
        old_weights = {f["name"]: f["sha256"] for f in provenance["provenance"]["fashion_clip"]["files"]}
        for name, digest in encoding["protocol"]["weights_sha256"].items():
            if old_weights.get(name) != digest:
                raise ValueError("Outfit/product model weights differ")
        if not protocol["encoder"]["pad_to_square"] or encoding["protocol"]["preprocessing"] != "team_pad_square_CLIP_slow":
            raise ValueError("Incompatible image preprocessing")
        if encoding["protocol"]["encoder_source_sha256"] != provenance["source_hashes"]["src/search/encoder.py"]:
            raise ValueError("Encoder source mismatch")
        self.products = read(prod / "product_manifest.json")
        frozen_manifest = prod / "catalog" / "manifest.json"
        if (sha(frozen_manifest) != encoding["protocol"]["manifest_sha256"]
                or self.products != read(frozen_manifest)):
            raise ValueError("Product metadata/vector ordering is not the frozen catalog")
        if sha(prod / "product_vectors_fashion_clip.npy") != encoding["vector_sha256"]:
            raise ValueError("Product vectors changed")

        self.queries = read(out / "queries.json")
        self.qvec = np.load(out / "query_embeddings.npy", allow_pickle=False)
        self.pvec = np.load(prod / "product_vectors_fashion_clip.npy", mmap_mode="r", allow_pickle=False)
        all_vec = np.load(out / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.garments = read(out / "approved_garments.json")
        self.reviewed_shoes = {}
        self.reviewed_dresses = set()
        self.reviewed_region_products = {}
        for slot, key in [('shoes', 'shoe_extension_file'), ('dress', 'dress_extension_file')]:
            if not self.cfg.get(key):
                continue
            extension = read(base / self.cfg[key])
            for name, digest in extension['source_sha256'].items():
                if name not in ('garments.json', 'embeddings.npy') or sha(out / name) != digest:
                    raise ValueError('Reviewed extension feature source changed')
            raw_garments = {g['garment_id']: g for g in read(out / 'garments.json')}
            existing_ids = {g['garment_id'] for g in self.garments}
            for gid in extension['garment_ids']:
                g = raw_garments[gid]
                if g['slot'] not in ({'shoes'} if slot == 'shoes' else {'dress', 'outer', 'shoes'}):
                    raise ValueError('Reviewed region has an unsupported slot')
                if gid not in existing_ids:
                    self.garments.append(g)
                    existing_ids.add(gid)
                if slot == 'dress' and g['slot'] == 'dress':
                    self.reviewed_dresses.add(gid)
            self.reviewed_shoes.update({item['garment_id']: set(item['products']) for item in extension['footwear'].values()})
            if slot == 'dress':
                for links in extension['connections'].values():
                    for selected in links.values():
                        for gid, ids in selected['groups'].items():
                            self.reviewed_region_products.setdefault(gid, set()).update(ids)
            product_index = {p['id']: i for i, p in enumerate(self.products)}
            extra_vectors = []
            for q in extension['queries']:
                row = product_index[q['product_id']]
                p = self.products[row]
                self.queries.append({'product_id': p['id'], 'query_id': extension['products'][p['id']]['query_id'],
                                     'name': p['title'], 'category': p['category_l3'], 'color': p.get('color', ''),
                                     'slot': slot, 'image_path': p['path'], 'forward_threshold': q['forward_threshold'],
                                     'candidate_garment_ids': q.get('candidate_garment_ids')})
                extra_vectors.append(self.pvec[row])
            self.qvec = np.concatenate([self.qvec, np.asarray(extra_vectors)], axis=0)
        self.replacements = {}
        self.review_only_garments = set()
        if self.cfg.get('replacement_extension_file'):
            extension = read(base / self.cfg['replacement_extension_file'])
            for name, digest in extension['source_sha256'].items():
                if name not in ('garments.json', 'embeddings.npy', 'queries.json', 'query_embeddings.npy') or sha(out / name) != digest:
                    raise ValueError('Replacement source changed')
            raw_garments = {g['garment_id']: g for g in read(out / 'garments.json')}
            existing_ids = {g['garment_id'] for g in self.garments}
            for gid in extension['garment_ids']:
                g = raw_garments[gid]
                if g['slot'] not in SLOT_LABEL:
                    raise ValueError('Unsupported replacement garment slot')
                if gid not in existing_ids:
                    self.garments.append(g)
                    existing_ids.add(gid)
                    self.review_only_garments.add(gid)
            self.replacements = extension['connections']
        self.query_by_id = {q["product_id"]: (i, q) for i, q in enumerate(self.queries)}
        self.gvec = np.asarray(all_vec[[g["embedding_row"] for g in self.garments]], dtype=np.float32)
        self.garment_by_id = {g["garment_id"]: (i, g) for i, g in enumerate(self.garments)}
        self.by_outfit = defaultdict(list)
        self.by_slot = defaultdict(list)
        for i, g in enumerate(self.garments):
            self.by_outfit[g["outfit_id"]].append(g)
            # Added regions are accessible through explicit reviewed pairs only.
            if g['garment_id'] not in self.review_only_garments:
                self.by_slot[g["slot"]].append(i)
        self.photos = {p["outfit_id"]: {k: p[k] for k in ["outfit_id", "original_path", "image_size", "publisher_style"]}
                       for p in read(out / "outfits.json") if p["outfit_id"] in self.by_outfit}
        if self.qvec.shape != (len(self.queries), 512) or self.pvec.shape != (len(self.products), 512):
            raise ValueError("Vector/metadata row count mismatch")
        for v in [self.qvec, self.gvec]:
            if not np.isfinite(v).all() or not np.allclose(np.linalg.norm(v, axis=1), 1, atol=.003):
                raise ValueError("Invalid normalized vectors")
        self.product_rows = {}
        self.category_rows = defaultdict(list)
        for i, p in enumerate(self.products):
            self.product_rows[p["id"]] = i
            self.category_rows[p.get("category_l3") or p.get("catalog", {}).get("category_l3")].append(i)
        sku_hash = hashlib.sha256("\n".join(p["sku"] for p in self.products).encode()).hexdigest()
        if sku_hash != encoding["protocol"]["sku_order_sha256"]:
            raise ValueError("Product ordering mismatch")
        self.kinds = provenance["kinds"]
        self.default_ids = [q["product_id"] for qid in self.cfg["default_queries"] for q in self.queries if q["query_id"] == qid]
        self.meta = {"service": "local-closet-demo-v1", "outfits": len(self.photos), "garments": len(self.garments),
                     "query_products": len(self.queries), "catalog_products": len(self.products),
                     "model": protocol["encoder"]["model_name"], "inference_calls": 0,
                     "startup_seconds": round(time.perf_counter() - started, 2),
                     "forward_floor": self.cfg["forward_floor"], "reverse_floor": self.cfg["reverse_floor"],
                     "reverse_candidate_max": 40,
                     "reviewed_shoe_regions": len(self.reviewed_shoes),
                     "reviewed_dress_regions": len(self.reviewed_dresses),
                     "replacement_connections": sum(map(len, self.replacements.values())),
                     "replacement_only_regions": len(self.review_only_garments),
                     "dress_scope": "Two reviewed dress styling examples; excludes unreviewed dress-slot detections",
                     "shoe_scope": "Small visually reviewed extension using existing cached vectors",
                     "note": "Automatic candidates, not human-validated matches; no personalized reranking in this demo."}

    def product(self, pid):
        if pid in self.query_by_id:
            _, p = self.query_by_id[pid]
            return {"id": pid, "name": p["name"], "category": p["category"], "color": p["color"],
                    "slot": p["slot"], "image": f"/media/product/{pid}", "can_recommend": True}
        if pid not in self.product_rows:
            raise KeyError(pid)
        p = self.products[self.product_rows[pid]]
        return {"id": pid, "name": p.get("title") or p.get("catalog", {}).get("name") or pid,
                "category": p.get("category_l3", ""), "color": p.get("color", ""),
                "image": f"/media/product/{pid}", "can_recommend": False}

    @lru_cache(maxsize=120)
    def recommend(self, pid):
        qi, q = self.query_by_id[pid]
        rows = np.asarray(self.by_slot[q["slot"]], dtype=int)
        if q.get('candidate_garment_ids'):
            # A dress query cannot search unreviewed jumpsuit/two-piece detections.
            rows = np.asarray([self.garment_by_id[gid][0] for gid in q['candidate_garment_ids']], dtype=int)
            if any(self.garments[row]['slot'] != q['slot'] for row in rows):
                raise ValueError('Query candidate slot mismatch')
        scores = self.gvec[rows] @ self.qvec[qi]
        gate = max(self.cfg["forward_floor"], float(scores.max()) - self.cfg["forward_gap"]) if len(scores) else self.cfg["forward_floor"]
        # Preserve the frozen review threshold when narrowing the candidate pool.
        gate = max(gate, q.get('forward_threshold', gate))
        results, seen = [], set()
        for ix in np.argsort(-scores, kind="stable"):
            if scores[ix] < gate:
                break
            g = self.garments[rows[ix]]
            oid = g["outfit_id"]
            if oid in seen or not media_path(self.photos[oid]["original_path"]).is_file():
                continue
            seen.add(oid)
            p = self.photos[oid]
            results.append({"id": oid, "image": f"/media/outfit/{oid}", "size": p["image_size"],
                            "style": p["publisher_style"] or "코디", "matched_garment_id": g["garment_id"],
                            "box": g["box"], "slot": g["slot"], "similarity": float(scores[ix])})
            if len(results) >= self.cfg["forward_limit"]:
                break
        # Keep automatic candidates/thresholds unchanged. Add only frozen visual-review pairs.
        for oid, selected in self.replacements.get(pid, {}).items():
            i, g = self.garment_by_id[selected['matched_garment_id']]
            score = float(self.gvec[i] @ self.qvec[qi])
            floor = max(self.cfg['forward_floor'], selected['review_threshold'])
            if g['outfit_id'] != oid or g['slot'] != q['slot'] or score < floor or abs(score - selected['similarity']) > 1e-5:
                raise ValueError('Invalid reviewed forward pair')
            p = self.photos[oid]
            item = {'id': oid, 'image': f'/media/outfit/{oid}', 'size': p['image_size'],
                    'style': p['publisher_style'] or '코디', 'matched_garment_id': g['garment_id'],
                    'box': g['box'], 'slot': g['slot'], 'similarity': score,
                    'reviewed_pair': True, 'review_threshold': floor}
            position = next((j for j, r in enumerate(results) if r['id'] == oid), None)
            if position is None:
                results.append(item)
            else:
                results[position] = item
        return {"product": self.product(pid), "outfits": results, "threshold": gate}

    @lru_cache(maxsize=4096)
    def similar(self, gid, excluded_pid, limit=None, allowed_ids=None):
        limit = self.cfg['reverse_limit'] if limit is None else int(limit)
        if not 1 <= limit <= 40:
            raise ValueError('Reverse candidate limit must be between 1 and 40')
        i, g = self.garment_by_id[gid]
        # Restrict by detected kind: do not substitute shorts for long trousers.
        categories = self.kinds.get(g["kind"], {}).get("l3", [])
        rows = np.asarray(sorted({i for c in categories for i in self.category_rows.get(c, [])}), dtype=int)
        if allowed_ids is not None:
            allowed = set(allowed_ids)
            rows = np.asarray([row for row in rows if self.products[row]['id'] in allowed], dtype=int)
        elif g['slot'] == 'shoes':
            allowed = self.reviewed_shoes.get(gid, set())
            rows = np.asarray([row for row in rows if self.products[row]['id'] in allowed], dtype=int)
        elif gid in self.reviewed_region_products:
            rows = np.asarray([row for row in rows if self.products[row]['id'] in self.reviewed_region_products[gid]], dtype=int)
        if not len(rows):
            return []
        scores = np.asarray(self.pvec[rows]) @ self.gvec[i]
        candidates = np.argsort(-scores, kind="stable")
        found, families = [], set()
        for j in candidates:
            if scores[j] < self.cfg["reverse_floor"]:
                break
            p = self.products[rows[j]]
            family = p["sku"][:-3]
            if p["id"] == excluded_pid or family in families or not media_path(p["path"]).is_file():
                continue
            families.add(family)
            found.append({**self.product(p["id"]), "similarity": float(scores[j])})
            if len(found) >= limit:
                break
        return found

    def detail(self, oid, pid, reverse_limit=None):
        match = next((r for r in self.recommend(pid)["outfits"] if r["id"] == oid), None)
        if match is None:
            raise KeyError("Outfit is not a recommendation for this product")
        p = self.photos[oid]
        reviewed = self.replacements.get(pid, {}).get(oid)
        groups = []
        for g in sorted(self.by_outfit[oid], key=lambda g: ["outer", "top", "bottom", "dress", "shoes"].index(g["slot"])):
            anchor = g["garment_id"] == match["matched_garment_id"]
            allowed = tuple(reviewed['groups'].get(g['garment_id'], [])) if reviewed else None
            groups.append({"id": g["garment_id"], "slot": g["slot"], "label": SLOT_LABEL[g["slot"]],
                           "box": g["box"], "is_anchor": anchor,
                           "products": [] if anchor else self.similar(g["garment_id"], pid, reverse_limit, allowed)})
        alternatives = tuple(reviewed.get('shoe_alternatives', [])) if reviewed else None
        shoe_alternatives = self.similar(match['matched_garment_id'], pid, reverse_limit, alternatives) if match['slot'] == 'shoes' else []
        return {**match, "groups": groups, "product": self.product(pid), "size": p["image_size"],
                "shoe_alternatives": shoe_alternatives}

    def media(self, kind, item):
        if kind == "outfit":
            return media_path(self.photos[item]["original_path"])
        if kind != "product":
            raise KeyError(kind)
        if item in self.query_by_id:
            return media_path(self.query_by_id[item][1]["image_path"])
        return media_path(self.products[self.product_rows[item]]["path"])


class Handler(BaseHTTPRequestHandler):
    catalog: Catalog

    def send_bytes(self, data, content_type, status=200, cache=False):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "private, max-age=3600" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, obj, status=200):
        self.send_bytes(json.dumps(obj, ensure_ascii=False, allow_nan=False).encode(), "application/json; charset=utf-8", status)

    def do_GET(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query)
        try:
            if path == "/health":
                return self.send_json(self.catalog.meta)
            if path == "/api/wardrobe":
                return self.send_json({"default_ids": self.catalog.default_ids,
                                       "products": [self.catalog.product(q["product_id"]) for q in self.catalog.queries]})
            if path.startswith("/api/products/"):
                rest = path.removeprefix("/api/products/")
                if rest.endswith("/outfits"):
                    return self.send_json(self.catalog.recommend(rest.removesuffix("/outfits")))
                return self.send_json(self.catalog.product(rest))
            if path.startswith("/api/outfits/"):
                limit = query.get('reverse_limit', [None])[0]
                return self.send_json(self.catalog.detail(path.removeprefix("/api/outfits/"), query.get("product_id", [""])[0], limit))
            if path.startswith("/media/"):
                _, _, kind, item = path.split("/")
                source = self.catalog.media(kind, item)
                return self.send_bytes(source.read_bytes(), mimetypes.guess_type(source.name)[0] or "image/jpeg", cache=True)
            static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
            if path in static:
                source = HERE / "static" / static[path]
                return self.send_bytes(source.read_bytes(), (mimetypes.guess_type(source.name)[0] or "text/plain") + "; charset=utf-8")
            self.send_json({"error": "찾을 수 없는 주소입니다."}, 404)
        except (KeyError, FileNotFoundError, ValueError):
            self.send_json({"error": "연결된 상품·코디 또는 이미지가 없습니다."}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            import traceback
            traceback.print_exc()
            self.send_json({"error": "데이터를 불러오지 못했습니다. 서버 기록을 확인해 주세요."}, 500)

    def log_message(self, fmt, *args):
        if not self.path.startswith("/media/"):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if read(config_path()).get('demo_mode') == 'curated_snapshot':
        from compact_catalog import CompactCatalog
        Handler.catalog = CompactCatalog()
    else:
        Handler.catalog = Catalog()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(json.dumps(Handler.catalog.meta, ensure_ascii=False), flush=True)
    print(f"Local: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
