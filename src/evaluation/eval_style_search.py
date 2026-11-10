"""스타일 사진 검색 평가 (상품 사진을 넣은 경우): 기존 이미지 검색보다 나빠지지 않는지.

    python -m src.evaluation.eval_style_search
    python -m src.evaluation.eval_style_search --model patrickjohncyh/fashion-clip

지금 쓸 수 있는 정답은 검색 평가셋의 이미지 쿼리(카탈로그 밖 H&M 사진, 관련도는 eval.relevance 기준:
src/evaluation/build_search_eval.py)다.
H&M 사진은 사람 없이 옷만 찍은 상품 사진이라, 사용자가 쇼핑몰 캡처를 올린 경우에 해당한다.
사람이 입은 스타일 사진에서의 검색 정확도는 착용 사진 ↔ 상품 사진 쌍 데이터(DeepFashion2, Dress Code 등)로 따로 재야 한다.

같은 쿼리로 비교
- whole          : 지금 이미지 검색 (/api/search, 사진 전체, 전체 상품)
- style_search   : 스타일 사진 검색 (/api/search/style, 텍스트 없음). 상품 사진으로 판단하면 검출 없이 사진 전체로 검색
- forced_detect  : (참고) 상품 사진에도 옷 검출을 그대로 쓰면: 가장 큰 박스를 잘라 그 부위 안에서 검색
                   → 검출 모델이 착용 사진으로 학습돼 상품 사진의 종류를 자주 틀려서, 사진 종류 판단이 필요한 이유

결과: artifacts/metrics/style_search_<모델>_<split>[_<정답 기준>].json
      (정답 기준이 기존 기준(design)이 아니면 파일 이름 끝에 붙여서 기존 결과를 덮어쓰지 않는다)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from typing import Any

import pandas as pd

from src.common.config import ROOT, load_config
from src.evaluation.eval_search import relevance_tag, score
from src.search.catalog import model_slug, with_overrides
from src.search.detector import FashionDetector, box_area, crop
from src.search.encoder import load_image
from src.search.searcher import Searcher
from src.search.style_search import StyleSearcher

METHODS = ("whole", "style_search", "forced_detect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="CLIP 모델 (기본: configs/search.yaml)")
    parser.add_argument("--device", default=None, help="CLIP 장치 (기본: clip.index_device)")
    parser.add_argument("--split", choices=["test", "valid", "all"], default="test")
    args = parser.parse_args()

    cfg = with_overrides(load_config("search"), args.model)
    style_cfg = load_config("style_search")
    k, min_grade = cfg["eval"]["k"], cfg["eval"]["mrr_min_grade"]
    df = pd.read_parquet(ROOT / cfg["paths"]["eval_set"])
    df = df[df["query_type"] == "image"]
    if args.split != "all":
        df = df[df["split"] == args.split]
    df = df.reset_index(drop=True)

    articles = pd.read_csv(ROOT / cfg["paths"]["articles"], usecols=["article_id", "product_type_name"],
                           dtype={"article_id": str})
    true_l3 = articles.assign(article_id=articles["article_id"].str.zfill(10)).set_index("article_id")["product_type_name"]

    searcher = Searcher(cfg, device=args.device or cfg["clip"]["index_device"])
    detector = FashionDetector(style_cfg)
    style = StyleSearcher(searcher, detector, style_cfg)
    root = ROOT / cfg["paths"]["image_root"]
    margin = style_cfg["detector"]["crop_margin"]

    ranked: dict[str, list[list[str]]] = {m: [] for m in METHODS}
    person, detected, kind_ok = [], [], []
    for path in df["query_image_path"]:
        image = load_image(root / path)
        whole_vec = searcher.encoder.encode_images([image])[0]
        whole = searcher.search_vectors(None, whole_vec, k).product_ids
        ranked["whole"].append(whole)

        res = style.search(image, None, k)
        person.append(res.is_person_photo)
        main_group = max(res.groups, key=lambda g: box_area(g.box) if g.box else float("inf"))
        ranked["style_search"].append(main_group.result.product_ids)

        dets = detector.detect(image)
        detected.append(bool(dets))
        if not dets:
            ranked["forced_detect"].append(whole)
            continue
        main_item = max(dets, key=lambda d: box_area(d.box))
        l3 = style.detected_l3(main_item)
        kind_ok.append(true_l3.get(path.split("/")[-1].removesuffix(".jpg")) in l3)
        q = searcher.encoder.encode_images([crop(image, main_item.box, margin)])[0]
        ranked["forced_detect"].append(searcher.rank_rows(q, searcher.select_rows(l3=l3), k).product_ids)

    results: dict[str, Any] = {m: score(ranked[m], df, k, min_grade) for m in METHODS}
    report = {
        "model_name": cfg["clip"]["model_name"],
        "detector": style_cfg["detector"],
        "split": args.split,
        "relevance": cfg["eval"].get("relevance", "design"),
        "eval_set": cfg["paths"]["eval_set"],
        "k": k,
        "n_queries": len(df),
        "person_photo_share": round(sum(person) / len(df), 4),  # 상품 사진인데 스타일 사진으로 잘못 본 비율
        "forced_detect": {
            "detected_share": round(sum(detected) / len(df), 4),
            "slot_accuracy": round(sum(kind_ok) / max(len(kind_ok), 1), 4),  # 검출 부위의 소분류에 실제 상품 종류가 있는 비율
        },
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "results": results,
    }
    name = f"style_search_{model_slug(cfg['clip']['model_name'])}_{args.split}{relevance_tag(cfg)}.json"
    out = ROOT / cfg["paths"]["metrics_dir"] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fd = report["forced_detect"]
    print(f"\n쿼리 {len(df):,}개 (상품 사진). 스타일 사진으로 잘못 본 비율 {report['person_photo_share']:.1%}")
    print(f"(참고) 강제 검출: 검출 {fd['detected_share']:.1%}, 부위 정확도 {fd['slot_accuracy']:.1%}")
    print(f"\n| 방법 | 쿼리 수 | MRR | NDCG@{k} | HitRate@{k} |")
    print("|---|---|---|---|---|")
    for name, r in results.items():
        print(f"| {name} | {r['n_queries']:,} | {r['mrr']:.4f} | {r[f'ndcg@{k}']:.4f} | {r[f'hit_rate@{k}']:.4f} |")
    print(f"\n[eval-style-search] 저장 → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
