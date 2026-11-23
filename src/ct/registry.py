"""Immutable model-version records and an explicitly activated pointer.

Training never promotes a model. ``python -m src.ct.registry activate VERSION``
only accepts a validated, complete version; old versions are retained for rollback.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import ROOT, load_config


class ModelRegistry:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root or ROOT / load_config("ct")["registry"]["dir"] / "registry")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.sqlite3"
        with self._connect() as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS versions (
                    version TEXT PRIMARY KEY, path TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL,
                    status TEXT NOT NULL, metrics TEXT NOT NULL, data_watermark INTEGER NOT NULL,
                    snapshot TEXT);
                CREATE TABLE IF NOT EXISTS pointers (name TEXT PRIMARY KEY, version TEXT NOT NULL);
            """)

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            with con:
                yield con
        finally:
            con.close()

    @staticmethod
    def check_version(version: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version) or version in ("latest", "active"):
            raise ValueError("Use a unique model version name, not a path or pointer")

    def register(self, version: str, path: str | Path, status: str = "trained",
                 metrics: dict[str, Any] | None = None, data_watermark: int = 0,
                 snapshot: str | Path | None = None) -> None:
        self.check_version(version)
        if status not in {"trained", "failed"}:
            raise ValueError("Register as trained or failed; validation is a separate step")
        with self._connect() as con:
            con.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?)", (
                version, str(Path(path).resolve()), datetime.now(timezone.utc).isoformat(), status,
                json.dumps(metrics or {}, allow_nan=False), int(data_watermark), str(snapshot) if snapshot else None
            ))

    def list_versions(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM versions ORDER BY created_at DESC").fetchall()
            active = con.execute("SELECT version FROM pointers WHERE name='active'").fetchone()
        return [{**dict(row), "metrics": json.loads(row["metrics"]),
                 "active": bool(active and active[0] == row["version"])} for row in rows]

    def validate(self, version: str, metrics: dict[str, float],
                 thresholds: dict[str, float] | None = None) -> bool:
        limits = thresholds if thresholds is not None else load_config("ct")["registry"]["validation"]
        passed = bool(limits) and all(
            key in metrics and math.isfinite(float(metrics[key])) and float(metrics[key]) >= threshold
            for key, threshold in limits.items()
        )
        with self._connect() as con:
            row = con.execute("SELECT * FROM versions WHERE version=?", (version,)).fetchone()
            if row is None or row["status"] == "failed":
                raise ValueError("Model version has not trained successfully")
            con.execute("UPDATE versions SET status=?,metrics=? WHERE version=?", (
                "validated" if passed else "rejected", json.dumps(metrics, allow_nan=False), version
            ))
        return passed

    def activate(self, version: str) -> Path:
        with self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute("SELECT * FROM versions WHERE version=?", (version,)).fetchone()
            if row is None or row["status"] != "validated":
                raise ValueError("Only an explicitly validated model can be activated")
            path = Path(row["path"])
            cfg = json.loads((path / "config.json").read_text())
            required = ["two_tower.pt", "items.faiss", "index_ids.json", "item_vectors.npy",
                        "features/metadata.json", f"ranking_{cfg['ranking'].get('objective', 'cvr')}.pt"]
            if any(not (path / name).is_file() for name in required):
                raise ValueError("Model bundle is incomplete")
            con.execute("INSERT INTO pointers VALUES ('active',?) ON CONFLICT(name) DO UPDATE SET version=excluded.version",
                        (version,))
        return path

    def resolve_active(self) -> Path | None:
        with self._connect() as con:
            row = con.execute("SELECT v.path FROM pointers p JOIN versions v ON p.version=v.version "
                              "WHERE p.name='active' AND v.status='validated'").fetchone()
        return Path(row[0]) if row else None

    def retention_candidates(self, keep_last: int = 5) -> list[str]:
        """List inactive old versions for review; never delete a model automatically."""
        rows = self.list_versions()
        return [row["version"] for row in rows[max(1, keep_last):] if not row["active"]]


def resolve_model_version(version: str = "latest") -> Path | None:
    if version not in {"latest", "active"}:
        return None
    path = ROOT / load_config("ct")["registry"]["dir"] / "registry" / "registry.sqlite3"
    if not path.is_file():
        return None
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT v.path FROM pointers p JOIN versions v ON p.version=v.version "
                          "WHERE p.name='active' AND v.status='validated'").fetchone()
        return Path(row[0]) if row else None
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["list", "activate"])
    p.add_argument("version", nargs="?")
    args = p.parse_args()
    registry = ModelRegistry()
    if args.command == "activate":
        if not args.version:
            p.error("activate requires VERSION")
        print(registry.activate(args.version))
    else:
        print(json.dumps(registry.list_versions(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
