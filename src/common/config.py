"""configs/*.yaml + .env 로더. 하이퍼파라미터/경로는 반드시 이걸로 읽는다."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

load_dotenv(ROOT / ".env")


@lru_cache
def load_config(name: str) -> dict[str, Any]:
    """load_config("recommend") → configs/recommend.yaml 내용."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def env(key: str, default: str | None = None) -> str | None:
    return os.getenv(key, default)


def data_dir() -> Path:
    return ROOT / (env("DATA_DIR", "./data") or "./data")


def artifact_dir() -> Path:
    return ROOT / (env("ARTIFACT_DIR", "./artifacts") or "./artifacts")
