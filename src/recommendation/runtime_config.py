"""저장한 모델 설정을 우선하고, 레거시 번들의 누락 키만 현재 기본값으로 채운다."""

from copy import deepcopy

from src.common.config import ROOT, load_config


def merge_config(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve_runtime_config(bundle_config: dict | None = None) -> dict:
    """학습/실험의 명시 설정은 전역 설정 변경에 영향받지 않는다. 입력은 변경하지 않는다."""
    return merge_config(load_config("recommend"), bundle_config or {})


def serving_version(cfg: dict) -> str:
    """서빙 묶음 이름. refit이 켜져 있고 재학습 묶음(<version><suffix>)이 있으면 그것, 아니면 version (선택용 묶음)."""
    version = cfg.get("version", "stage12-cvr-v1")
    refit = cfg.get("refit") or {}
    if refit.get("enabled"):
        candidate = version + refit.get("suffix", "-refit")
        if (ROOT / cfg["paths"]["model_dir"] / candidate / "config.json").exists():
            return candidate
    return version
