"""검색과 추천이 공유하는 고객층 범위. 회원 프로필과 독립적인 요청 조건."""

import numpy as np

from src.common.config import load_config


def audience_mask(store, audience: str | None = None) -> np.ndarray | None:
    """내부 상품 번호(0은 padding)에 맞춘 허용 마스크."""
    if audience in (None, "", "all"):
        return None
    groups = load_config("style_search")["audience"]["groups"]
    if audience not in groups:
        raise ValueError("audience must be all, women, men or kids")
    categories = store.products.set_index("product_id").loc[store.item_ids, "category_l1"]
    return np.r_[False, categories.isin(groups[audience]["category_l1"]).to_numpy()]
