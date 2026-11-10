"""공개 Parquet 데이터만 읽는다. 학습/서빙에서 같은 시점별 피처를 사용."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEM_FIELDS = ["category_l1", "category_l2", "category_l3", "brand", "color", "pattern"]
USER_FIELDS = ["user_id", "age_group", "gender", "signup_at"]
EVENT_FIELDS = [
    "event_id",
    "user_id",
    "session_id",
    "timestamp",
    "event_type",
    "product_id",
]
AGES = ["10s", "20s", "30s", "40s", "50s", "60s+"]
GENDERS = ["F", "M"]
PROFILE_DIM = 16


def seconds(values):
    return (
        pd.to_datetime(values, utc=True)
        .astype("datetime64[ns, UTC]")
        .astype("int64")
        .to_numpy()
        / 1e9
    )


def stamp(value):
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        raise ValueError("Timestamp must include timezone")
    return t.timestamp()


def split_info(root):
    """Use the team's published partitions without repartitioning any session."""
    root = Path(root)
    meta = json.loads((root / "splits/meta.json").read_text(encoding="utf-8-sig"))
    names = ["train", "valid", "test"]
    starts = [pd.Timestamp(meta["splits"][n]["from"], tz="UTC") for n in names]
    end = pd.Timestamp(meta["splits"]["test"]["to"], tz="UTC") + pd.Timedelta(days=1)
    return meta, [t.timestamp() for t in [*starts, end]]


class FeatureStore:
    def __init__(self, root, history_len=20, metadata=None):
        root = Path(root)
        self.history_len = history_len
        self.users = pd.read_parquet(root / "users.parquet", columns=USER_FIELDS).copy()
        self.products = pd.read_parquet(root / "products.parquet").copy()
        for field in ITEM_FIELDS:
            if field not in self.products:
                self.products[field] = ""
        self.products = (
            self.products[["product_id", *ITEM_FIELDS, "price", "created_at"]]
            .sort_values("product_id")
            .reset_index(drop=True)
        )
        splits = root.parent / "splits"
        if (splits / "meta.json").exists():
            frames = []
            for i, name in enumerate(["train", "valid", "test"]):
                frame = pd.read_parquet(splits / f"{name}_events.parquet", columns=EVENT_FIELDS)
                frame["_split"] = i
                frames.append(frame)
            self.events = pd.concat(frames, ignore_index=True)
        else:
            self.events = pd.read_parquet(root / "events.parquet").copy()
            self.events = self.events[[*EVENT_FIELDS, *(["_split"] if "_split" in self.events else [])]]
        if (
            self.users.user_id.duplicated().any()
            or self.products.product_id.duplicated().any()
        ):
            raise ValueError("Duplicate IDs")
        if self.events.event_id.duplicated().any():
            raise ValueError("Duplicate event IDs")
        if not set(self.events.event_type) <= {"search", "view", "cart", "purchase"}:
            raise ValueError("Unknown event type")
        if not set(self.events.user_id) <= set(self.users.user_id):
            raise ValueError("Unknown user reference")
        actions = self.events[~self.events.event_type.eq("search")]
        if actions.product_id.isna().any() or not set(actions.product_id) <= set(
            self.products.product_id
        ):
            raise ValueError("Unknown item reference")
        self.item_ids = self.products.product_id.tolist()
        self.pmap = {s: i + 1 for i, s in enumerate(self.item_ids)}
        self.umap = {s: i for i, s in enumerate(self.users.user_id)}
        self.created = np.r_[-np.inf, seconds(self.products.created_at)]
        self.signup = seconds(self.users.signup_at)
        self.public_profiles = np.zeros((len(self.users), PROFILE_DIM), np.float32)
        for i, row in enumerate(self.users.itertuples(index=False)):
            self.public_profiles[i, 5 + (AGES.index(row.age_group) if row.age_group in AGES else len(AGES))] = 1
            self.public_profiles[i, 12 + (GENDERS.index(row.gender) if row.gender in GENDERS else len(GENDERS))] = 1
        self.events["_time"] = seconds(self.events.timestamp)
        self.events["_pid"] = (
            self.events.product_id.map(self.pmap).fillna(0).astype(int)
        )
        self.events = self.events.sort_values(
            ["_time", "event_id"], kind="stable"
        ).reset_index(drop=True)
        invalid_signup = self.events["_time"].to_numpy() < self.signup[self.events.user_id.map(self.umap)]
        self.excluded_events = self.events.loc[invalid_signup, EVENT_FIELDS].copy()
        self.audit = {
            "input_events": len(self.events),
            "pre_signup_events_excluded": int(invalid_signup.sum()),
            "affected_users": int(self.excluded_events.user_id.nunique()),
            "excluded_by_split": {str(k): int(v) for k, v in self.events.loc[invalid_signup].groupby("_split").size().items()} if "_split" in self.events else {},
        }
        self.events = self.events.loc[~invalid_signup].reset_index(drop=True)
        actions = self.events[~self.events.event_type.eq("search")]
        if (actions["_time"].to_numpy() < self.created[actions["_pid"]]).any():
            raise ValueError("Event before product creation")
        self.history = {}
        for (user, kind), g in self.events.groupby(
            ["user_id", "event_type"], sort=False
        ):
            self.history[(user, kind)] = (g["_time"].to_numpy(), g["_pid"].to_numpy())
        self.metadata = metadata
        if metadata:
            self._encode()

    def fit(self, cutoff):
        available = self.created[1:] < cutoff
        if not available.any():
            raise ValueError("No catalog at training cutoff")
        maps = {}
        for field in ITEM_FIELDS:
            values = self.products.loc[available, field].fillna("").astype(str)
            maps[field] = {s: i + 1 for i, s in enumerate(sorted(set(values) - {""}))}
        price = pd.to_numeric(self.products.price, errors="coerce").to_numpy(float)
        known = available & np.isfinite(price) & (price > 0)
        scale = float(np.percentile(np.log1p(price[known]), 95)) if known.any() else 1.0
        self.metadata = {
            "maps": maps,
            "price_scale": max(scale, 1e-8),
            "history_len": self.history_len,
            "profile_dim": PROFILE_DIM,
            "item_ids": self.item_ids,
            "fit_cutoff": cutoff,
            "objective": "cvr",
            "public_user_fields": USER_FIELDS,
            "item_fields": ITEM_FIELDS,
        }
        self._encode()
        return self

    def _encode(self):
        if self.item_ids != self.metadata["item_ids"]:
            raise ValueError("Catalog ID mapping mismatch")
        self.cats = np.zeros((len(self.products) + 1, len(ITEM_FIELDS)), np.int64)
        for j, field in enumerate(ITEM_FIELDS):
            self.cats[1:, j] = (
                self.products[field]
                .fillna("")
                .astype(str)
                .map(self.metadata["maps"][field])
                .fillna(0)
                .astype(int)
            )
        self.sizes = [len(self.metadata["maps"][f]) + 1 for f in ITEM_FIELDS]
        price = pd.to_numeric(self.products.price, errors="coerce").to_numpy(float)
        known = np.isfinite(price) & (price > 0)
        self.numeric = np.zeros((len(price) + 1, 2), np.float32)
        self.numeric[:, 1] = 1
        self.numeric[1:, 0] = np.clip(
            np.log1p(np.where(known, price, 0)) / self.metadata["price_scale"], 0, 5
        )
        self.numeric[1:, 1] = ~known

    def state(self, user, at):
        h = np.zeros(self.history_len, np.int64)
        p = np.zeros(PROFILE_DIM, np.float32)
        p[2] = 1
        ui = self.umap.get(user)
        if ui is None or self.signup[ui] > at:
            return h, 1, p
        p = self.public_profiles[ui].copy()
        history_at = at
        if self.metadata and self.metadata.get("history_policy") == "frozen_split":
            train_end, valid_end = self.metadata["history_cutoffs"]
            history_at = min(at, train_end if at < valid_end else valid_end)
        times, items = self.history.get(
            (user, "view"), (np.array([]), np.array([], dtype=int))
        )
        e = np.searchsorted(times, history_at, side="left")
        seq = items[max(0, e - self.history_len) : e]
        h[: len(seq)] = seq
        p[:3] = [
            min(np.log1p(e) / 10, 1),
            len(set(seq)) / self.history_len,
            min(np.log1p(at - times[e - 1]) / 15, 1) if e else 1,
        ]
        for j, kind in [(3, "cart"), (4, "purchase")]:
            ts, _ = self.history.get((user, kind), (np.array([]), None))
            p[j] = min(np.log1p(np.searchsorted(ts, history_at, side="left")) / 5, 1)
        p[-1] = min(np.log1p(max(0, at - self.signup[ui]) / 86400) / 10, 1)
        return h, max(1, len(seq)), p

    def frozen_time(self, times):
        """기록을 읽을 기준 시각 (state와 같은 frozen_split 규칙): valid 시각 → train 끝, test 시각 → valid 끝."""
        times = np.asarray(times, dtype=float)
        if not self.metadata or self.metadata.get("history_policy") != "frozen_split":
            return times
        train_end, valid_end = self.metadata["history_cutoffs"]
        return np.minimum(times, np.where(times < valid_end, train_end, valid_end))

    def popularity(self, targets, times, days=7):
        """상품의 최근 days일 조회 수 (기록 기준 시각의 전날까지, 하루 단위) → min(log1p(수) / 8, 1)."""
        if getattr(self, "_view_cum", None) is None:
            views = self.events[self.events.event_type.eq("view")]
            day = np.floor(views["_time"].to_numpy() / 86400).astype(np.int64)
            self._day0 = int(day.min()) if len(day) else 0
            n_days = int(day.max() - self._day0 + 1) if len(day) else 1
            daily = np.zeros((len(self.item_ids) + 1, n_days), np.int32)
            np.add.at(daily, (views["_pid"].to_numpy(), day - self._day0), 1)
            self._view_cum = np.concatenate(
                [np.zeros((len(daily), 1), np.int32), np.cumsum(daily, 1, dtype=np.int32)], 1
            )
        d = np.floor(self.frozen_time(times) / 86400).astype(np.int64) - self._day0
        n = self._view_cum.shape[1] - 1
        hi, lo = np.clip(d, 0, n), np.clip(d - days, 0, n)
        counts = self._view_cum[targets, hi] - self._view_cum[targets, lo]
        return np.minimum(np.log1p(counts) / 8, 1).astype(np.float32)

    def item_age(self, targets, times):
        """등록 후 경과일 → (min(log1p(일) / log1p(365), 1), 등록 7일 이내 여부)."""
        days = np.maximum(np.asarray(times, dtype=float) - self.created[targets], 0) / 86400
        age = np.minimum(np.log1p(days) / np.log1p(365), 1).astype(np.float32)
        return age, (days < 7).astype(np.float32)

    @staticmethod
    def append_recent(h, length, recent):
        """state의 조회 이력(h: 앞에서부터 채운 상품 번호, 뒤는 0) 뒤에 최근 조회 번호를 붙인다.

        저장된 이력은 frozen_split 규칙으로 잘려 있어서, 서빙·평가 시점의 새 조회는 여기로 더한다.
        길이는 len(h)를 넘지 않게 오래된 것부터 버린다. 0(모르는 상품)은 무시. → (h, length)
        """
        recent = np.asarray([i for i in recent if i > 0], dtype=np.int64)
        if not len(recent):
            return h, length
        seq = np.concatenate([h[h > 0], recent])[-len(h):]
        out = np.zeros_like(h)
        out[: len(seq)] = seq
        return out, max(1, len(seq))

    def save(self, folder):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        self.products.to_parquet(folder / "products.parquet", index=False)
        self.users.to_parquet(folder / "users.parquet", index=False)
        history = self.events
        if self.metadata.get("history_policy") == "frozen_split":
            history = history[history["_time"] < self.metadata["history_cutoffs"][1]]
            if "_split" in history:
                history = history[history["_split"] < 2]
        history[[*EVENT_FIELDS, *(["_split"] if "_split" in history else [])]].to_parquet(
            folder / "events.parquet", index=False
        )
        (folder / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, folder):
        folder = Path(folder)
        meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        return cls(folder, meta["history_len"], meta)


def purchase_labels(views, purchases, window):
    v = views[["user_id", "product_id", "_time"]].copy()
    v["row"] = np.arange(len(v))
    label = np.zeros(len(v), np.float32)
    if purchases.empty or views.empty:
        return label
    match = pd.merge_asof(
        purchases[["user_id", "product_id", "_time"]].sort_values("_time"),
        v.rename(columns={"_time": "view_time"}).sort_values("view_time"),
        left_on="_time",
        right_on="view_time",
        by=["user_id", "product_id"],
        tolerance=float(window),
        direction="backward",
        allow_exact_matches=False,
    )
    label[match["row"].dropna().astype(int).unique()] = 1
    return label


def prepare_examples(store, root, window):
    root = Path(root)
    _, cuts = split_info(root)
    views = store.events[store.events.event_type.eq("view")].reset_index(drop=True)
    split = views["_split"].to_numpy(dtype=np.int8).copy()
    time = views["_time"].to_numpy()
    # Preserve official session membership; exclude spillover views from fitting/selection.
    for i in [0, 1]:
        split[(split == i) & (time >= cuts[i + 1])] = -1
    labels = np.zeros(len(views), np.float32)
    purchases = store.events[store.events.event_type.eq("purchase")]
    for i in [0, 1, 2]:
        ix = np.flatnonzero(split == i)
        eligible = purchases[(purchases["_split"] == i) & (purchases["_time"] < cuts[i + 1])]
        labels[ix] = purchase_labels(views.iloc[ix], eligible, window)
    states = [
        store.state(u, t)
        for u, t in views[["user_id", "_time"]].itertuples(index=False, name=None)
    ]
    if not states:
        raise ValueError("No views")
    time = views["_time"].to_numpy()
    cut = np.array(cuts[1:])[np.maximum(split, 0)]
    return {
        "history": np.stack([s[0] for s in states]),
        "length": np.array([s[1] for s in states]),
        "profile": np.stack([s[2] for s in states]),
        "time": time,
        "target": views["_pid"].to_numpy(),
        "user": views.user_id.to_numpy(dtype=str),
        "label": labels,
        "split": split,
        "mature": (split >= 0) & (time + window < cut),
    }


def prepare_impression_examples(store, root, negatives, rng, purchase_window=None):
    """Ranking 학습 예시 — 명세 평가 프로토콜 "Ranking: 노출 후 비클릭 implicit negative".

    노출 목록(data/splits/*_impressions.parquet)마다 클릭한 상품 전부 + 같은 목록에서 보여줬지만 클릭하지 않은
    상품 중 클릭 1개당 negatives개(무작위). 클릭이 없는 목록은 비교할 대상이 없어서 쓰지 않는다.
    라벨
    - purchase_window 없음 (ctr)  : 클릭 = 1, 보여줬지만 안 누름 = 0
    - purchase_window 있음 (ctcvr): 노출 후 구매 = 1 (그 목록에서 누른 상품을 window초 안에 구매, 같은 구간의
      구매를 가장 가까운 앞선 클릭 노출 하나에만 연결), 나머지 = 0 (안 누름 + 눌렀지만 안 삼)
      → 구매 확률을 "본 상품"이 아니라 "보여준 상품 전체"에서 학습 (서빙의 후보처럼 아직 안 본 상품도 포함)
    사용자 상태는 노출 시각 기준 (조회 예시와 같은 frozen_split 규칙), 구간 경계를 넘긴 노출은 뺀다.
    반환 키는 prepare_examples와 같다 (mature: ctcvr은 구매를 끝까지 관찰할 수 있는 노출만).
    """
    root = Path(root)
    _, cuts = split_info(root)
    frames = []
    for i, name in enumerate(["train", "valid", "test"]):
        imp = pd.read_parquet(root / "splits" / f"{name}_impressions.parquet",
                              columns=["user_id", "timestamp", "product_ids", "clicked_ids"])
        imp["_split"] = i
        frames.append(imp)
    imp = pd.concat(frames, ignore_index=True)
    times = seconds(imp["timestamp"])
    split = imp["_split"].to_numpy(dtype=np.int8)
    rows, targets, labels, item_names = [], [], [], []
    for k, (shown, clicked) in enumerate(zip(imp["product_ids"], imp["clicked_ids"])):
        if len(clicked) == 0 or (split[k] < 2 and times[k] >= cuts[split[k] + 1]):
            continue
        pos = [p for p in dict.fromkeys(clicked) if p in store.pmap]  # 목록 순서 그대로 (set 순서는 실행마다 달라짐)
        clicked = set(clicked)
        pool = [p for p in shown if p not in clicked and p in store.pmap]
        neg = [pool[j] for j in rng.permutation(len(pool))[: negatives * len(pos)]]
        rows += [k] * (len(pos) + len(neg))
        item_names += pos + neg
        labels += [1.0] * len(pos) + [0.0] * len(neg)
    rows = np.asarray(rows, dtype=np.int64)
    targets = [store.pmap[p] for p in item_names]
    labels = np.asarray(labels, dtype=np.float32)
    clicked = labels.astype(bool)  # 그 목록에서 누른 상품 (ctcvr 라벨을 바꿔도 남겨 둠)
    mature = np.ones(len(rows), dtype=bool)
    if purchase_window is not None:  # 클릭 = 1 → 노출 후 구매 = 1
        clicks = np.flatnonzero(labels == 1)
        users = imp["user_id"].to_numpy()
        frame = pd.DataFrame({"user_id": users[rows[clicks]], "product_id": np.asarray(item_names)[clicks],
                              "_time": times[rows[clicks]], "_split": split[rows[clicks]]})
        # 구매는 분할 파일에서 읽는다 (모델 묶음에 저장된 store.events는 frozen_split 규칙으로 test가 빠져 있음)
        purchases = []
        for i, name in enumerate(["train", "valid", "test"]):
            ev = pd.read_parquet(root / "splits" / f"{name}_events.parquet",
                                 columns=["user_id", "product_id", "timestamp", "event_type"])
            ev = ev[ev["event_type"] == "purchase"]
            purchases.append(ev.assign(_time=seconds(ev["timestamp"]), _split=i))
        purchases = pd.concat(purchases, ignore_index=True)
        bought = np.zeros(len(clicks), np.float32)
        for i in [0, 1, 2]:
            ix = np.flatnonzero(frame["_split"].to_numpy() == i)
            eligible = purchases[(purchases["_split"] == i) & (purchases["_time"] < cuts[i + 1])]
            bought[ix] = purchase_labels(frame.iloc[ix], eligible, purchase_window)
        labels[:] = 0.0
        labels[clicks] = bought
        mature = times[rows] + purchase_window < np.array(cuts[1:])[split[rows]]
    keep = np.unique(rows)
    states = {k: store.state(u, t) for k, u, t in zip(keep, imp["user_id"].to_numpy()[keep], times[keep])}
    return {
        "history": np.stack([states[k][0] for k in rows]),
        "length": np.array([states[k][1] for k in rows]),
        "profile": np.stack([states[k][2] for k in rows]),
        "time": times[rows],
        "target": np.asarray(targets, dtype=np.int64),
        "user": imp["user_id"].to_numpy(dtype=str)[rows],
        "label": labels,
        "clicked": clicked,
        "split": split[rows],
        "mature": mature,
    }


RANK_FEATURES_V2 = ["same_l2", "same_l3", "same_brand", "same_color", "two_tower", "popular_7d", "age", "is_new"]


def rank_features(store, targets, history, profile, times, version=1, affinity=None):
    """Stage 2 입력 (sparse, dense). 명세: User / Item / Cross / Context 피처.

    version 1 (기존): 상품 범주·브랜드·색·무늬, 시간대·요일 (sparse) + 사용자 프로필, 가격, 이력과의 같은 상품·
                     같은 대분류 비율, 가격 차이 (dense)
    version 2: + RANK_FEATURES_V2 (configs/recommend.yaml ranking.features)
        Cross  : 이력과 같은 중분류 / 소분류 / 브랜드 / 색 비율, Two-Tower 취향 점수(affinity = 사용자·상품 벡터 내적)
        Item   : 최근 7일 조회 수(popular_7d), 등록 후 경과일(age), 등록 7일 이내(is_new)
    """
    targets = np.asarray(targets)
    dates = pd.to_datetime(times, unit="s", utc=True)
    sparse = np.column_stack(
        [store.cats[targets], dates.hour // 4, dates.dayofweek]
    ).astype(np.int64)
    valid = history > 0
    denom = np.maximum(valid.sum(1), 1)
    repeat = ((history == targets[:, None]) & valid).sum(1) / denom
    category = (
        (store.cats[history, 0] == store.cats[targets, 0, None])
        & valid
        & (store.cats[targets, 0, None] > 0)
    ).sum(1) / denom
    known = valid & (store.numeric[history, 1] == 0)
    mean = (store.numeric[history, 0] * known).sum(1) / np.maximum(known.sum(1), 1)
    gap = np.where(
        (known.sum(1) > 0) & (store.numeric[targets, 1] == 0),
        store.numeric[targets, 0] - mean,
        0,
    )
    dense = [profile, store.numeric[targets], repeat, category, gap]
    if version >= 2:
        if affinity is None:
            raise ValueError("ranking features v2 needs Two-Tower affinity")
        for j in (1, 2, 3, 4):  # ITEM_FIELDS: category_l2, category_l3, brand, color
            same = (store.cats[history, j] == store.cats[targets, j, None]) & valid & (store.cats[targets, j, None] > 0)
            dense.append(same.sum(1) / denom)
        age, is_new = store.item_age(targets, times)
        dense += [np.asarray(affinity, dtype=np.float32), store.popularity(targets, times), age, is_new]
    return sparse, np.column_stack(dense).astype(np.float32)
