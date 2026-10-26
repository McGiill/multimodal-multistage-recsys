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


def rank_features(store, targets, history, profile, times):
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
    return sparse, np.column_stack(
        [profile, store.numeric[targets], repeat, category, gap]
    ).astype(np.float32)
