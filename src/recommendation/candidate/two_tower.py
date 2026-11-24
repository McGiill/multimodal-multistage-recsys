"""공개 프로필 + GRU / 상품 카테고리·속성·가격 (+ 선택적 CLIP 이미지 특징)."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TwoTower(nn.Module):
    def __init__(self, cats, numeric, sizes, profile_dim, dim=64, image_vectors=None, image_projection_dim=32):
        super().__init__()
        self.register_buffer("cats", torch.as_tensor(cats, dtype=torch.long))
        self.register_buffer("numeric", torch.as_tensor(numeric, dtype=torch.float32))
        self.identity = nn.Embedding(len(cats), dim, padding_idx=0)
        self.attributes = nn.ModuleList(
            [nn.Embedding(n, 8, padding_idx=0) for n in sizes]
        )
        self.image_projection = None
        image_width = 0
        if image_vectors is not None:
            vectors = torch.as_tensor(image_vectors, dtype=torch.float32)
            if (vectors.ndim != 2 or vectors.shape[0] != len(cats) or vectors.shape[1] == 0
                    or not torch.isfinite(vectors).all() or torch.count_nonzero(vectors[0])):
                raise ValueError("Image vectors need one finite row per item and a zero padding row")
            if not isinstance(image_projection_dim, int) or isinstance(image_projection_dim, bool) or image_projection_dim < 1:
                raise ValueError("candidate.image_features.projection_dim must be a positive integer")
            self.register_buffer("image_vectors", vectors)
            image_width = image_projection_dim
            self.image_projection = nn.Linear(vectors.shape[1], image_width, bias=False)
        self.item_net = nn.Sequential(
            nn.Linear(dim + 8 * len(sizes) + 2 + image_width, 64), nn.ReLU(), nn.Linear(64, dim)
        )
        self.gru = nn.GRU(dim, dim, batch_first=True)
        self.user_net = nn.Sequential(
            nn.Linear(dim + profile_dim, 64), nn.ReLU(), nn.Linear(64, dim)
        )

    def item(self, ids):
        c = self.cats[ids]
        identity = self.identity(ids)
        x = torch.cat(
            [
                identity,
                *[e(c[..., j]) for j, e in enumerate(self.attributes)],
                self.numeric[ids],
                *([self.image_projection(self.image_vectors[ids])] if self.image_projection is not None else []),
            ],
            -1,
        )
        return nn.functional.normalize(self.item_net(x) + identity, dim=-1) * (
            ids != 0
        ).unsqueeze(-1)

    def user(self, profile, history, length):
        seq = self.item(history)
        packed = pack_padded_sequence(
            seq, length.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        last = seq[torch.arange(len(seq)), length - 1]
        return nn.functional.normalize(
            self.user_net(torch.cat([h[-1], profile], -1)) + last, dim=-1
        )


def make_model(store, cfg, image_vectors=None):
    image_cfg = cfg["candidate"].get("image_features", {})
    if bool(image_cfg.get("enabled", False)) != (image_vectors is not None):
        raise ValueError("Image feature configuration and checkpoint/input vectors disagree")
    return TwoTower(
        store.cats,
        store.numeric,
        store.sizes,
        store.metadata["profile_dim"],
        cfg["candidate"]["embedding_dim"],
        image_vectors=image_vectors,
        image_projection_dim=image_cfg.get("projection_dim", 32),
    )
