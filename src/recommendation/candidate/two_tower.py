"""기존 데모 이식: 공개 프로필 + GRU / 상품 카테고리·속성·가격."""

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class TwoTower(nn.Module):
    def __init__(self, cats, numeric, sizes, profile_dim, dim=64):
        super().__init__()
        self.register_buffer("cats", torch.as_tensor(cats, dtype=torch.long))
        self.register_buffer("numeric", torch.as_tensor(numeric, dtype=torch.float32))
        self.identity = nn.Embedding(len(cats), dim, padding_idx=0)
        self.attributes = nn.ModuleList(
            [nn.Embedding(n, 8, padding_idx=0) for n in sizes]
        )
        self.item_net = nn.Sequential(
            nn.Linear(dim + 8 * len(sizes) + 2, 64), nn.ReLU(), nn.Linear(64, dim)
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


def make_model(store, cfg):
    return TwoTower(
        store.cats,
        store.numeric,
        store.sizes,
        store.metadata["profile_dim"],
        cfg["candidate"]["embedding_dim"],
    )
