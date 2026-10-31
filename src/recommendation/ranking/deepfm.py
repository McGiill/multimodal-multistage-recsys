"""Linear + FM + MLP. 이 경로는 조회 조건부 CVR만 학습한다."""

import torch
from torch import nn


class DeepFM(nn.Module):
    def __init__(self, sizes, n_dense, dim=16, hidden=(256, 128), dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(n, dim) for n in sizes])
        self.linear = nn.ModuleList([nn.Embedding(n, 1) for n in sizes])
        self.dense_factors = nn.Parameter(torch.randn(n_dense, dim) * 0.01)
        self.dense_linear = nn.Linear(n_dense, 1)
        layers = []
        width = (len(sizes) + n_dense) * dim
        for out in hidden:
            layers.extend([nn.Linear(width, out), nn.ReLU(), nn.Dropout(dropout)])
            width = out
        self.deep = nn.Sequential(*layers, nn.Linear(width, 1))
        for e in self.embeddings:
            nn.init.normal_(e.weight, std=0.01)
        for e in self.linear:
            nn.init.zeros_(e.weight)

    def forward(self, sparse, dense):
        factors = torch.cat(
            [
                torch.stack(
                    [e(sparse[:, j]) for j, e in enumerate(self.embeddings)], 1
                ),
                dense[:, :, None] * self.dense_factors[None],
            ],
            1,
        )
        fm = 0.5 * (factors.sum(1).square() - factors.square().sum(1)).sum(1)
        linear = sum(
            e(sparse[:, j]).squeeze(1) for j, e in enumerate(self.linear)
        ) + self.dense_linear(dense).squeeze(1)
        return linear + fm + self.deep(factors.flatten(1)).squeeze(1)


def make_model(store, cfg):
    r = cfg["ranking"]
    return DeepFM(
        store.sizes + [6, 7],
        store.metadata["profile_dim"] + 5,
        r["embedding_dim"],
        r["dnn_hidden"],
        r["dropout"],
    )
