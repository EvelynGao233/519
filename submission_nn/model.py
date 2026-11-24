"""fastText-style neural classifier in PyTorch.

headline -> hashed sub-word features (word 1-2 grams, char 3-5 grams, length token)
         -> EmbeddingBag(mean) -> Linear -> ReLU -> Dropout -> Linear -> 2 logits

All shapes are fixed here, so the backend can build Model() with no arguments and load
model.pt as a state dict. The embedding table may be stored as float16 in model.pt;
load_state_dict casts it back to float32.
"""
from __future__ import annotations

import re
import zlib
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch import nn

N_BUCKETS = 2 ** 20
EMB_DIM = 64
HIDDEN = 128
LABELS = ["FoxNews", "NBC"]


def _h(s: str) -> int:
    return zlib.crc32(s.encode("utf-8")) % N_BUCKETS


def token_ids(text: str) -> List[int]:
    words = re.sub(r"[^a-z' ]+", " ", str(text).lower()).split()
    ids = [_h("w" + w) for w in words]
    ids += [_h("b" + a + " " + b) for a, b in zip(words, words[1:])]
    for w in words:
        w = f"<{w}>"
        for n in (3, 4, 5):
            ids += [_h("c" + w[i : i + n]) for i in range(len(w) - n + 1)]
    ids.append(_h(f"len{min(len(words), 20)}"))
    return ids


def batch_inputs(texts: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    flat: List[int] = []
    offsets: List[int] = []
    for t in texts:
        offsets.append(len(flat))
        flat.extend(token_ids(t))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(N_BUCKETS, EMB_DIM, mode="mean", sparse=True)
        self.mlp = nn.Sequential(
            nn.Linear(EMB_DIM, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN, 2),
        )

    def forward_ids(self, flat: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(flat, offsets))

    def forward(self, batch) -> torch.Tensor:
        flat, offsets = batch_inputs(list(batch))
        return self.forward_ids(flat, offsets)

    @torch.no_grad()
    def proba_nbc(self, texts: List[str], bs: int = 1024) -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(texts), bs):
            out.append(torch.softmax(self.forward(texts[i : i + bs]), -1)[:, 1].numpy())
        return np.concatenate(out) if out else np.zeros(0)

    @torch.no_grad()
    def predict(self, batch) -> List[str]:
        self.eval()
        logits = self.forward(list(batch))
        return [LABELS[int(i)] for i in logits.argmax(-1)]


def get_model() -> Model:
    return Model()
