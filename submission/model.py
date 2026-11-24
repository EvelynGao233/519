"""Hashed TF-IDF (word 1-3 grams + char_wb 1-5 grams) + logistic regression, in PyTorch.

Every tensor has a fixed shape known at construction time, so the backend can
instantiate Model() with no arguments and then load model.pt as a state dict.

The featurizer (`featurize`) is shared by training (train_tfidf.py) and inference,
so the two can never drift apart.
"""
from __future__ import annotations

import re
import zlib
from typing import Iterable, List

import numpy as np
import torch
from torch import nn

N_WORD = 2 ** 20
N_CHAR = 2 ** 20
N_FEAT = N_WORD + N_CHAR
LABELS = ["FoxNews", "NBC"]


def _h(s: str, mod: int) -> int:
    return zlib.crc32(s.encode("utf-8")) % mod


def _word_ngrams(words: List[str], n_max: int = 3) -> List[str]:
    out = []
    for n in range(1, n_max + 1):
        for i in range(len(words) - n + 1):
            out.append(" ".join(words[i : i + n]))
    out.append(f"__len{min(len(words), 20)}")  # headline length bucket
    return out


def _char_ngrams(words: List[str], n_min: int = 1, n_max: int = 5) -> List[str]:
    out = []
    for w in words:
        w = f" {w} "
        for n in range(n_min, n_max + 1):
            for i in range(len(w) - n + 1):
                out.append(w[i : i + n])
    return out


def featurize(texts: Iterable[str]):
    """texts -> (rows, cols, counts) COO triplets over N_FEAT hashed buckets.

    Returns raw term counts; tf/idf weighting is applied in `weight`.
    """
    rows, cols, vals = [], [], []
    for r, t in enumerate(texts):
        words = re.sub(r"[^a-z' ]+", " ", str(t).lower()).split()
        counts: dict = {}
        for g in _word_ngrams(words):
            k = _h("w" + g, N_WORD)
            counts[k] = counts.get(k, 0) + 1
        for g in _char_ngrams(words):
            k = N_WORD + _h("c" + g, N_CHAR)
            counts[k] = counts.get(k, 0) + 1
        for k, c in counts.items():
            rows.append(r)
            cols.append(k)
            vals.append(c)
    return np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64), np.array(vals, dtype=np.float32)


def weight(rows, cols, counts, idf: np.ndarray, n_rows: int):
    """Sublinear tf * idf, then L2-normalise the word block and char block separately."""
    v = (1.0 + np.log(counts)) * idf[cols]
    is_char = cols >= N_WORD
    seg = rows * 2 + is_char  # (row, block) id
    sq = np.bincount(seg, weights=v * v, minlength=n_rows * 2)
    norm = np.sqrt(sq[seg])
    v = np.where(norm > 0, v / np.maximum(norm, 1e-12), 0.0)
    return v.astype(np.float32)


class Model(nn.Module):
    """Hashed TF-IDF + logistic regression. No lookup of training headlines."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("idf", torch.ones(N_FEAT))
        self.register_buffer("coef", torch.zeros(N_FEAT))
        self.register_buffer("bias", torch.zeros(1))

    def logits(self, texts: List[str]) -> torch.Tensor:
        """Decision value for each text (>0 means NBC)."""
        rows, cols, counts = featurize(texts)
        v = weight(rows, cols, counts, self.idf.numpy(), len(texts))
        score = np.bincount(rows, weights=v * self.coef.numpy()[cols], minlength=len(texts))
        return torch.from_numpy(score).float() + self.bias

    def forward(self, batch) -> torch.Tensor:
        s = self.logits(list(batch))
        return torch.stack([-s / 2, s / 2], dim=-1)

    @torch.no_grad()
    def predict(self, batch) -> List[str]:
        s = self.logits(list(batch))
        return [LABELS[int(x > 0)] for x in s]


def get_model() -> Model:
    return Model()
