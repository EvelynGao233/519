"""DistilBERT (base, uncased) sequence classifier in plain PyTorch, with its WordPiece tokenizer.

The backend has no `transformers` package, so this file re-implements the architecture
with module names identical to HuggingFace's DistilBertForSequenceClassification. A model
fine-tuned with `transformers` (see notebooks/finetune_distilbert.ipynb) is saved with
`state_dict()` plus one extra buffer, `vocab_bytes` (the WordPiece vocabulary as UTF-8
bytes, one token per line), and loads here unchanged. Weights may be stored as float16;
load_state_dict casts them back to float32.
"""
from __future__ import annotations

import math
import unicodedata
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

VOCAB_SIZE = 30522
VOCAB_BYTES = 231507  # len("\n".join(distilbert-base-uncased vocab).encode("utf-8"))
DIM, N_LAYERS, N_HEADS, HIDDEN_DIM, MAX_POS = 768, 6, 12, 3072, 512
MAX_LEN = 64  # headlines are short; [CLS] + tokens + [SEP] is truncated to this
LABELS = ["FoxNews", "NBC"]
CLS, SEP, PAD, UNK = "[CLS]", "[SEP]", "[PAD]", "[UNK]"


# ----------------------------------------------------------------------------- tokenizer
def _is_punct(ch: str) -> bool:
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def basic_tokenize(text: str) -> List[str]:
    """BERT BasicTokenizer (uncased): clean, split CJK, lowercase, strip accents, split punctuation."""
    out = []
    for ch in str(text):
        cp = ord(ch)
        if cp == 0 or cp == 0xFFFD or (unicodedata.category(ch).startswith("C") and ch not in "\t\n\r"):
            continue
        if _is_cjk(cp):
            out.append(f" {ch} ")
        elif ch.isspace():
            out.append(" ")
        else:
            out.append(ch)
    tokens = []
    for word in "".join(out).lower().split():
        word = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
        cur = ""
        for ch in word:
            if _is_punct(ch):
                if cur:
                    tokens.append(cur)
                    cur = ""
                tokens.append(ch)
            else:
                cur += ch
        if cur:
            tokens.append(cur)
    return tokens


def wordpiece(word: str, vocab: Dict[str, int]) -> List[int]:
    if len(word) > 100:
        return [vocab[UNK]]
    ids, start = [], 0
    while start < len(word):
        end, cur = len(word), None
        while start < end:
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                cur = vocab[piece]
                break
            end -= 1
        if cur is None:
            return [vocab[UNK]]
        ids.append(cur)
        start = end
    return ids


# ----------------------------------------------------------------------------- model
class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embeddings = nn.Embedding(VOCAB_SIZE, DIM, padding_idx=0)
        self.position_embeddings = nn.Embedding(MAX_POS, DIM)
        self.LayerNorm = nn.LayerNorm(DIM, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, ids):
        pos = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
        return self.dropout(self.LayerNorm(self.word_embeddings(ids) + self.position_embeddings(pos)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_lin, self.k_lin = nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)
        self.v_lin, self.out_lin = nn.Linear(DIM, DIM), nn.Linear(DIM, DIM)

    def forward(self, x, mask):
        b, t, _ = x.shape
        split = lambda z: z.view(b, t, N_HEADS, DIM // N_HEADS).transpose(1, 2)
        q, k, v = split(self.q_lin(x)), split(self.k_lin(x)), split(self.v_lin(x))
        ctx = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :].bool())
        return self.out_lin(ctx.transpose(1, 2).reshape(b, t, DIM))


class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(DIM, HIDDEN_DIM), nn.Linear(HIDDEN_DIM, DIM)

    def forward(self, x):
        return self.lin2(F.gelu(self.lin1(x)))


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = MultiHeadSelfAttention()
        self.sa_layer_norm = nn.LayerNorm(DIM, eps=1e-12)
        self.ffn = FFN()
        self.output_layer_norm = nn.LayerNorm(DIM, eps=1e-12)

    def forward(self, x, mask):
        x = self.sa_layer_norm(x + self.attention(x, mask))
        return self.output_layer_norm(x + self.ffn(x))


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList([TransformerBlock() for _ in range(N_LAYERS)])

    def forward(self, x, mask):
        for blk in self.layer:
            x = blk(x, mask)
        return x


class DistilBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = Embeddings()
        self.transformer = Transformer()

    def forward(self, ids, mask):
        return self.transformer(self.embeddings(ids), mask)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.distilbert = DistilBert()
        self.pre_classifier = nn.Linear(DIM, DIM)
        self.classifier = nn.Linear(DIM, 2)
        self.dropout = nn.Dropout(0.2)
        self.register_buffer("vocab_bytes", torch.zeros(VOCAB_BYTES, dtype=torch.uint8))
        self._vocab: Dict[str, int] | None = None

    # -- tokenizer -----------------------------------------------------------
    def vocab(self) -> Dict[str, int]:
        if self._vocab is None:
            raw = bytes(self.vocab_bytes.tolist()).decode("utf-8")
            toks = raw.split("\n")
            if len(toks) != VOCAB_SIZE:
                raise RuntimeError("vocab_bytes not loaded: load model.pt before predicting")
            self._vocab = {t: i for i, t in enumerate(toks)}
        return self._vocab

    def encode(self, texts: List[str]):
        v = self.vocab()
        seqs = []
        for t in texts:
            ids = [i for w in basic_tokenize(t) for i in wordpiece(w, v)][: MAX_LEN - 2]
            seqs.append([v[CLS]] + ids + [v[SEP]])
        width = max(len(s) for s in seqs)
        ids = torch.zeros(len(seqs), width, dtype=torch.long)
        mask = torch.zeros(len(seqs), width, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, : len(s)] = torch.tensor(s)
            mask[i, : len(s)] = 1
        return ids, mask

    # -- model ---------------------------------------------------------------
    def forward_ids(self, ids, mask):
        h = self.distilbert(ids, mask)[:, 0]
        return self.classifier(self.dropout(F.relu(self.pre_classifier(h))))

    def forward(self, batch) -> torch.Tensor:
        ids, mask = self.encode(list(batch))
        return self.forward_ids(ids, mask)

    @torch.no_grad()
    def predict(self, batch) -> List[str]:
        self.eval()
        return [LABELS[int(i)] for i in self.forward(list(batch)).argmax(-1)]


def get_model() -> Model:
    return Model()
