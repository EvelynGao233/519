"""Check submission_bert/model.py against HuggingFace DistilBERT.

1. vocab byte length matches VOCAB_BYTES
2. our WordPiece tokenizer == HF tokenizer on every val20 headline + edge cases
3. our forward pass == HF forward pass (same weights) to float tolerance
4. CPU inference time for N headlines (backend limit: 7 minutes)
"""
import importlib.util
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("bert_model", ROOT / "submission_bert/model.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
sys.path.insert(0, str(ROOT / "submission"))
from preprocess import prepare_data  # noqa: E402

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
inv = [t for t, _ in sorted(tok.vocab.items(), key=lambda kv: kv[1])]
raw = "\n".join(inv).encode("utf-8")
print("vocab bytes", len(raw), "constant", B.VOCAB_BYTES, "OK" if len(raw) == B.VOCAB_BYTES else "MISMATCH")

torch.manual_seed(0)
hf = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).eval()
sd = {k: v.half() for k, v in hf.state_dict().items()}  # stored as fp16, like the real model.pt
sd["vocab_bytes"] = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
ours = B.Model()
missing, unexpected = ours.load_state_dict(sd, strict=False)
print("missing", missing, "unexpected", unexpected)
ours.eval()
hf = hf.float()

X, _ = prepare_data(str(ROOT / "data/splits/val20.csv"))
edge = ["isn't it", "café résumé naïve", "trump’s tariffs", "qwertyuiopxyz", "", "a " * 80, "中国 news"]
texts = list(X) + edge
bad = 0
for t in texts:
    a = ours.encode([t])[0][0].tolist()
    b = tok(t, truncation=True, max_length=B.MAX_LEN)["input_ids"]
    if a != b:
        bad += 1
        if bad <= 5:
            print("TOKEN MISMATCH", repr(t[:60]), a[:12], b[:12])
print(f"tokenizer mismatches: {bad}/{len(texts)}")

# compare logits with HF using HF's fp16-rounded weights, so both sides see identical numbers
hf.load_state_dict({k: v.float() for k, v in sd.items() if k != "vocab_bytes"})
with torch.no_grad():
    ids, mask = ours.encode(texts[:64])
    a = ours.forward_ids(ids, mask)
    b = hf(input_ids=ids, attention_mask=mask).logits
print("max |logit diff|", (a - b).abs().max().item())

n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
big = (list(X) * (n // len(X) + 1))[:n]
t0 = time.time()
with torch.no_grad():
    for i in range(0, n, 32):
        ours.predict(big[i : i + 32])
dt = time.time() - t0
print(f"CPU inference: {n} headlines in {dt:.1f}s -> 8000 would take {dt * 8000 / n:.0f}s ({torch.get_num_threads()} threads)")
