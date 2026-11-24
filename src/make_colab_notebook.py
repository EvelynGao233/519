"""Generate notebooks/finetune_distilbert.ipynb (run once; edit the cells here, not in the .ipynb)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

cells = [
("md", """# Fine-tune DistilBERT for Fox News vs NBC headline classification

**Before running:** Runtime → Change runtime type → **T4 GPU**.

Upload three files from the project when the upload cell asks:
- `colab/train_honest.csv` (637k training headlines, val20 removed)
- `colab/val20.csv` (959 held-out headlines)
- `submission_bert/model.py` (our plain-PyTorch DistilBERT, used to check the saved weights)

The notebook fine-tunes `distilbert-base-uncased`, keeps the epoch with the best val20 accuracy,
saves it as `model.pt` (float16 weights + the WordPiece vocabulary), checks that our `model.py`
reproduces the same accuracy, and downloads `model.pt` and `bert_history.json`."""),
("code", "!nvidia-smi -L"),
("code", """from google.colab import files
uploaded = files.upload()   # pick train_honest.csv, val20.csv, model.py
print(list(uploaded))"""),
("code", """import json, math, random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

SEED, MAX_LEN, BS, LR, EPOCHS = 0, 64, 256, 5e-5, 2
SUBSAMPLE = None          # e.g. 50000 for a quick dry run; None = all 637k rows
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
dev = 'cuda'

tr = pd.read_csv('train_honest.csv', keep_default_na=False)
va = pd.read_csv('val20.csv', keep_default_na=False)
if SUBSAMPLE: tr = tr.sample(SUBSAMPLE, random_state=SEED)
print(len(tr), 'train', len(va), 'val', 'NBC frac', tr.label.mean().round(3))

tok = AutoTokenizer.from_pretrained('distilbert-base-uncased')
enc = lambda texts: tok(list(texts), truncation=True, max_length=MAX_LEN)['input_ids']
t0 = time.time(); tr_ids, va_ids = enc(tr.text), enc(va.text); print(f'tokenized in {time.time()-t0:.0f}s')
tr_y, va_y = tr.label.values, va.label.values"""),
("code", """def collate(batch):
    ids, ys = zip(*batch)
    w = max(len(s) for s in ids)
    x = torch.zeros(len(ids), w, dtype=torch.long); m = torch.zeros_like(x)
    for i, s in enumerate(ids):
        x[i, :len(s)] = torch.tensor(s); m[i, :len(s)] = 1
    return x, m, torch.tensor(ys)

train_dl = DataLoader(list(zip(tr_ids, tr_y)), batch_size=BS, shuffle=True, collate_fn=collate, num_workers=2)
val_dl = DataLoader(list(zip(va_ids, va_y)), batch_size=512, shuffle=False, collate_fn=collate)

@torch.no_grad()
def evaluate(model):
    model.eval(); correct = 0
    for x, m, y in val_dl:
        with torch.autocast('cuda', dtype=torch.float16):
            correct += (model(input_ids=x.to(dev), attention_mask=m.to(dev)).logits.argmax(-1).cpu() == y).sum().item()
    model.train(); return correct / len(va_y)"""),
("code", """model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
steps = EPOCHS * len(train_dl)
sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
scaler = torch.amp.GradScaler()
history, best, best_sd = [], -1, None
evals_per_epoch = 4
every = math.ceil(len(train_dl) / evals_per_epoch)

model.train(); t0 = time.time()
for ep in range(EPOCHS):
    run = 0.0
    for i, (x, m, y) in enumerate(train_dl, 1):
        with torch.autocast('cuda', dtype=torch.float16):
            loss = model(input_ids=x.to(dev), attention_mask=m.to(dev), labels=y.to(dev)).loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        run += loss.item()
        if i % every == 0 or i == len(train_dl):
            acc = evaluate(model)
            frac = ep + i / len(train_dl)
            history.append({'epoch': round(frac, 2), 'train_loss': run / every, 'val_acc': acc})
            print(f'epoch {frac:.2f}  loss {run/every:.4f}  val20 acc {acc:.4f}  ({(time.time()-t0)/60:.1f} min)')
            run = 0.0
            if acc > best:
                best = acc
                best_sd = {k: v.detach().to('cpu', torch.float16) for k, v in model.state_dict().items()}
print('best val20 acc', best)"""),
("code", """# model.pt = best weights in float16 + the WordPiece vocabulary as one uint8 buffer
vocab = [t for t, _ in sorted(tok.vocab.items(), key=lambda kv: kv[1])]
raw = '\\n'.join(vocab).encode('utf-8')
sd = dict(best_sd)
sd['vocab_bytes'] = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
torch.save(sd, 'model.pt')
json.dump({'max_len': MAX_LEN, 'bs': BS, 'lr': LR, 'epochs': EPOCHS, 'subsample': SUBSAMPLE,
           'history': history, 'best_val_acc': best}, open('bert_history.json', 'w'), indent=1)
print(len(raw), 'vocab bytes;', round(sum(v.numel() for v in sd.values()) * 2 / 1e6), 'MB approx')"""),
("code", """# check: our plain-PyTorch model.py (what the leaderboard runs, on CPU) gives the same val20 accuracy
import importlib.util
spec = importlib.util.spec_from_file_location('bert_model', 'model.py'); B = importlib.util.module_from_spec(spec); spec.loader.exec_module(B)
ours = B.Model(); ours.load_state_dict(torch.load('model.pt'), strict=True); ours.eval()
preds = []
for i in range(0, len(va), 64):
    preds += ours.predict(list(va.text[i:i+64]))
acc = np.mean([(p == 'NBC') == bool(y) for p, y in zip(preds, va_y)])
print(f'model.py on CPU: val20 acc {acc:.4f}  (GPU fp16 during training: {best:.4f})')"""),
("code", """files.download('model.pt')
files.download('bert_history.json')"""),
]

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 0}
for kind, src in cells:
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = ROOT / "notebooks/finetune_distilbert.ipynb"
out.parent.mkdir(exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
