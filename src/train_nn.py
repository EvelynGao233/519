"""Train the fastText-style neural model (submission_nn/model.py) and save model.pt.

  python src/train_nn.py --csv data/processed/all_links.csv \
      --since "FoxNews=2022-01-01,NBC=2015-01-01" \
      --exclude data/splits/val20.csv --eval data/splits/val20.csv \
      --out submission_nn/model.pt

Honest protocol: --exclude removes every val20 headline from training, so the val20
score is accuracy on headlines the model has never seen. The best epoch on val20 is kept.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_tfidf import load  # noqa: E402  (shared data loading / cleaning)
from preprocess import prepare_data  # noqa: E402  (submission/preprocess.py, same as submission_nn)

spec = importlib.util.spec_from_file_location("nn_model", ROOT / "submission_nn/model.py")
NN = importlib.util.module_from_spec(spec)
spec.loader.exec_module(NN)


def encode(texts):
    """-> flat int32 ids, int64 start offsets, lengths."""
    lens = np.empty(len(texts), dtype=np.int64)
    chunks = []
    for i, t in enumerate(texts):
        ids = NN.token_ids(t)
        lens[i] = len(ids)
        chunks.append(np.asarray(ids, dtype=np.int32))
    flat = np.concatenate(chunks)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    return flat, starts, lens


def make_batch(flat, starts, lens, idx):
    parts = [flat[starts[i] : starts[i] + lens[i]] for i in idx]
    offs = np.concatenate([[0], np.cumsum(lens[idx])[:-1]])
    return torch.from_numpy(np.concatenate(parts).astype(np.int64)), torch.from_numpy(offs)


@torch.no_grad()
def evaluate(model, enc, y, bs=4096):
    model.eval()
    flat, starts, lens = enc
    correct = 0
    for i in range(0, len(y), bs):
        idx = np.arange(i, min(i + bs, len(y)))
        f, o = make_batch(flat, starts, lens, idx)
        correct += (model.forward_ids(f, o).argmax(-1).numpy() == y[idx]).sum()
    return correct / len(y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", required=True)
    p.add_argument("--since", default="")
    p.add_argument("--exclude", action="append", default=[])
    p.add_argument("--eval", required=True)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(ROOT / "submission_nn/model.pt"))
    p.add_argument("--log", default=str(ROOT / "experiments/results/nn_train.json"))
    a = p.parse_args()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    t0 = time.time()
    X, y, _ = load(a.csv, a.exclude, since=a.since)
    Xe, ye_s = prepare_data(a.eval)
    ye = (np.array(ye_s) == "NBC").astype(int)
    enc, enc_e = encode(X), encode(Xe)
    print(f"{len(X)} train / {len(ye)} eval, {len(enc[0])/1e6:.0f}M ids, encoded in {time.time()-t0:.0f}s", flush=True)

    model = NN.Model()
    sparse_opt = torch.optim.SparseAdam(list(model.embedding.parameters()), lr=a.lr)
    dense_opt = torch.optim.Adam(model.mlp.parameters(), lr=a.lr)
    steps = a.epochs * ((len(y) + a.bs - 1) // a.bs)
    lr_at = lambda s: a.lr * max(0.05, 1 - s / steps)  # linear decay
    loss_fn = nn.CrossEntropyLoss()
    yt = torch.from_numpy(y)

    best, history, step = -1.0, [], 0
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = rng.permutation(len(y))
        t = time.time()
        tot = 0.0
        for b in range(0, len(y), a.bs):
            idx = perm[b : b + a.bs]
            f, o = make_batch(*enc, idx)
            for opt in (sparse_opt, dense_opt):
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                opt.zero_grad()
            loss = loss_fn(model.forward_ids(f, o), yt[idx])
            loss.backward()
            sparse_opt.step()
            dense_opt.step()
            tot += loss.item() * len(idx)
            step += 1
        acc = evaluate(model, enc_e, ye)
        history.append({"epoch": ep, "train_loss": tot / len(y), "val_acc": float(acc)})
        print(f"epoch {ep}: train loss {tot/len(y):.4f}, val acc {acc:.4f} ({time.time()-t:.0f}s)", flush=True)
        if acc > best:
            best = acc
            sd = {k: (v.half() if k.startswith("embedding") else v) for k, v in model.state_dict().items()}
            torch.save(sd, a.out)
    json.dump({"args": vars(a), "history": history, "best_val_acc": float(best)}, open(a.log, "w"), indent=1)
    print(f"best val acc {best:.4f}; saved {a.out}")


if __name__ == "__main__":
    main()
