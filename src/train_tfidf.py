"""Train the hashed TF-IDF + LogReg model and write submission/model.pt.

Inputs are CSVs with a `url` column (labels come from the host). If a CSV also has
a `source` column (data/processed/all_links.csv), rows with source=='provided' can be
up-weighted with --provided-weight.

Usage:
  python src/train_tfidf.py --csv data/processed/all_links.csv               # final model
  python src/train_tfidf.py --csv data/processed/all_links.csv \
      --exclude data/splits/val20.csv --eval data/splits/val20.csv --out experiments/x.pt
  python src/train_tfidf.py --csv data/raw/url_train.csv --cv --C 3 10 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
import model as M  # noqa: E402
from preprocess import _label, prepare_data, reduce_url, slug_to_text  # noqa: E402


def load(csvs, exclude=(), max_per_class=0, seed=0, since=""):
    """-> texts (object array), y (1=NBC), is_provided (bool array).

    since: keep only provided rows and scraped rows dated on/after this YYYY-MM-DD.
    """
    frames = []
    for c in csvs:
        d = pd.read_csv(c, dtype=str)
        d["provided"] = d["source"].eq("provided") if "source" in d else True
        if since and "date" in d:
            # since: 'YYYY-MM-DD' for both outlets, or 'FoxNews=YYYY-MM-DD,NBC=YYYY-MM-DD'
            if "=" in since:
                cut = dict(kv.split("=") for kv in since.split(","))
                lab = d.url.map(_label)
                thr = lab.map(lambda l: cut.get(l, ""))
            else:
                thr = since
            d = d[d.provided | (d.date.fillna("") >= thr)]
        frames.append(d[["url", "provided"]])
    d = pd.concat(frames, ignore_index=True).dropna(subset=["url"])
    d["label"] = d.url.map(_label)
    d["text"] = d.url.map(lambda u: slug_to_text(reduce_url(u)))
    d = d[(d.label != "") & (d.text != "")]
    d = d.sort_values("provided", ascending=False).drop_duplicates(["text", "label"])
    banned = set()
    for c in exclude:
        banned |= set(prepare_data(c)[0])
    if banned:
        n = len(d)
        d = d[~d.text.isin(banned)]
        print(f"excluded {n - len(d)} rows overlapping held-out data")
    d = d.sort_index()
    if max_per_class:
        d = d.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), max_per_class), random_state=seed))
    return d.text.values, (d.label == "NBC").astype(int).values, d.provided.values.astype(bool)


def count_matrix(texts, chunk=20000):
    """Raw hashed n-gram counts as CSR, built in chunks to bound memory."""
    parts = []
    for i in range(0, len(texts), chunk):
        part = list(texts[i : i + chunk])
        r, c, v = M.featurize(part)
        parts.append(csr_matrix((v, (r, c)), shape=(len(part), M.N_FEAT), dtype=np.float32))
    return vstack(parts).tocsr()


def tfidf(Cm, idf):
    coo = Cm.tocoo()
    v = M.weight(coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data, idf, Cm.shape[0])
    return csr_matrix((v, (coo.row, coo.col)), shape=Cm.shape)


def fit(Cm, y, C=10.0, sample_weight=None):
    df = np.bincount(Cm.indices, minlength=M.N_FEAT)
    idf = (np.log((1 + Cm.shape[0]) / (1 + df)) + 1).astype(np.float32)
    clf = LogisticRegression(C=C, max_iter=5000, solver="liblinear")
    clf.fit(tfidf(Cm, idf), y, sample_weight=sample_weight)
    m = M.Model()
    m.idf.copy_(torch.from_numpy(idf))
    m.coef.copy_(torch.from_numpy(clf.coef_[0].astype(np.float32)))
    m.bias.fill_(float(clf.intercept_[0]))
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", action="append", required=True)
    p.add_argument("--C", type=float, nargs="+", default=[10.0])
    p.add_argument("--cv", action="store_true")
    p.add_argument("--exclude", action="append", default=[], help="held-out CSV whose headlines are removed")
    p.add_argument("--max-per-class", type=int, default=0, help="downsample each class to at most N")
    p.add_argument("--provided-weight", type=float, nargs="+", default=[1.0])
    p.add_argument("--eval", help="after training, report accuracy on this labelled CSV")
    p.add_argument("--since", default="",
                   help="keep only provided + scraped rows dated >= this "
                        "('YYYY-MM-DD' or 'FoxNews=YYYY-MM-DD,NBC=YYYY-MM-DD')")
    p.add_argument("--out", default=str(ROOT / "submission/model.pt"))
    a = p.parse_args()

    t0 = time.time()
    X, y, prov = load(a.csv, a.exclude, a.max_per_class, since=a.since)
    print(f"{len(X)} examples ({prov.sum()} provided), NBC frac {y.mean():.3f}")
    Cm = count_matrix(X)
    print(f"featurized nnz={Cm.nnz/1e6:.1f}M in {time.time()-t0:.0f}s")

    if a.cv:
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
        for C in a.C:
            accs = []
            for tr, va in cv.split(X, y, X):
                m = fit(Cm[tr], y[tr], C)
                pred = (m.logits(list(X[va])) > 0).numpy().astype(int)
                accs.append((pred == y[va]).mean())
            print(f"C={C:<6} CV acc {np.mean(accs):.4f} +- {np.std(accs):.4f}")
        return

    if a.eval:
        Xe, ye_s = prepare_data(a.eval)
        ye = (np.array(ye_s) == "NBC").astype(int)
    m = None
    for w in a.provided_weight:
        sw = np.where(prov, w, 1.0) if w != 1.0 else None
        t = time.time()
        m = fit(Cm, y, a.C[0], sw)
        msg = f"provided_weight={w}: trained in {time.time()-t:.0f}s"
        if a.eval:
            pred = (m.logits(list(Xe)) > 0).numpy().astype(int)
            msg += f", accuracy on {Path(a.eval).name}: {(pred == ye).mean():.4f} (n={len(ye)})"
        print(msg, flush=True)
    torch.save(m.state_dict(), a.out)
    print("saved", a.out, "(last weight setting)")


if __name__ == "__main__":
    main()
