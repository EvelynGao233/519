"""Learning curve: val20 accuracy vs. amount of (balanced) training data.

Two evaluation protocols on data/splits/val20.csv (20% of url_train, never trained on
as provided rows):
  honest    - every val20 headline is removed from the training pool, including the
              scraped copies, so the model is scored on headlines it never saw.
  realistic - scraped copies of val20 headlines stay in the pool (the hidden test set
              comes from the same sites, and ~92% of url_train reappears in our scrape).

Writes experiments/results/learning_curve.json and .png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submission"))
import model as M  # noqa: E402
from preprocess import prepare_data, reduce_url, slug_to_text  # noqa: E402

RES = ROOT / "experiments/results"
RES.mkdir(parents=True, exist_ok=True)


def count_matrix(texts, chunk=20000):
    """Raw hashed n-gram counts as CSR, built in chunks to bound memory."""
    parts = []
    for i in range(0, len(texts), chunk):
        r, c, v = M.featurize(texts[i : i + chunk])
        parts.append(csr_matrix((v, (r, c)), shape=(min(chunk, len(texts) - i), M.N_FEAT), dtype=np.float32))
    return vstack(parts).tocsr()


def tfidf(Csub, idf):
    coo = Csub.tocoo()
    v = M.weight(coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data, idf, Csub.shape[0])
    return csr_matrix((v, (coo.row, coo.col)), shape=Csub.shape)


def fit_eval(Ctr, ytr, Cva, yva, C=10.0, class_weight=None):
    df = np.bincount(Ctr.indices, minlength=M.N_FEAT)
    idf = (np.log((1 + Ctr.shape[0]) / (1 + df)) + 1).astype(np.float32)
    clf = LogisticRegression(C=C, max_iter=5000, solver="liblinear", class_weight=class_weight)
    clf.fit(tfidf(Ctr, idf), ytr)
    return float((clf.predict(tfidf(Cva, idf)) == yva).mean())


def main():
    t0 = time.time()
    df = pd.read_csv(ROOT / "data/processed/all_links.csv")
    val_urls = pd.read_csv(ROOT / "data/splits/val20.csv").url
    val_heads = set(val_urls.map(reduce_url))
    # provided val20 rows themselves never go into training
    df = df[~df.url.isin(set(val_urls))].reset_index(drop=True)
    df["y"] = (df.label == "NBC").astype(int)
    df["in_val"] = df.headline.isin(val_heads)
    print(f"pool {len(df)} rows; {df.in_val.sum()} scraped rows share a headline with val20")

    Xva, yva_s = prepare_data(str(ROOT / "data/splits/val20.csv"))
    yva = (np.array(yva_s) == "NBC").astype(int)
    Cva = count_matrix(Xva)

    texts = [slug_to_text(h) for h in df.headline]
    Call = count_matrix(texts)
    print(f"featurized {Call.shape} nnz={Call.nnz/1e6:.0f}M in {time.time()-t0:.0f}s")
    y = df.y.values
    rng = np.random.default_rng(0)
    perm = {k: rng.permutation(np.where(y == k)[0]) for k in (0, 1)}
    is_prov = (df.source == "provided").values

    results = []

    def run(name, n_per_class, protocol, idx=None, **kw):
        if idx is None:
            ok = ~df.in_val.values if protocol == "honest" else np.ones(len(df), bool)
            idx = np.concatenate([p[ok[p]][:n_per_class] for p in perm.values()])
        t = time.time()
        acc = fit_eval(Call[idx], y[idx], Cva, yva, **kw)
        r = dict(name=name, n_train=int(len(idx)), protocol=protocol, acc=acc, **{k: str(v) for k, v in kw.items()})
        results.append(r)
        print(f"{name:28s} {protocol:9s} n={len(idx):>7} acc={acc:.4f} ({time.time()-t:.0f}s)", flush=True)
        json.dump(results, open(RES / "learning_curve.json", "w"), indent=1)

    run("provided train80 only", 0, "honest", idx=np.where(is_prov)[0])
    for n in [1000, 2000, 5000, 10000, 30000, 100000, 250000]:
        for protocol in ["honest", "realistic"]:
            run(f"balanced {n}/class", n, protocol)
    for protocol in ["honest", "realistic"]:
        ok = ~df.in_val.values if protocol == "honest" else np.ones(len(df), bool)
        run("all (imbalanced)", 0, protocol, idx=np.where(ok)[0])
        run("all, class_weight=balanced", 0, protocol, idx=np.where(ok)[0], class_weight="balanced")
    for C in [1.0, 3.0, 30.0]:
        run(f"balanced 250000/class C={C}", 250000, "honest", C=C)
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
