"""Honest val20 accuracy of each model and of their probability average.

All models here were trained with every val20 headline removed. The ensemble is an
unweighted mean of P(NBC) (no weights tuned on val20, so the score stays honest).

  python experiments/ensemble_eval.py --linear experiments/tmp.pt --nn experiments/nn_lr2e-3.pt [--bert submission_bert/model.pt]
"""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_weights(model, path):
    sd = torch.load(path, map_location="cpu")
    own = model.state_dict()
    model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
    return model.eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--linear")
    p.add_argument("--nn")
    p.add_argument("--bert")
    a = p.parse_args()
    pre = load_module("pre", ROOT / "submission/preprocess.py")
    X, y = pre.prepare_data(str(ROOT / "data/splits/val20.csv"))
    y = (np.array(y) == "NBC").astype(int)

    probs = {}
    if a.linear:
        m = load_weights(load_module("lin", ROOT / "submission/model.py").Model(), a.linear)
        probs["tfidf"] = torch.sigmoid(m.logits(X)).numpy()
    if a.nn:
        m = load_weights(load_module("nnm", ROOT / "submission_nn/model.py").Model(), a.nn)
        probs["fasttext_nn"] = m.proba_nbc(X)
    if a.bert:
        m = load_weights(load_module("bert", ROOT / "submission_bert/model.py").Model(), a.bert)
        with torch.no_grad():
            probs["distilbert"] = np.concatenate([
                torch.softmax(m.forward(X[i:i + 64]), -1)[:, 1].numpy() for i in range(0, len(X), 64)])

    res = {}
    for k, v in probs.items():
        res[k] = float(((v > 0.5) == y).mean())
    names = list(probs)
    for r in range(2, len(names) + 1):
        for combo in itertools.combinations(names, r):
            avg = np.mean([probs[c] for c in combo], axis=0)
            res[" + ".join(combo)] = float(((avg > 0.5) == y).mean())
    for k, v in res.items():
        print(f"{k:40s} {v:.4f}")
    if len(names) >= 2:
        preds = {k: (v > 0.5) for k, v in probs.items()}
        for x1, x2 in itertools.combinations(names, 2):
            print(f"agreement {x1} vs {x2}: {(preds[x1] == preds[x2]).mean():.3f}")
    json.dump(res, open(ROOT / "experiments/results/ensemble_val20.json", "w"), indent=1)


if __name__ == "__main__":
    main()
