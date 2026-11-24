"""Same 5-fold grouped CV, but for the official hashed EmbeddingBag baseline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "starter/Proj B Template/tested baseline"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(BASE))
from textutil import label_from_url, reduce_url, slug_to_text  # noqa: E402
from model import NewsClassifier  # noqa: E402
from train_baseline import train, set_seed  # noqa: E402

df = pd.read_csv(ROOT / "data/raw/url_train.csv")
X = df.url.map(lambda u: slug_to_text(reduce_url(u))).values
y = torch.tensor((df.url.map(label_from_url) == "NBC").astype(int).values)

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
accs = []
for tr, va in cv.split(X, y.numpy(), X):
    set_seed(42)
    m = NewsClassifier()
    train(m, list(X[tr]), y[tr], list(X[va]), y[va], epochs=5)
    accs.append((np.array(m.predict(list(X[va]))) == y[va].numpy()).mean())
print(f"baseline CV: {np.mean(accs):.4f} +- {np.std(accs):.4f}")
