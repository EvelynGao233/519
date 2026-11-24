"""Write compact (text,label) CSVs for the Colab DistilBERT notebook.

  colab/train_honest.csv  same 637k training rows as the linear / NN models, with every
                          val20 headline removed (honest evaluation)
  colab/val20.csv         the 959 held-out links as model inputs + labels

`text` is exactly what submission/preprocess.py feeds the model at test time.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_tfidf import load  # noqa: E402
from preprocess import prepare_data  # noqa: E402

out = ROOT / "colab"
out.mkdir(exist_ok=True)
X, y, _ = load([str(ROOT / "data/processed/all_links.csv")],
               exclude=[str(ROOT / "data/splits/val20.csv")],
               since="FoxNews=2022-01-01,NBC=2015-01-01")
pd.DataFrame({"text": X, "label": y}).to_csv(out / "train_honest.csv", index=False)
Xe, ye = prepare_data(str(ROOT / "data/splits/val20.csv"))
pd.DataFrame({"text": Xe, "label": (np.array(ye) == "NBC").astype(int)}).to_csv(out / "val20.csv", index=False)
print(f"train {len(X)} (NBC {y.mean():.3f}), val {len(Xe)} -> {out}")
