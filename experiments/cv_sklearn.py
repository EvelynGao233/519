"""Quick 5-fold CV of classic models on reduced headlines (url_train.csv only).

Grouped by reduced text so duplicate headlines never sit in both train and val.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from textutil import label_from_url, reduce_url, slug_to_text  # noqa: E402

df = pd.read_csv(ROOT / "data/raw/url_train.csv")
df["y"] = df.url.map(label_from_url)
df["text"] = df.url.map(lambda u: slug_to_text(reduce_url(u)))
X, y, groups = df.text.values, (df.y == "NBC").astype(int).values, df.text.values


def word(ng=(1, 2), **kw):
    kw.setdefault("sublinear_tf", True)
    return TfidfVectorizer(ngram_range=ng, min_df=1, **kw)


def char(ng=(2, 5)):
    return TfidfVectorizer(analyzer="char_wb", ngram_range=ng, sublinear_tf=True, min_df=2)


configs = {
    "word12+LR": ([word()], lambda: LogisticRegression(C=10, max_iter=3000)),
    "word12+NB": ([word(use_idf=False, norm=None, sublinear_tf=False)], lambda: MultinomialNB(alpha=0.3)),
    "char25+LR": ([char()], lambda: LogisticRegression(C=10, max_iter=3000)),
    "word12+char25+LR": ([word(), char()], lambda: LogisticRegression(C=10, max_iter=3000)),
    "word12+char25+SVM": ([word(), char()], lambda: LinearSVC(C=0.5)),
    "word13+char15+SVM": ([word((1, 3)), char((1, 5))], lambda: LinearSVC(C=0.5)),
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for name, (vecs, mk) in configs.items():
    accs = []
    for tr, va in cv.split(X, y, groups):
        fitted = [v.fit(X[tr]) for v in vecs]
        Xtr = hstack([v.transform(X[tr]) for v in fitted]).tocsr()
        Xva = hstack([v.transform(X[va]) for v in fitted]).tocsr()
        clf = mk().fit(Xtr, y[tr])
        accs.append((clf.predict(Xva) == y[va]).mean())
    print(f"{name:22s} {np.mean(accs):.4f} +- {np.std(accs):.4f}")
