# News Source Classification from Headline Links (CIS 4190/5190, Track B)

Predict whether a news link comes from **Fox News** or **NBC News** from its headline words
alone (the leaderboard strips host, section and any word containing a digit, e.g.
`/biden-signs-infrastructure-bill`).

Final model: an equal-weight ensemble of three classifiers trained on 637k scraped headlines.

| Model | val20 accuracy (unseen headlines) |
| --- | --- |
| Provided baseline (hashed bag-of-words MLP) | 73.9% |
| Hashed TF-IDF + logistic regression | 88.4% |
| fastText-style network (sub-word EmbeddingBag + MLP) | 88.1% |
| DistilBERT, fine-tuned, re-implemented in plain PyTorch | 88.5% |
| **Ensemble of the three (submitted)** | **89.9%** |

Leaderboard (`url_val16k`, 8,000 links): **96.34%**. val20 is 20% of `url_train.csv` with every
one of its headlines removed from all training data, so it measures generalization; the
leaderboard is inflated by test headlines that also appear in the public sitemaps we crawled
(see the report, Section 6). The final model contains **no lookup of training headlines**.

## Repository layout

```
submission_final/   leaderboard files: model.py (3-model ensemble), preprocess.py  [model.pt: see Weights]
submission/         TF-IDF + logistic regression alone (model.py, preprocess.py, model.pt 16 MB)
submission_nn/      fastText-style network (model.py, preprocess.py)
submission_bert/    DistilBERT + WordPiece tokenizer in plain PyTorch (model.py), training history
src/
  crawl.py            polite sitemap crawler (robots.txt, 1.5 s delay, cached)
  build_dataset.py    merge, reduce exactly like the leaderboard, dedupe, drop ambiguous headlines
  textutil.py         leaderboard link reduction + labels
  train_tfidf.py      TF-IDF model (CV, ablations, final training)
  train_nn.py         fastText-style network
  export_for_colab.py writes the CSVs the DistilBERT notebook trains on
  make_colab_notebook.py  generates notebooks/finetune_distilbert.ipynb
  build_final.py      merges the three models into submission_final/ (one model.py + one model.pt)
notebooks/finetune_distilbert.ipynb   DistilBERT fine-tuning on a Colab T4
experiments/        CV comparisons, learning curve, ensemble evaluation, port verification, results/
data/raw/url_train.csv, data/splits/{train80,val20}.csv
```

## Weights and data (not in git: GitHub's 100 MB file limit)

- `submission_final/model.pt` (285 MB), `submission_nn/model.pt` (134 MB),
  `submission_bert/model.pt` (134 MB): **[Hugging Face link to add]**
- Scraped dataset (2.16M unique headlines, `data/processed/all_links.csv`): **[Hugging Face link to add]**

## Reproduce

Python 3.11, CPU is enough for everything except DistilBERT fine-tuning.

```bash
python -m venv .venv
.venv/Scripts/pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install numpy pandas scikit-learn scipy requests matplotlib transformers
```

1. Crawl and build the dataset
   ```bash
   python src/crawl.py --site fox --since 2000-01-01 --max-sitemaps 1000
   python src/crawl.py --site nbc --since 2005-01-01 --max-sitemaps 1500
   python src/build_dataset.py
   ```
2. Experiments: `experiments/cv_sklearn.py`, `cv_baseline.py`, `learning_curve.py`, `plot_learning_curve.py`
3. Train the three models (honest version: val20 excluded)
   ```bash
   python src/train_tfidf.py --csv data/processed/all_links.csv --since "FoxNews=2022-01-01,NBC=2015-01-01"
   python src/train_nn.py --csv data/processed/all_links.csv --since "FoxNews=2022-01-01,NBC=2015-01-01" --exclude data/splits/val20.csv --eval data/splits/val20.csv --lr 1e-2 --epochs 6
   python src/export_for_colab.py   # then run notebooks/finetune_distilbert.ipynb on Colab, save model.pt to submission_bert/
   ```
4. Evaluate and package
   ```bash
   python experiments/ensemble_eval.py --linear <honest tfidf model.pt> --nn submission_nn/model.pt --bert submission_bert/model.pt
   python src/build_final.py
   ```
5. Check with the course evaluator (on Windows set `PYTHONUTF8=1`)
   ```bash
   python eval_project_b.py --model submission_final/model.py --preprocess submission_final/preprocess.py --weights submission_final/model.pt --csv data/splits/val20.csv
   ```

## Report

Full write-up: **[report link]**
