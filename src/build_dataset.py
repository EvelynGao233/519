"""Merge provided + scraped links into one clean dataset.

Input:  data/raw/url_train.csv, data/scraped/{fox,nbc}_links.csv (whichever exist)
Output: data/processed/all_links.csv  columns: url, label, source, date, headline
        (headline = the leaderboard-style reduced link, e.g. '/some-headline')

Cleaning:
  1. keep only foxnews.com / nbcnews.com links; label from host
  2. reduce each link exactly like the leaderboard (textutil.reduce_url)
  3. drop links whose reduced headline is empty
  4. dedupe on (headline, label); provided rows win over scraped duplicates
  5. drop headlines that occur under BOTH labels (ambiguous, e.g. '/video')
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from textutil import label_from_url, reduce_url  # noqa: E402


def main():
    frames = []
    prov = pd.read_csv(ROOT / "data/raw/url_train.csv")
    prov["source"], prov["date"] = "provided", ""
    frames.append(prov)
    for site in ["fox", "nbc"]:
        p = ROOT / f"data/scraped/{site}_links.csv"
        if p.exists():
            s = pd.read_csv(p, dtype=str).rename(columns={"lastmod": "date"})
            s["source"] = "scraped"
            frames.append(s[["url", "source", "date"]])
            print(f"loaded {len(s):>7} scraped {site} links")
        else:
            print(f"(no {p.name} yet)")
    df = pd.concat(frames, ignore_index=True)
    n0 = len(df)
    df["label"] = df.url.map(label_from_url)
    df = df[df.label.notna()]
    df["headline"] = df.url.map(reduce_url)
    df = df[df.headline != "/"]
    n1 = len(df)
    df["prio"] = (df.source != "provided").astype(int)
    df = df.sort_values("prio").drop_duplicates(["headline", "label"]).drop(columns="prio")
    n2 = len(df)
    amb = df.groupby("headline").label.nunique()
    amb = set(amb[amb > 1].index)
    df = df[~df.headline.isin(amb)]
    print(f"\nrows {n0} -> host/empty filter {n1} -> dedupe {n2} -> drop {len(amb)} ambiguous headlines {len(df)}")
    print("\ncounts by source x label:")
    print(pd.crosstab(df.source, df.label, margins=True))

    # how much of url_train does our scrape re-discover? (proxy for test-distribution coverage)
    full = pd.concat(frames, ignore_index=True)
    full["label"] = full.url.map(label_from_url)
    full["headline"] = full.url.map(reduce_url)
    scraped = full[full.source == "scraped"]
    pv = full[full.source == "provided"]
    for lab in ["FoxNews", "NBC"]:
        have = set(scraped[scraped.label == lab].headline)
        want = set(pv[pv.label == lab].headline)
        print(f"url_train {lab:7s} headlines found in scrape: {len(want & have)}/{len(want)}")

    out = ROOT / "data/processed/all_links.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["url", "label", "source", "date", "headline"]].to_csv(out, index=False)
    print("\nwrote", out)


if __name__ == "__main__":
    main()
