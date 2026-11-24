"""Leaderboard preprocessing for Project B (News Source Classification).

prepare_data(csv_path) -> (X, y)
  X: list[str]  one lowercased headline string per CSV row, same order, no rows dropped
  y: list[str]  'FoxNews' / 'NBC' from the host name, or '' when the link has no host
                (the test CSV only has '/some-headline', so y is all '' there)

Full training links are reduced exactly like the leaderboard does (last path
segment, no extension, words containing digits removed), so train == test format.
"""
from __future__ import annotations

import csv
import re
from typing import List, Tuple
from urllib.parse import unquote, urlparse


def reduce_url(url: str) -> str:
    parsed = urlparse(url.strip())
    segments = [s for s in parsed.path.split("/") if s]
    last = unquote(segments[-1]) if segments else ""
    last = re.sub(r"\.(html?|shtml|php)$", "", last, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[-_]+", last) if w and not re.search(r"\d", w)]
    return "/" + "-".join(words)


def slug_to_text(slug: str) -> str:
    return " ".join(w for w in re.split(r"[-_/\s]+", slug.lower()) if w)


def _label(url: str) -> str:
    host = urlparse(url.strip()).netloc.lower()
    if "foxnews" in host:
        return "FoxNews"
    if "nbcnews" in host:
        return "NBC"
    return ""


def prepare_data(csv_path: str) -> Tuple[List[str], List[str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and rows[0][0].strip().lower() == "url":
        rows = rows[1:]
    X: List[str] = []
    y: List[str] = []
    for row in rows:
        # blank lines are not data rows (the evaluator skips them too)
        if not row or not row[0].strip():
            continue
        url = row[0]
        X.append(slug_to_text(reduce_url(url)))
        y.append(_label(url))
    return X, y
