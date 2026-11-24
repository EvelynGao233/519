"""Shared text utilities.

`reduce_url` reproduces EXACTLY what the leaderboard does to test links
(copied from `_hide_answers` in eval_project_b.py), so training inputs match
test inputs. `label_from_url` gives the training label from the host name.
"""
from __future__ import annotations

import re
from urllib.parse import unquote, urlparse


def reduce_url(url: str) -> str:
    """Full link -> '/headline-words-only' (no host, section, or digit words)."""
    parsed = urlparse(url.strip())
    segments = [s for s in parsed.path.split("/") if s]
    last = unquote(segments[-1]) if segments else ""
    last = re.sub(r"\.(html?|shtml|php)$", "", last, flags=re.IGNORECASE)
    words = [w for w in re.split(r"[-_]+", last) if w and not re.search(r"\d", w)]
    return "/" + "-".join(words)


def label_from_url(url: str) -> str | None:
    host = urlparse(url.strip()).netloc.lower()
    if "foxnews" in host:
        return "FoxNews"
    if "nbcnews" in host:
        return "NBC"
    return None


def slug_to_text(slug: str) -> str:
    """'/some-headline' -> 'some headline' (lowercased)."""
    return " ".join(w for w in re.split(r"[-_/\s]+", slug.lower()) if w)
