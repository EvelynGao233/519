"""Polite sitemap crawler for Fox News / NBC News article links.

Walks each site's public sitemap index (standard sitemaps.org XML) recursively,
collects article URLs with their <lastmod> date, and writes
data/scraped/<site>_links.csv (columns: url, lastmod, sitemap).

Politeness: obeys robots.txt for our user agent, waits --delay seconds between
requests, and caches every fetched sitemap under data/scraped/cache/ so reruns
never hit the site twice for the same file.

Examples:
  python src/crawl.py --site fox --since 2022-01-01
  python src/crawl.py --site nbc --since 2022-01-01
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import html
import re
import sys
import time
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/scraped"
CACHE = OUT / "cache"
UA = "CIS5190-course-project-crawler (educational, low-rate)"

SITES = {
    "fox": {
        # liveblogs first: url_train has ~5% /live-news/ links that are not in the article sitemaps
        "start": ["https://www.foxnews.com/sitemap.xml"],
        # only the paginated article sitemaps (skip videos, shows, personalities...).
        # (the liveblogs sitemap is empty, so url_train's ~5% /live-news/ links can't be added)
        "child_include": r"type=articles",
        "url_host": "foxnews.com",
    },
    "nbc": {
        "start": [
            "https://www.nbcnews.com/sitemap/nbcnews/sitemap-index",
            "https://www.nbcnews.com/sitemap/nbcnews/sitemap-select.xml",
            # url_train has ~5% /politics/20xx-elections/<race>-results pages
            "https://www.nbcnews.com/politics/election-results/sitemap.xml",
        ],
        "child_include": r".",
        # keep everything: url_train's NBC half has /now/, /nightly-news/, /slideshow/
        # and election-result pages.
        "url_host": "nbcnews.com",
    },
}


class Fetcher:
    def __init__(self, delay: float):
        self.delay = delay
        self.last = 0.0
        self.s = requests.Session()
        self.s.headers["User-Agent"] = UA
        self.robots: dict = {}
        self.n_net = 0

    def allowed(self, url: str) -> bool:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.robots:
            rp = robotparser.RobotFileParser(base + "/robots.txt")
            rp.read()
            self.robots[base] = rp
        return self.robots[base].can_fetch(UA, url)

    def get(self, url: str) -> str | None:
        key = hashlib.md5(url.encode()).hexdigest()
        cpath = CACHE / f"{key}.xml"
        if cpath.exists():
            return cpath.read_text(encoding="utf-8")
        if not self.allowed(url):
            print(f"  robots.txt disallows {url}, skipping")
            return None
        for attempt in range(4):
            wait = self.delay - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.time()
            try:
                r = self.s.get(url, timeout=60)
                self.n_net += 1
            except requests.RequestException as e:
                print(f"  error {e!r}, retry {attempt + 1}")
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                body = r.content
                if body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                text = body.decode("utf-8", errors="replace")
                CACHE.mkdir(parents=True, exist_ok=True)
                cpath.write_text(text, encoding="utf-8")
                return text
            if r.status_code in (429, 503):
                print(f"  HTTP {r.status_code}, backing off")
                time.sleep(30 * (attempt + 1))
                continue
            print(f"  HTTP {r.status_code} for {url}")
            return None
        return None


ENTRY = re.compile(r"<(sitemap|url)>(.*?)</\1>", re.S)
LOC = re.compile(r"<loc>\s*(.*?)\s*</loc>", re.S)
MOD = re.compile(r"<lastmod>\s*(.*?)\s*</lastmod>", re.S)
NEWSDATE = re.compile(r"<news:publication_date>\s*(.*?)\s*</news:publication_date>", re.S)


def parse(xml: str):
    """-> (kind, [(loc, date_str)]) where kind is 'index' or 'urlset'."""
    kind = "index" if "<sitemapindex" in xml else "urlset"
    out = []
    for _, body in ENTRY.findall(xml):
        m = LOC.search(body)
        if not m:
            continue
        loc = html.unescape(m.group(1).replace("<![CDATA[", "").replace("]]>", "").strip())
        d = NEWSDATE.search(body) or MOD.search(body)
        out.append((loc, d.group(1)[:10] if d else ""))
    return kind, out


def crawl(site: str, since: str, until: str, max_sitemaps: int, delay: float):
    cfg = SITES[site]
    f = Fetcher(delay)
    queue = list(cfg["start"])
    seen_maps, links = set(), {}
    n_maps = 0
    while queue and n_maps < max_sitemaps:
        sm = queue.pop(0)
        if sm in seen_maps:
            continue
        seen_maps.add(sm)
        xml = f.get(sm)
        n_maps += 1
        if xml is None:
            continue
        kind, entries = parse(xml)
        if kind == "index":
            kids = []
            for loc, d in entries:
                if not re.search(cfg.get("child_include", "."), loc):
                    continue
                if cfg.get("child_exclude") and re.search(cfg["child_exclude"], loc):
                    continue
                # a child sitemap last modified before `since` holds only older articles
                if d and d < since:
                    continue
                # NBC: no <lastmod> in the index, but the month is in the name
                # (sitemap-2009-09-article.xml) -> skip months before `since`
                ym = re.search(r"(\d{4})-(\d{2})", loc)
                if ym and f"{ym.group(1)}-{ym.group(2)}" < since[:7]:
                    continue
                if ym and f"{ym.group(1)}-{ym.group(2)}" > until[:7]:
                    continue
                kids.append(loc)
            kids.sort(key=_page_no)
            queue.extend(kids)
            print(f"[{n_maps}] index {sm} -> {len(kids)} child sitemaps")
            continue
        kept = 0
        dates = [d for _, d in entries if d]
        for loc, d in entries:
            if cfg["url_host"] not in urlparse(loc).netloc:
                continue
            if d and not (since <= d <= until):
                continue
            if loc not in links:
                links[loc] = (d, sm)
                kept += 1
        rng = f"{min(dates)}..{max(dates)}" if dates else "no dates"
        print(f"[{n_maps}] urlset {sm}: {len(entries)} urls ({rng}), kept {kept}, total {len(links)}")
        # paginated newest-first sitemaps (Fox): stop once a whole page is older than `since`
        if dates and max(dates) < since and site == "fox":
            print("  reached pages older than --since, stopping")
            break
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{site}_links.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["url", "lastmod", "sitemap"])
        for u, (d, sm) in links.items():
            w.writerow([u, d, sm])
    print(f"\n{site}: {len(links)} links -> {path}  ({f.n_net} network requests, rest from cache)")


def _page_no(u: str):
    m = re.search(r"page=(\d+)", u)
    if m:
        return (0, int(m.group(1)), u)
    if "type=" in u:  # Fox: '?type=articles' without page= is page 1 (newest)
        return (0, 1, u)
    ym = re.search(r"(\d{4})-(\d{2})", u)
    if ym:  # NBC monthly sitemaps: newest month first
        return (1, -int(ym.group(1) + ym.group(2)), u)
    return (2, 0, u)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--site", choices=list(SITES), required=True)
    p.add_argument("--since", default="2022-01-01", help="keep links dated on/after (YYYY-MM-DD)")
    p.add_argument("--until", default="9999-12-31")
    p.add_argument("--max-sitemaps", type=int, default=400)
    p.add_argument("--delay", type=float, default=1.5, help="seconds between requests")
    a = p.parse_args()
    crawl(a.site, a.since, a.until, a.max_sitemaps, a.delay)


if __name__ == "__main__":
    sys.exit(main())
