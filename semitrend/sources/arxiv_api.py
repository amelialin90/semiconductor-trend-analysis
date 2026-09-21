"""Direct arXiv API adapter (https://info.arxiv.org/help/api/).

Used with ``fetch --direct``.  arXiv asks for >= 3 s between requests and, as
of 2026, returns HTTP 429 "Rate exceeded" aggressively (including on the first
request from some networks).  This adapter therefore backs off exponentially
and gives up cleanly; the OpenAlex route (default) is the recommended way to
obtain the arXiv corpus.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path

import feedparser
import requests

from ..schema import Doc, RAW_DIR

API = "https://export.arxiv.org/api/query"
PAGE = 200
UA = "semitrend/0.1 (research trend analysis; contact via GitHub)"


def build_query(domain_terms: list[str]) -> str:
    parts = []
    for t in domain_terms:
        t = t.strip()
        parts.append(f'abs:"{t}"' if " " in t else f"abs:{t}")
    return "(" + " OR ".join(parts) + ")"


def _request(session: requests.Session, params: dict, max_wait: float = 300.0) -> feedparser.FeedParserDict:
    delay = 5.0
    waited = 0.0
    while True:
        r = session.get(API, params=params, timeout=90)
        if r.status_code == 200:
            return feedparser.parse(r.text)
        if r.status_code == 429 or r.status_code >= 500:
            if waited >= max_wait:
                raise RuntimeError(f"arXiv API keeps returning {r.status_code}: {r.text[:120]!r}")
            time.sleep(delay)
            waited += delay
            delay = min(delay * 2, 60)
            continue
        r.raise_for_status()


def fetch(years: range, domain_terms: list[str], max_per_year: int = 0, out: Path | None = None) -> int:
    out = out or RAW_DIR / "arxiv_direct.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_name(out.stem + "_meta.json")
    q = build_query(domain_terms)
    session = requests.Session()
    session.headers["User-Agent"] = UA
    meta: dict = {"source": "arxiv", "query": q, "years": {}}
    n_total = 0

    with out.open("w", encoding="utf-8") as fh:
        for year in years:
            sq = f"{q} AND submittedDate:[{year}01010000 TO {year}12312359]"
            start, fetched, total = 0, 0, None
            while True:
                feed = _request(session, {"search_query": sq, "start": start, "max_results": PAGE,
                                          "sortBy": "submittedDate", "sortOrder": "ascending"})
                if total is None:
                    total = int(feed.feed.get("opensearch_totalresults", 0))
                entries = feed.entries
                if not entries:
                    break
                for e in entries:
                    aid = e.id.rsplit("/", 1)[-1]
                    doc = Doc(
                        source="arxiv", id=aid, title=" ".join(e.title.split()),
                        abstract=" ".join(e.summary.split()), year=int(e.published[:4]), date=e.published[:10],
                        venue=",".join(t.term for t in e.get("tags", [])), doc_type="paper", url=e.id,
                        extra={"primary_category": (e.get("arxiv_primary_category") or {}).get("term")},
                    )
                    fh.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")
                    fetched += 1
                    if max_per_year and fetched >= max_per_year:
                        break
                if (max_per_year and fetched >= max_per_year) or start + PAGE >= total:
                    break
                start += PAGE
                time.sleep(3.2)
            meta["years"][str(year)] = {"total": total or 0, "fetched": fetched}
            n_total += fetched
            print(f"[arxiv-api] {year}: total={total} fetched={fetched}", flush=True)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            time.sleep(3.2)
    return n_total
