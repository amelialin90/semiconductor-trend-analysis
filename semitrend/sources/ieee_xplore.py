"""IEEE Xplore Metadata API adapter (https://developer.ieee.org).

Requires an API key in the ``IEEE_API_KEY`` environment variable.  Free keys
allow 200 calls/day and 200 records/call, so a full multi-year pull needs to be
spread over several days or use ``--max-per-year``.  Used with ``fetch --direct``;
by default the IEEE corpus is obtained through OpenAlex instead.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from ..schema import Doc, RAW_DIR

API = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
PAGE = 200


def build_query(domain_terms: list[str]) -> str:
    return " OR ".join(f'"{t}"' for t in domain_terms)


def fetch(years: range, domain_terms: list[str], max_per_year: int = 0, out: Path | None = None) -> int:
    key = os.environ.get("IEEE_API_KEY")
    if not key:
        raise SystemExit("IEEE_API_KEY is not set - register at https://developer.ieee.org or drop --direct to use OpenAlex")
    out = out or RAW_DIR / "ieee_direct.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out.with_name(out.stem + "_meta.json")
    q = build_query(domain_terms)
    meta: dict = {"source": "ieee", "query": q, "years": {}}
    n_total = 0
    with out.open("w", encoding="utf-8") as fh:
        for year in years:
            start, fetched, total = 1, 0, None
            while True:
                params = {"apikey": key, "abstract": q, "start_year": year, "end_year": year,
                          "max_records": PAGE, "start_record": start, "sort_field": "publication_year"}
                r = requests.get(API, params=params, timeout=90)
                if r.status_code == 429:
                    time.sleep(10)
                    continue
                r.raise_for_status()
                data = r.json()
                total = int(data.get("total_records", 0))
                arts = data.get("articles", [])
                if not arts:
                    break
                for a in arts:
                    doc = Doc(
                        source="ieee", id=a.get("doi") or str(a.get("article_number")), title=a.get("title", ""),
                        abstract=a.get("abstract", ""), year=int(a.get("publication_year") or year),
                        date=a.get("publication_date", ""), venue=a.get("publication_title", ""), doc_type="paper",
                        url=a.get("html_url", ""), extra={"content_type": a.get("content_type"),
                                                            "citing_paper_count": a.get("citing_paper_count")},
                    )
                    fh.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")
                    fetched += 1
                    if max_per_year and fetched >= max_per_year:
                        break
                if (max_per_year and fetched >= max_per_year) or start + PAGE > total:
                    break
                start += PAGE
                time.sleep(0.5)
            meta["years"][str(year)] = {"total": total or 0, "fetched": fetched}
            n_total += fetched
            print(f"[ieee-api] {year}: total={total} fetched={fetched}", flush=True)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n_total
