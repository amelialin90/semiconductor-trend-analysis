"""OpenAlex adapter (https://docs.openalex.org) - no API key required.

OpenAlex indexes both arXiv and IEEE Xplore content with abstracts, so it is
used as the *default* route for the "arxiv" and "ieee" sources:

* arxiv : works that have a location hosted on arXiv (source S4306400194)
* ieee  : works whose primary location is published by IEEE (publisher P4310319808)

The domain query (``domain_terms`` in config/keywords.json) is sent as an OR
list to ``title_and_abstract.search``.  Per (source, year) we either fetch the
complete result set with cursor paging, or a uniform random ``sample`` of
``max_per_year`` works (OpenAlex sampling is seeded, so it is reproducible).
The true total for every year is stored in ``*_meta.json`` so shares can be
scaled back to absolute counts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import requests

from ..schema import Doc, RAW_DIR

API = "https://api.openalex.org/works"
SOURCE_FILTERS = {
    "arxiv": "locations.source.id:S4306400194",
    "ieee": "primary_location.source.host_organization:P4310319808",
}
SELECT = "id,doi,title,publication_year,publication_date,abstract_inverted_index,primary_location,type,cited_by_count,ids"
UA = "semitrend/0.1 (https://github.com/; mailto:{mailto})"


def _abstract_from_inverted(idx: dict | None) -> str:
    if not idx:
        return ""
    pos: list[tuple[int, str]] = []
    for word, positions in idx.items():
        for p in positions:
            pos.append((p, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def _to_doc(w: dict, source: str) -> Doc:
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}) if loc else {}
    ids = w.get("ids") or {}
    if source == "arxiv":
        # prefer the arXiv id when present
        arxiv_url = next((l.get("landing_page_url") for l in [loc] if l and "arxiv.org" in (l.get("landing_page_url") or "")), None)
        native_id = arxiv_url or w.get("doi") or w["id"]
    else:
        native_id = w.get("doi") or w["id"]
    return Doc(
        source=source,
        id=str(native_id),
        title=w.get("title") or "",
        abstract=_abstract_from_inverted(w.get("abstract_inverted_index")),
        year=int(w.get("publication_year") or 0),
        date=w.get("publication_date") or "",
        venue=src.get("display_name") or "",
        doc_type="paper",
        url=w.get("doi") or w["id"],
        extra={"openalex_id": w["id"], "type": w.get("type"), "cited_by": w.get("cited_by_count", 0),
               "pmid": ids.get("pmid")},
    )


def _get(session: requests.Session, params: dict, retries: int = 10) -> dict:
    delay = 3.0
    last = ""
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=120)
        except requests.RequestException as e:  # network hiccup
            last = f"network error: {e}"
            time.sleep(delay)
            delay = min(delay * 2, 120)
            continue
        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as e:
                last = f"bad JSON: {e}"
        elif r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}: {r.text[:200]!r}"
        else:
            r.raise_for_status()
        print(f"[openalex] retry {attempt + 1}/{retries} after {last} (sleep {delay:.0f}s)", flush=True)
        time.sleep(delay)
        delay = min(delay * 2, 120)
    raise RuntimeError(f"OpenAlex request failed after {retries} retries ({last}); cursor={params.get('cursor')!r}")


def build_query(domain_terms: list[str]) -> str:
    return " OR ".join(f'"{t}"' if " " in t else t for t in domain_terms)


def fetch(
    source: str,
    years: range,
    domain_terms: list[str],
    max_per_year: int = 0,
    seed: int = 42,
    mailto: str = "",
    pause: float = 0.25,
    progress: bool = True,
) -> tuple[int, dict]:
    """Fetch docs for *source* ("arxiv" | "ieee") into data/raw/openalex_<source>.jsonl.

    ``max_per_year == 0`` fetches everything; otherwise a seeded random sample.
    Returns (n_docs_written, meta) where meta[year] = {"total": N, "fetched": n}.
    """
    if source not in SOURCE_FILTERS:
        raise ValueError(f"unknown OpenAlex source {source!r}; choose from {list(SOURCE_FILTERS)}")
    out = RAW_DIR / f"openalex_{source}.jsonl"
    meta_path = RAW_DIR / f"openalex_{source}_meta.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    q = build_query(domain_terms)
    session = requests.Session()
    session.headers["User-Agent"] = UA.format(mailto=mailto or "unknown")
    meta: dict = {"source": source, "query": q, "max_per_year": max_per_year, "seed": seed, "years": {}}
    n_total = 0
    seen: set[str] = set()

    # Resume support: years already recorded as complete in the meta file are skipped and
    # the JSONL is appended to (ids already present are never written twice).
    done_years: set[int] = set()
    if meta_path.exists() and out.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("query") == q and old.get("max_per_year") == max_per_year:
                for y, v in old.get("years", {}).items():
                    if v.get("fetched", 0) > 0 and (v["fetched"] >= v["total"] or (max_per_year and v["fetched"] >= max_per_year)):
                        done_years.add(int(y))
                        meta["years"][y] = v
                with out.open(encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            seen.add(json.loads(line)["extra"]["openalex_id"])
                        except (ValueError, KeyError, TypeError):
                            pass
                if done_years:
                    print(f"[openalex:{source}] resuming; already complete: {sorted(done_years)}", flush=True)
        except (ValueError, OSError):
            done_years = set()
    mode = "a" if done_years else "w"
    if mode == "w":
        seen = set()

    with out.open(mode, encoding="utf-8") as fh:
        for year in years:
            if year in done_years:
                continue
            flt = f"publication_year:{year},{SOURCE_FILTERS[source]},title_and_abstract.search:{q}"
            base = {"filter": flt, "per_page": 200, "select": SELECT}
            if mailto:
                base["mailto"] = mailto

            # total count for this year
            head = _get(session, {**base, "per_page": 1})
            total = int(head["meta"]["count"])
            fetched = 0

            if max_per_year and total > max_per_year:
                # seeded random sample, paged with page= (sample supports up to 10k)
                params = {**base, "sample": min(max_per_year, 10000), "seed": seed}
                page = 1
                while fetched < max_per_year:
                    data = _get(session, {**params, "page": page})
                    res = data.get("results", [])
                    if not res:
                        break
                    for w in res:
                        if w["id"] in seen:
                            continue
                        seen.add(w["id"])
                        fh.write(json.dumps(_to_doc(w, source).__dict__, ensure_ascii=False) + "\n")
                        fetched += 1
                    page += 1
                    time.sleep(pause)
            else:
                cursor = "*"
                while cursor:
                    data = _get(session, {**base, "cursor": cursor})
                    res = data.get("results", [])
                    for w in res:
                        if w["id"] in seen:
                            continue
                        seen.add(w["id"])
                        fh.write(json.dumps(_to_doc(w, source).__dict__, ensure_ascii=False) + "\n")
                        fetched += 1
                    cursor = data.get("meta", {}).get("next_cursor")
                    if not res:
                        break
                    time.sleep(pause)

            meta["years"][str(year)] = {"total": total, "fetched": fetched}
            n_total += fetched
            if progress:
                print(f"[openalex:{source}] {year}: total={total} fetched={fetched}", flush=True)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n_total, meta
