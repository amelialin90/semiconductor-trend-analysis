"""Semantic Scholar bulk-search adapter (https://api.semanticscholar.org) - no key needed.

Semantic Scholar (S2) indexes IEEE Xplore *and* arXiv with abstracts, so one
adapter serves both.  A record is tagged

* ``ieee``  when its DOI starts with ``10.1109`` (IEEE's DOI prefix) or its venue name contains "IEEE",
* ``arxiv`` when it carries an arXiv id (a preprint later published in IEEE gets both tags),
* ``all``   always (the whole S2 corpus that matched the domain query, every publisher).

The domain query returns ~100k works per year - too much to download - so
the adapter uses a two-pool design:

1. **sample pool** - the first ``sample_pages`` x 1000 results of the domain query
   per year.  S2 returns bulk results in paper-id order and paper ids are
   hashes, so this is effectively a uniform random sample.  It yields the
   per-source *denominators* (fraction of the domain total that is IEEE / arXiv
   and passes the local relevance gate) and feeds the emerging-term miner.
2. **concept pool** - one query per concept per year built from its synonyms.
   These are small (hundreds to a few thousand hits) and are fetched completely
   up to ``max_pages_concept`` pages; beyond that the fetched part is a random
   sample and the numerator is scaled by total/fetched.  Every document is
   re-verified locally with the regex matcher, so S2's fuzzy search only
   determines *recall*, never precision.

Rate limit without a key is ~1 request/s shared; set ``S2_API_KEY`` for a
private quota.  Progress is checkpointed in ``s2_meta.json`` so the fetch can
be resumed.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import requests

from ..matcher import load_config
from ..schema import Doc, RAW_DIR

API = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
FIELDS = "title,abstract,year,publicationDate,externalIds,venue,publicationVenue,citationCount,publicationTypes"
PAGE = 1000
UA = "semitrend/0.1 (research; mailto:{mailto})"
_TOKEN_OK = re.compile(r"^[A-Za-z0-9]+$")


# ----------------------------------------------------------------------------- query building
def s2_term(t: str) -> str:
    t = t.strip()
    if _TOKEN_OK.match(t):
        return t
    return '"' + re.sub(r"[\s\-]+", " ", t).strip('"') + '"'


def domain_query(domain_terms: list[str]) -> str:
    return " | ".join(s2_term(t) for t in domain_terms)


def concept_query(synonyms: list[str], s2_extra: list[str] | None = None) -> str:
    parts: list[str] = []
    for s in synonyms:
        if s.startswith("re:"):
            continue
        parts.append(s2_term(s))
        if re.search(r"[\s\-]", s):
            joined = re.sub(r"[\s\-]+", "", s)
            if _TOKEN_OK.match(joined) and len(joined) >= 6:
                parts.append(joined)  # "nano sheet" -> nanosheet
    for s in s2_extra or []:
        parts.append(s2_term(s))
    seen, out = set(), []
    for p in parts:
        if p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return " | ".join(out)


# ----------------------------------------------------------------------------- http
def _get(session: requests.Session, params: dict, retries: int = 12) -> dict:
    delay = 4.0
    last = ""
    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=180)
        except requests.RequestException as e:
            last = f"network error: {e}"
        else:
            if r.status_code == 200:
                try:
                    return r.json()
                except ValueError as e:
                    last = f"bad JSON: {e}"
            elif r.status_code in (429, 500, 502, 503, 504):
                ra = r.headers.get("Retry-After")
                last = f"HTTP {r.status_code}"
                if ra and ra.isdigit():
                    delay = max(delay, float(ra))
            else:
                r.raise_for_status()
        print(f"[s2] retry {attempt + 1}/{retries} after {last} (sleep {delay:.0f}s)", flush=True)
        time.sleep(delay)
        delay = min(delay * 1.8, 120)
    raise RuntimeError(f"Semantic Scholar request failed after {retries} retries ({last})")


def _tags(rec: dict) -> list[str]:
    ext = rec.get("externalIds") or {}
    doi = (ext.get("DOI") or "").lower()
    venue = (rec.get("venue") or "") + " " + ((rec.get("publicationVenue") or {}).get("name") or "")
    tags = ["all"]
    if doi.startswith("10.1109") or "ieee" in venue.lower():
        tags.append("ieee")
    if ext.get("ArXiv"):
        tags.append("arxiv")
    return tags


def _to_doc(rec: dict, pool: str, year: int) -> Doc:
    ext = rec.get("externalIds") or {}
    doi = ext.get("DOI") or ""
    arx = ext.get("ArXiv") or ""
    url = f"https://doi.org/{doi}" if doi else (f"https://arxiv.org/abs/{arx}" if arx else f"https://www.semanticscholar.org/paper/{rec['paperId']}")
    return Doc(
        source="s2", id=rec["paperId"], title=rec.get("title") or "", abstract=rec.get("abstract") or "",
        year=int(rec.get("year") or year), date=rec.get("publicationDate") or "", venue=rec.get("venue") or "",
        doc_type="paper", url=url,
        extra={"pool": pool, "sources": _tags(rec), "doi": doi, "arxiv": arx, "cited_by": rec.get("citationCount") or 0,
               "types": rec.get("publicationTypes") or []},
    )


def _paged(session: requests.Session, query: str, year: int, max_pages: int, pause: float) -> tuple[int, list[dict]]:
    params = {"query": query, "year": f"{year}-{year}", "fields": FIELDS, "limit": PAGE}
    total, out, token, pages = 0, [], None, 0
    while True:
        if token:
            params["token"] = token
        data = _get(session, params)
        total = int(data.get("total") or 0)
        out.extend(data.get("data") or [])
        token = data.get("token")
        pages += 1
        time.sleep(pause)
        if not token or pages >= max_pages or not data.get("data"):
            break
    return total, out


# ----------------------------------------------------------------------------- main fetch
def fetch(years: range, config_path: str | Path, sample_pages: int = 10, max_pages_concept: int = 3,
          mailto: str = "", pause: float = 1.0, shard: int = 0, nshards: int = 1) -> int:
    """Fetch both pools.  With ``nshards > 1`` the concept queries are split across
    processes (concept index % nshards == shard); only shard 0 fetches the sample pool.
    Shard files: s2_concepts.<shard>.jsonl / s2_meta.<shard>.json (shard 0 keeps the plain names)."""
    cfg = load_config(config_path)
    q_domain = domain_query(cfg["domain_terms"])
    concepts = {k: spec for cat in cfg["categories"].values() for k, spec in cat.items()}
    if nshards > 1:
        concepts = {k: spec for i, (k, spec) in enumerate(concepts.items()) if i % nshards == shard}

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    sample_path = RAW_DIR / "s2_sample.jsonl"
    suffix = "" if shard == 0 else f".{shard}"
    concept_path = RAW_DIR / f"s2_concepts{suffix}.jsonl"
    meta_path = RAW_DIR / f"s2_meta{suffix}.json"

    session = requests.Session()
    session.headers["User-Agent"] = UA.format(mailto=mailto or "unknown")
    key = os.environ.get("S2_API_KEY")
    if key:
        session.headers["x-api-key"] = key
        pause = min(pause, 0.4)

    meta: dict = {"source": "s2", "domain_query": q_domain, "sample_pages": sample_pages, "shard": shard, "nshards": nshards,
                  "max_pages_concept": max_pages_concept, "years": {}, "concepts": {}}
    if meta_path.exists():
        try:
            old = json.loads(meta_path.read_text(encoding="utf-8"))
            if old.get("domain_query") == q_domain:
                meta["years"] = old.get("years", {})
                meta["concepts"] = old.get("concepts", {})
                print(f"[s2 shard {shard}/{nshards}] resuming: {len(meta['years'])} sample years, "
                      f"{sum(len(v) for v in meta['concepts'].values())} concept-years done", flush=True)
        except ValueError:
            pass
    if shard == 0 and not meta["years"]:
        sample_path.unlink(missing_ok=True)
    if not meta["concepts"]:
        concept_path.unlink(missing_ok=True)

    def save_meta() -> None:
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")

    n_written = 0
    # ---- pool 1: yearly domain sample (denominators + emerging terms) - shard 0 only
    with sample_path.open("a", encoding="utf-8") as fh:
        for year in years:
            if shard != 0 or str(year) in meta["years"]:
                continue
            total, recs = _paged(session, q_domain, year, sample_pages, pause)
            tag_counts = {"all": 0, "ieee": 0, "arxiv": 0}
            for rec in recs:
                d = _to_doc(rec, "sample", year)
                for t in d.extra["sources"]:
                    tag_counts[t] += 1
                fh.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
            fh.flush()
            meta["years"][str(year)] = {"domain_total": total, "sample_n": len(recs), "sample_tags": tag_counts}
            n_written += len(recs)
            print(f"[s2] sample {year}: domain_total={total} sampled={len(recs)} ieee={tag_counts['ieee']} arxiv={tag_counts['arxiv']}", flush=True)
            save_meta()

    def done_elsewhere(key_: str, year: int) -> bool:
        """True when another shard's meta file already records this concept-year (lets finer
        re-sharding of a slow shard run alongside it without duplicate downloads)."""
        for mp in RAW_DIR.glob("s2_meta*.json"):
            if mp == meta_path:
                continue
            try:
                v = json.loads(mp.read_text(encoding="utf-8")).get("concepts", {}).get(key_, {}).get(str(year))
            except (ValueError, OSError):
                continue
            if v and v.get("fetched", 0) > 0:
                return True
        return False

    # ---- pool 2: per-concept queries (numerators)
    with concept_path.open("a", encoding="utf-8") as fh:
        for key_, spec in concepts.items():
            q = concept_query(spec["synonyms"], spec.get("s2_extra"))
            if not q:
                continue
            done = meta["concepts"].setdefault(key_, {})
            for year in years:
                if str(year) in done or done_elsewhere(key_, year):
                    continue
                total, recs = _paged(session, q, year, max_pages_concept, pause)
                for rec in recs:
                    d = _to_doc(rec, "concept", year)
                    d.extra["retrieved_for"] = key_
                    fh.write(json.dumps(d.__dict__, ensure_ascii=False) + "\n")
                fh.flush()
                done[str(year)] = {"total": total, "fetched": len(recs), "query": q if year == years[0] else None}
                n_written += len(recs)
                print(f"[s2] {key_:14s} {year}: total={total} fetched={len(recs)}", flush=True)
                save_meta()
    save_meta()
    return n_written
