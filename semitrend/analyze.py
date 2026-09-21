"""Keyword-trend analysis: raw JSONL -> per-document matches -> trend tables.

Two kinds of raw data are understood:

* **plain sources** (``openalex_*.jsonl``, ``*_direct.jsonl``, ``google_patents.jsonl``):
  every document of the source-year corpus is present; denominators are simply
  the number of documents passing the local relevance gate.
* **Semantic Scholar two-pool data** (``s2_sample.jsonl`` + ``s2_concepts*.jsonl``):
  a random *sample* of the domain corpus gives the denominators per source tag
  (all / ieee / arxiv), while per-concept queries give (near-)complete
  numerators, scaled by total/fetched when a concept-year was truncated.

Everything is streamed year by year (raw files are first split into per-year
files under data/processed/by_year/), so memory stays around a few hundred MB
even for a ~1 GB concept pool.

Outputs (output/tables/):
  totals.csv        analysed / in-domain documents per source-year (+ corpus totals, abstract rate)
  concept_year.csv  docs & share (%) per concept per source-year
  synonym_year.csv  which surface form (synonym) produced the hits, per year
  growth.csv        recent-vs-prior window growth per concept (ratio, delta, slope, first year)
  cooccurrence.csv  concept pairs that appear in the same document (with lift)
  examples.csv      most-cited example documents per concept (recent years)
  emerging.csv      bursting n-grams, flagged when not covered by the taxonomy
  summary.json      run metadata
"""

from __future__ import annotations

import datetime as dt
import heapq
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from . import discover
from .matcher import DEFAULT_CONFIG, Matcher
from .normalize import normalize
from .schema import OUTPUT_DIR, PROCESSED_DIR, RAW_DIR

RAW_CANDIDATES = {
    "arxiv": ["arxiv_direct.jsonl", "openalex_arxiv.jsonl"],
    "ieee": ["ieee_direct.jsonl", "openalex_ieee.jsonl"],
    "google_patents": ["google_patents.jsonl"],
}
S2_TAGS = ["all", "ieee", "arxiv"]
NGRAM_TAGS = ["all", "ieee"]          # arXiv sample is too small per year for n-gram bursts
NGRAM_MIN_DF = 5
TABLES = OUTPUT_DIR / "tables"
BY_YEAR = PROCESSED_DIR / "by_year"
_YEAR_RE = re.compile(r'"year":\s*(\d{4})')

# ----------------------------------------------------------------------------- worker
_M: Matcher | None = None


def _init(config: str) -> None:
    global _M
    _M = Matcher(config)


def _process_year(task: tuple[str, int, list[dict], list[str] | None]) -> tuple[str, int, list[tuple], dict[str, Counter]]:
    """Normalise + gate + match every doc of one (source, year).

    Returns rows ``(id, in_domain, hits, cited, title, url, extra)`` and, when
    ``ngram_tags`` is given, an n-gram document-frequency Counter per tag
    (tag "" = all docs) computed over in-domain docs.
    """
    source, year, docs, ngram_tags = task
    assert _M is not None
    rows: list[tuple] = []
    # n-gram counts do not depend on the keyword config, so they are cached per (source, year, tag)
    cache = {t: PROCESSED_DIR / "ngrams" / f"{source}.{year}.{t or 'all'}.json" for t in (ngram_tags or [])}
    cached = {t: p for t, p in cache.items() if p.exists()}
    texts: dict[str, list[str]] = {t: [] for t in (ngram_tags or []) if t not in cached}
    for d in docs:
        text = normalize(f"{d.get('title', '')}. {d.get('abstract', '')}")
        in_dom = _M.in_domain(text)
        hits = _M.match(text) if in_dom else {}
        extra = d.get("extra") or {}
        cited = extra.get("cited_by") or extra.get("citing_paper_count") or 0
        rows.append((d["id"], in_dom, hits, int(cited or 0), d.get("title", "") if hits else "", d.get("url", "") if hits else "", extra))
        if in_dom and texts:
            tags = extra.get("sources") or [""]
            for t in texts:
                if t == "" or t in tags:
                    texts[t].append(text)
    ngrams = {t: discover.count_year(tx, min_df=NGRAM_MIN_DF) for t, tx in texts.items()}
    for t, c in ngrams.items():
        cache[t].parent.mkdir(parents=True, exist_ok=True)
        cache[t].write_text(json.dumps(c), encoding="utf-8")
    for t, p in cached.items():
        ngrams[t] = Counter(json.loads(p.read_text(encoding="utf-8")))
    return source, year, rows, ngrams


def _run_tasks(tasks: Iterable[tuple], config: str, workers: int) -> Iterator[tuple]:
    """Run year-tasks on a process pool with at most ``workers`` in flight (tasks are
    produced lazily, so only that many per-year document lists exist at once)."""
    it = iter(tasks)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(config,)) as ex:
            pending = []
            for t in it:
                pending.append(ex.submit(_process_year, t))
                if len(pending) >= workers:
                    yield pending.pop(0).result()
            for f in pending:
                yield f.result()
    else:
        _init(config)
        for t in it:
            yield _process_year(t)


# ----------------------------------------------------------------------------- helpers
def find_raw_files(sources: list[str] | None = None) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {}
    want_s2 = not sources or any(s in ("s2", "arxiv", "ieee") for s in sources)
    s2c = sorted(p for p in RAW_DIR.glob("s2_concepts*.jsonl") if p.stat().st_size > 0)
    s2s = RAW_DIR / "s2_sample.jsonl"
    if want_s2 and s2c and s2s.exists() and s2s.stat().st_size > 0 and (RAW_DIR / "s2_meta.json").exists():
        found["s2"] = s2c
    for src, names in RAW_CANDIDATES.items():
        if sources and src not in sources:
            continue
        if src in ("arxiv", "ieee") and "s2" in found and not (sources and src in sources):
            continue  # S2 already provides these tags; explicit request still allowed
        cands = [RAW_DIR / n for n in names if (RAW_DIR / n).exists() and (RAW_DIR / n).stat().st_size > 0]
        if cands:
            found[src] = [max(cands, key=lambda p: p.stat().st_mtime)]
    return found


def _load_meta(raw: Path) -> dict:
    mp = raw.with_name(raw.stem + "_meta.json")
    return json.loads(mp.read_text(encoding="utf-8")) if mp.exists() else {}


def _split_by_year(paths: list[Path], name: str) -> dict[int, Path]:
    """Copy JSONL lines into data/processed/by_year/<name>.<year>.jsonl (one pass, regex on year).
    Skipped when the split files are newer than every raw file (re-running after a keyword edit)."""
    BY_YEAR.mkdir(parents=True, exist_ok=True)
    existing = sorted(BY_YEAR.glob(f"{name}.*.jsonl"))
    newest_raw = max(p.stat().st_mtime for p in paths)
    if existing and all(p.stat().st_mtime >= newest_raw for p in existing):
        print(f"[analyze] reusing {len(existing)} per-year files for {name}", flush=True)
        return {int(p.stem.rsplit(".", 1)[1]): p for p in existing}
    for old in existing:
        old.unlink()
    handles: dict[int, object] = {}
    out: dict[int, Path] = {}
    try:
        for p in paths:
            with p.open(encoding="utf-8") as fh:
                for line in fh:
                    m = _YEAR_RE.search(line)
                    if not m:
                        continue
                    y = int(m.group(1))
                    if y not in handles:
                        out[y] = BY_YEAR / f"{name}.{y}.jsonl"
                        handles[y] = out[y].open("w", encoding="utf-8")
                    handles[y].write(line)
    finally:
        for h in handles.values():
            h.close()
    return dict(sorted(out.items()))


def _slim(d: dict) -> dict:
    extra = dict(d.get("extra") or {})
    extra["has_abstract"] = bool((d.get("abstract") or "").strip())
    return {"id": d["id"], "title": d.get("title", ""), "abstract": d.get("abstract", ""), "url": d.get("url", ""), "extra": extra}


def _read_year(path: Path, merge_retrieved: bool = False) -> list[dict]:
    """Read one per-year file, dropping duplicate ids; optionally merge the S2 ``retrieved_for``
    concept keys of duplicates (the same paper is returned by several concept queries)."""
    docs: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:  # truncated last line of a file still being written
                continue
            slot = docs.get(d["id"])
            if slot is None:
                slot = _slim(d)
                if merge_retrieved:
                    slot["extra"]["retrieved_for"] = {slot["extra"].get("retrieved_for")} - {None}
                docs[d["id"]] = slot
            elif merge_retrieved:
                slot["extra"]["retrieved_for"].add((d.get("extra") or {}).get("retrieved_for"))
    if merge_retrieved:
        for slot in docs.values():
            slot["extra"]["retrieved_for"] = sorted(k for k in slot["extra"]["retrieved_for"] if k)
    return list(docs.values())


def _slope(years: list[int], vals: list[float]) -> float:
    if len(years) < 3:
        return float("nan")
    return float(np.polyfit(years, vals, 1)[0])


class Acc:
    """Row accumulators shared by both loaders."""

    def __init__(self) -> None:
        self.totals: list[dict] = []
        self.cy: list[dict] = []
        self.syn: list[dict] = []
        self.co: list[dict] = []
        self.em: list[dict] = []
        self.examples: dict[tuple[str, str], list[tuple]] = defaultdict(list)  # (source, concept) -> heap of (cited, y, id, title, url)
        self.ngrams: dict[str, dict[int, Counter]] = defaultdict(dict)   # source -> year -> Counter
        self.n_in_domain: dict[str, dict[int, int]] = defaultdict(dict)  # source -> year -> analysed in-domain docs

    def example(self, src: str, concept: str, item: tuple) -> None:
        h = self.examples[(src, concept)]
        if any(x[2] == item[2] for x in h):
            return
        if len(h) < 3:
            heapq.heappush(h, item)
        elif item > h[0]:
            heapq.heapreplace(h, item)


def _emit_year(acc: Acc, src: str, year: int, this_year: int, matcher: Matcher, concept_meta: dict,
               docs_iter, denominator: float, analysed: int, in_dom_analysed: int, source_total: float,
               weight_of=None, count_ok=lambda hits_key, extra: True, abstract_rate: float = float("nan")) -> None:
    """Aggregate one source-year.  ``docs_iter`` yields (hits, cited, title, url, extra) for matched docs."""
    concept_w: Counter = Counter()
    concept_raw: Counter = Counter()
    syn_docs: Counter = Counter()
    pair_docs: Counter = Counter()
    recent = year >= this_year - 3
    for hits, cited, title, url, extra in docs_iter:
        keys = sorted(k for k in hits if count_ok(k, extra))
        for k in keys:
            w = weight_of(k, extra) if weight_of else 1.0
            concept_w[k] += w
            concept_raw[k] += 1
            for s in hits[k]:
                syn_docs[(k, s)] += 1
            if recent:
                acc.example(src, k, (cited, year, extra.get("_id", title), title, url))
        for a, b in combinations(sorted(hits), 2):
            pair_docs[(a, b)] += 1
    scale = (source_total / analysed) if analysed else 1.0
    acc.totals.append({"source": src, "year": year, "fetched": analysed, "in_domain": denominator,
                       "in_domain_analysed": in_dom_analysed, "source_total": source_total, "scale": scale,
                       "abstract_rate": abstract_rate, "partial_year": year == this_year})
    for k in matcher.keys():
        cat, label = concept_meta[k]
        n = concept_w.get(k, 0.0)
        acc.cy.append({"source": src, "category": cat, "concept": k, "label": label, "year": year,
                       "docs": round(n, 2), "docs_raw": concept_raw.get(k, 0), "in_domain_total": denominator,
                       "share_pct": (100.0 * n / denominator) if denominator else 0.0,
                       "est_docs": round(n * (1.0 if weight_of else scale), 1), "partial_year": year == this_year})
    for (k, s), n in syn_docs.items():
        acc.syn.append({"source": src, "concept": k, "synonym": s, "year": year, "docs": n})
    for (a, b), n in pair_docs.items():
        acc.co.append({"source": src, "year": year, "concept_a": a, "concept_b": b, "docs_both": n})


def _matched_iter(rows: list[tuple], tag: str | None = None):
    for did, in_dom, hits, cited, title, url, extra in rows:
        if hits and (tag is None or tag in (extra.get("sources") or [])):
            extra["_id"] = did
            yield hits, cited, title, url, extra


# ----------------------------------------------------------------------------- loaders
def _load_plain(acc: Acc, src: str, raw: Path, matcher: Matcher, concept_meta: dict, config: str, workers: int, this_year: int) -> None:
    print(f"[analyze] {src}: splitting {raw.name} by year", flush=True)
    year_files = _split_by_year([raw], src)
    meta_totals = {int(y): int(v["total"]) for y, v in _load_meta(raw).get("years", {}).items()}
    tasks = ((src, y, _read_year(p), [""]) for y, p in year_files.items())
    with (PROCESSED_DIR / f"matches_{src}.jsonl").open("w", encoding="utf-8") as fh:
        for s, y, rows, ng in _run_tasks(tasks, config, workers):
            acc.ngrams[src][y] = ng[""]
            in_dom = sum(r[1] for r in rows)
            acc.n_in_domain[src][y] = in_dom
            print(f"[analyze] {s} {y}: {in_dom}/{len(rows)} in-domain", flush=True)
            for did, in_dom_flag, hits, cited, title, url, extra in rows:
                if hits:
                    fh.write(json.dumps({"id": did, "year": y, "concepts": hits, "cited": cited, "title": title, "url": url}, ensure_ascii=False) + "\n")
            _emit_year(acc, src, y, this_year, matcher, concept_meta, _matched_iter(rows),
                       denominator=in_dom, analysed=len(rows), in_dom_analysed=in_dom, source_total=meta_totals.get(y, len(rows)),
                       abstract_rate=(sum(1 for r in rows if r[6].get("has_abstract")) / len(rows)) if rows else float("nan"))
            del rows


def _load_s2(acc: Acc, concept_files: list[Path], matcher: Matcher, concept_meta: dict, config: str, workers: int, this_year: int) -> None:
    meta = json.loads((RAW_DIR / "s2_meta.json").read_text(encoding="utf-8"))
    years_meta = {int(y): v for y, v in meta["years"].items()}
    concept_meta_s2: dict[str, dict[int, dict]] = {}
    for mp in sorted(RAW_DIR.glob("s2_meta*.json")):  # shard 0 + any parallel shards
        try:
            for k, d in json.loads(mp.read_text(encoding="utf-8")).get("concepts", {}).items():
                concept_meta_s2.setdefault(k, {}).update({int(y): v for y, v in d.items()})
        except ValueError:
            continue

    print("[analyze] s2: splitting sample and concept pools by year", flush=True)
    sample_files = _split_by_year([RAW_DIR / "s2_sample.jsonl"], "s2sample")
    concept_year_files = _split_by_year(concept_files, "s2concepts")

    # ---- pool 1: sample -> denominators + emerging terms
    denom: dict[str, dict[int, float]] = defaultdict(dict)
    tag_total: dict[str, dict[int, float]] = defaultdict(dict)
    tag_n: dict[str, dict[int, int]] = defaultdict(dict)
    tag_abs: dict[str, dict[int, float]] = defaultdict(dict)
    tasks = (("s2-sample", y, _read_year(p), NGRAM_TAGS) for y, p in sample_files.items() if y in years_meta)
    for _, y, rows, ng in _run_tasks(tasks, config, workers):
        n = len(rows)
        dt_ = years_meta[y]["domain_total"]
        for t in S2_TAGS:
            tagged = [r for r in rows if t in (r[6].get("sources") or [])]
            n_tag_dom = sum(1 for r in tagged if r[1])
            denom[t][y] = dt_ * n_tag_dom / n if n else 0.0
            tag_total[t][y] = dt_ * len(tagged) / n if n else 0.0
            tag_n[t][y] = len(tagged)
            tag_abs[t][y] = (sum(1 for r in tagged if r[6].get("has_abstract")) / len(tagged)) if tagged else float("nan")
            acc.n_in_domain[t][y] = n_tag_dom
            if t in ng:
                acc.ngrams[t][y] = ng[t]
        print(f"[analyze] s2 sample {y}: n={n} in-domain all/ieee/arxiv = "
              f"{acc.n_in_domain['all'][y]}/{acc.n_in_domain['ieee'][y]}/{acc.n_in_domain['arxiv'][y]} "
              f"-> est. corpus {denom['all'][y]:.0f}/{denom['ieee'][y]:.0f}/{denom['arxiv'][y]:.0f}", flush=True)
        del rows

    # ---- pool 2: concept queries -> numerators
    def make_weight(y: int):
        def weight_of(k: str, extra: dict) -> float:
            m = concept_meta_s2.get(k, {}).get(y, {})
            tot, fet = m.get("total", 0), m.get("fetched", 0)
            return (tot / fet) if fet and tot > fet else 1.0
        return weight_of

    def count_ok(k: str, extra: dict) -> bool:
        return k in (extra.get("retrieved_for") or [])

    tasks = (("s2-concepts", y, _read_year(p, merge_retrieved=True), None) for y, p in concept_year_files.items() if y in years_meta)
    with (PROCESSED_DIR / "matches_s2.jsonl").open("w", encoding="utf-8") as fh:
        for _, y, rows, _ng in _run_tasks(tasks, config, workers):
            print(f"[analyze] s2 concepts {y}: {sum(1 for r in rows if r[2])}/{len(rows)} docs matched locally", flush=True)
            for did, in_dom, hits, cited, title, url, extra in rows:
                if hits:
                    fh.write(json.dumps({"id": did, "year": y, "concepts": hits, "cited": cited, "title": title, "url": url,
                                         "sources": extra.get("sources"), "retrieved_for": extra.get("retrieved_for")}, ensure_ascii=False) + "\n")
            weight_of = make_weight(y)
            for t in S2_TAGS:
                _emit_year(acc, t, y, this_year, matcher, concept_meta, _matched_iter(rows, t),
                           denominator=denom[t].get(y, 0.0), analysed=tag_n[t].get(y, 0),
                           in_dom_analysed=acc.n_in_domain[t].get(y, 0), source_total=tag_total[t].get(y, 0.0),
                           weight_of=weight_of, count_ok=count_ok, abstract_rate=tag_abs[t].get(y, float("nan")))
            del rows
    for y in sorted(years_meta):  # years with a sample but no concept docs still need denominator rows
        if y not in concept_year_files:
            for t in S2_TAGS:
                _emit_year(acc, t, y, this_year, matcher, concept_meta, iter(()), denominator=denom[t].get(y, 0.0),
                           analysed=tag_n[t].get(y, 0), in_dom_analysed=acc.n_in_domain[t].get(y, 0),
                           source_total=tag_total[t].get(y, 0.0), weight_of=make_weight(y), abstract_rate=tag_abs[t].get(y, float("nan")))


# ----------------------------------------------------------------------------- main
def run(sources: list[str] | None = None, config: str | Path = DEFAULT_CONFIG, workers: int = 4,
        recent_window: int = 3) -> None:
    config = str(config)
    raw_files = find_raw_files(sources)
    if not raw_files:
        sys.exit(f"no raw data found in {RAW_DIR}; run `python -m semitrend fetch` first")
    matcher = Matcher(config)
    concept_meta = {c.key: (c.category, c.label) for c in matcher.concepts}
    this_year = dt.date.today().year
    TABLES.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    acc = Acc()

    for src, paths in raw_files.items():
        if src == "s2":
            _load_s2(acc, paths, matcher, concept_meta, config, workers, this_year)
        else:
            _load_plain(acc, src, paths[0], matcher, concept_meta, config, workers, this_year)

    totals = pd.DataFrame(acc.totals).sort_values(["source", "year"])
    cy = pd.DataFrame(acc.cy)
    syn = pd.DataFrame(acc.syn)
    co = pd.DataFrame(acc.co)
    ex = pd.DataFrame([{"source": s, "concept": k, "year": y, "id": did, "title": title, "url": url, "cited": cited}
                       for (s, k), h in acc.examples.items() for cited, y, did, title, url in sorted(h, reverse=True)])
    plain_sources = [s for s in raw_files if s != "s2"]

    # "all" pseudo-source for plain multi-source runs (S2 provides its own "all" tag)
    if "s2" not in raw_files and len(plain_sources) > 1:
        g = cy.groupby(["category", "concept", "label", "year", "partial_year"], as_index=False)[["docs", "docs_raw", "in_domain_total", "est_docs"]].sum()
        g["source"] = "all"
        g["share_pct"] = np.where(g["in_domain_total"] > 0, 100.0 * g["docs"] / g["in_domain_total"], 0.0)
        cy = pd.concat([cy, g[cy.columns]], ignore_index=True)
        gt = totals.groupby(["year", "partial_year"], as_index=False)[["fetched", "in_domain", "in_domain_analysed", "source_total"]].sum()
        gt["source"], gt["scale"], gt["abstract_rate"] = "all", 1.0, float("nan")
        totals = pd.concat([totals, gt[totals.columns]], ignore_index=True)

    # ------------------------------------------------------------------ growth
    growth_rows = []
    for src, sub in cy.groupby("source"):
        complete = sorted(y for y in sub["year"].unique() if y != this_year)
        recent_years = complete[-recent_window:]
        prior_years = complete[-2 * recent_window:-recent_window]
        for k, cs in sub.groupby("concept"):
            cs = cs.sort_values("year")
            comp = cs[cs["year"].isin(complete)]
            rec = comp[comp["year"].isin(recent_years)]
            pri = comp[comp["year"].isin(prior_years)]
            share_rec = 100.0 * rec["docs"].sum() / max(1.0, rec["in_domain_total"].sum())
            share_pri = 100.0 * pri["docs"].sum() / max(1.0, pri["in_domain_total"].sum())
            eps = 100.0 / max(1.0, pri["in_domain_total"].sum())
            nonzero = comp[comp["docs_raw"] >= 3]
            first_year = int(nonzero["year"].min()) if len(nonzero) else None
            peak = comp.loc[comp["share_pct"].idxmax()] if len(comp) and comp["share_pct"].max() > 0 else None
            cat, label = concept_meta[k]
            growth_rows.append({
                "source": src, "category": cat, "concept": k, "label": label,
                "docs_total": int(round(cs["docs"].sum())), "docs_recent": int(round(rec["docs"].sum())), "docs_prior": int(round(pri["docs"].sum())),
                "share_recent_pct": share_rec, "share_prior_pct": share_pri,
                "growth_ratio": (share_rec + eps) / (share_pri + eps), "delta_pp": share_rec - share_pri,
                "slope_pp_per_year": _slope(list(comp["year"]), list(comp["share_pct"])),
                "first_year_ge3": first_year, "peak_year": int(peak["year"]) if peak is not None else None,
                "recent_years": f"{recent_years[0]}-{recent_years[-1]}" if recent_years else "",
                "prior_years": f"{prior_years[0]}-{prior_years[-1]}" if prior_years else "",
            })
    growth = pd.DataFrame(growth_rows).sort_values(["source", "growth_ratio"], ascending=[True, False])

    # ------------------------------------------------------------------ co-occurrence with lift
    if len(co):
        co_tot = co.groupby(["source", "concept_a", "concept_b"], as_index=False)["docs_both"].sum()
        doc_tot = cy.groupby(["source", "concept"], as_index=False)["docs_raw"].sum().rename(columns={"docs_raw": "n"})
        n_dom = totals.groupby("source", as_index=False)["in_domain"].sum()
        co_tot = co_tot.merge(doc_tot.rename(columns={"concept": "concept_a", "n": "docs_a"}), on=["source", "concept_a"])
        co_tot = co_tot.merge(doc_tot.rename(columns={"concept": "concept_b", "n": "docs_b"}), on=["source", "concept_b"])
        co_tot = co_tot.merge(n_dom, on="source")
        co_tot["lift"] = co_tot["docs_both"] * co_tot["in_domain"] / (co_tot["docs_a"] * co_tot["docs_b"]).clip(lower=1)
        co_tot["jaccard"] = co_tot["docs_both"] / (co_tot["docs_a"] + co_tot["docs_b"] - co_tot["docs_both"]).clip(lower=1)
        co_tot = co_tot.sort_values(["source", "docs_both"], ascending=[True, False])
    else:
        co_tot = pd.DataFrame(columns=["source", "concept_a", "concept_b", "docs_both", "docs_a", "docs_b", "in_domain", "lift", "jaccard"])

    # ------------------------------------------------------------------ emerging terms
    for src, ng in acc.ngrams.items():
        complete = sorted(y for y in ng if y != this_year)
        recent_years = complete[-recent_window:]
        prior_years = complete[-2 * recent_window:-recent_window]
        if not recent_years or not prior_years:
            continue
        rows = discover.burst_scores(ng, acc.n_in_domain[src], recent_years, prior_years, min_recent_df=15, top_k=80)
        for r in rows:
            covered = matcher.match(normalize(r["term"]))
            acc.em.append({"source": src, **r, "covered_by": ",".join(sorted(covered)) if covered else "",
                           "recent_years": f"{recent_years[0]}-{recent_years[-1]}", "prior_years": f"{prior_years[0]}-{prior_years[-1]}"})
    emerging = pd.DataFrame(acc.em)

    # ------------------------------------------------------------------ write
    totals.to_csv(TABLES / "totals.csv", index=False)
    cy.sort_values(["source", "category", "concept", "year"]).to_csv(TABLES / "concept_year.csv", index=False)
    syn.sort_values(["source", "concept", "year", "docs"], ascending=[True, True, True, False]).to_csv(TABLES / "synonym_year.csv", index=False)
    growth.to_csv(TABLES / "growth.csv", index=False)
    co_tot.to_csv(TABLES / "cooccurrence.csv", index=False)
    ex.to_csv(TABLES / "examples.csv", index=False)
    emerging.to_csv(TABLES / "emerging.csv", index=False)
    base = totals[totals["source"] == "all"] if "s2" in raw_files else totals[~totals["source"].isin(["all"])]
    summary = {
        "run_at": dt.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "sources": {s: [str(p) for p in ps] for s, ps in raw_files.items()},
        "mode": "s2" if "s2" in raw_files else "plain",
        "docs_analysed": int(base["fetched"].sum()),
        "docs_in_domain_analysed": int(base["in_domain_analysed"].sum()),
        "corpus_in_domain_est": float(base["in_domain"].sum()),
        "years": [int(y) for y in sorted(cy["year"].unique())],
        "partial_year": this_year,
        "recent_window": recent_window,
        "n_concepts": len(matcher.concepts),
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[analyze] done: {summary['docs_in_domain_analysed']} in-domain docs analysed; est. corpus {summary['corpus_in_domain_est']:.0f}; tables in {TABLES}")
