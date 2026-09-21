"""Emerging-term discovery: find n-grams whose document frequency is *bursting*.

This is the safety net for the synonym problem: the taxonomy in
``config/keywords.json`` is hand-written, so new surface forms (a new product
name, a new acronym) are invisible to it.  Here we count 1/2/3-gram document
frequencies per year, then score each n-gram by how much its share of documents
grew in the recent window versus the prior window.  Terms that are *not*
already matched by any concept are flagged as candidate synonyms / new topics.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .normalize import tokenize

STOP = set("""
a an the and or of in on at to for with by from as is are was were be been being this that these those it its into
via using used use based novel new approach method methods paper propose proposed present presented presents study
results result show shows shown we our can which than then when where while also both between over under about
such more most high low large small first two three one due each per non only well very further however moreover
between within without through during after before above below across among against several various different
achieve achieved achieves obtain obtained provides provide demonstrate demonstrated demonstrates investigate
investigated compared comparison respectively performance proposed-based total value values number time
significantly effect effects analysis model models design designs device devices technique techniques data
system systems application applications process processes structure structures material materials layer layers
work works et al fig table
""".split())

_NUMERIC = re.compile(r"^[\d\.\-]+$")


def doc_ngrams(text: str, max_n: int = 3) -> set[str]:
    """Set of 1..max_n-grams (space-joined, lower-case) for one document."""
    toks = tokenize(text)
    out: set[str] = set()
    n = len(toks)
    for i, t in enumerate(toks):
        if t in STOP or _NUMERIC.match(t) or len(t) < 3:
            continue
        if len(t) >= 4:
            out.add(t)
        for k in range(2, max_n + 1):
            if i + k > n:
                break
            last = toks[i + k - 1]
            if last in STOP or _NUMERIC.match(last) or len(last) < 2:
                continue
            gram = toks[i:i + k]
            if any(_NUMERIC.match(g) for g in gram):
                continue
            out.add(" ".join(gram))
    return out


def count_year(texts: list[str], min_df: int = 3) -> Counter:
    """Document-frequency Counter of n-grams over *texts*, pruned to df >= min_df."""
    c: Counter = Counter()
    for t in texts:
        c.update(doc_ngrams(t))
    return Counter({g: n for g, n in c.items() if n >= min_df})


def burst_scores(
    df_by_year: dict[int, Counter],
    n_docs_by_year: dict[int, int],
    recent_years: list[int],
    prior_years: list[int],
    min_recent_df: int = 15,
    top_k: int = 80,
) -> list[dict]:
    """Rank n-grams by growth of document share (recent vs prior window)."""
    n_rec = sum(n_docs_by_year.get(y, 0) for y in recent_years) or 1
    n_pri = sum(n_docs_by_year.get(y, 0) for y in prior_years) or 1
    rec: Counter = Counter()
    pri: Counter = Counter()
    for y in recent_years:
        rec.update(df_by_year.get(y, Counter()))
    for y in prior_years:
        pri.update(df_by_year.get(y, Counter()))
    rows = []
    for g, d in rec.items():
        if d < min_recent_df:
            continue
        r = d / n_rec
        p = pri.get(g, 0) / n_pri
        eps = 1.0 / n_pri  # add-one smoothing on the prior window
        ratio = (r + eps) / (p + eps)
        # weight by log(recent df) so that terms with real volume rank above tiny spikes
        score = math.log(ratio) * math.log1p(d)
        rows.append({"term": g, "recent_df": d, "prior_df": pri.get(g, 0), "recent_share_pct": 100 * r,
                     "prior_share_pct": 100 * p, "growth_ratio": ratio, "score": score})
    rows.sort(key=lambda x: x["score"], reverse=True)
    # de-duplicate: drop an n-gram if a longer n-gram containing it ranks higher with similar df
    kept: list[dict] = []
    for row in rows:
        if any(row["term"] in k["term"] and k["recent_df"] >= 0.7 * row["recent_df"] for k in kept):
            continue
        kept.append(row)
        if len(kept) >= top_k:
            break
    return kept
