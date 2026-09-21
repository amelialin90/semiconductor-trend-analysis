"""Command-line entry point.

    python -m semitrend fetch   --source ieee arxiv --years 2014-2026 [--max-per-year 0]
    python -m semitrend analyze [--sources arxiv ieee google_patents] [--workers 4]
    python -m semitrend report
    python -m semitrend all
    python -m semitrend test-keywords "some text to match"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .matcher import DEFAULT_CONFIG


def _years(spec: str) -> range:
    a, _, b = spec.partition("-")
    a = int(a)
    b = int(b) if b else a
    return range(a, b + 1)


def _load_cfg(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_fetch(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)
    terms = cfg["domain_terms"]
    years = _years(args.years)
    mailto = args.mailto or os.environ.get("OPENALEX_MAILTO", "")
    paper_sources = [s for s in args.source if s in ("arxiv", "ieee")]
    if paper_sources and args.via == "s2":
        from .sources import semanticscholar
        shard, _, nshards = args.shard.partition("/")
        n = semanticscholar.fetch(years, args.config, sample_pages=args.sample_pages,
                                  max_pages_concept=args.max_pages_concept, mailto=mailto,
                                  shard=int(shard), nshards=int(nshards or 1))
        print(f"[fetch] arxiv+ieee: {n} records written via Semantic Scholar")
    for src in args.source:
        if src in ("arxiv", "ieee") and args.via == "s2":
            continue
        if src in ("arxiv", "ieee") and args.via == "openalex":
            from .sources import openalex
            n, _ = openalex.fetch(src, years, terms, max_per_year=args.max_per_year, seed=args.seed, mailto=mailto)
            print(f"[fetch] {src}: {n} docs written via OpenAlex")
        elif src == "arxiv":
            from .sources import arxiv_api
            n = arxiv_api.fetch(years, terms, max_per_year=args.max_per_year)
            print(f"[fetch] arxiv: {n} docs written via arXiv API")
        elif src == "ieee":
            from .sources import ieee_xplore
            n = ieee_xplore.fetch(years, terms, max_per_year=args.max_per_year)
            print(f"[fetch] ieee: {n} docs written via IEEE Xplore API")
        elif src == "google_patents":
            from .sources import google_patents
            n = google_patents.fetch(years, terms, max_per_year=args.max_per_year)
            print(f"[fetch] google_patents: {n} docs written")
        else:
            sys.exit(f"unknown source {src}")


def cmd_analyze(args: argparse.Namespace) -> None:
    from . import analyze
    analyze.run(sources=args.sources, config=args.config, workers=args.workers, recent_window=args.recent_window)


def cmd_report(args: argparse.Namespace) -> None:
    from . import report
    out = report.build(config=args.config)
    print(f"[report] written {out}")


def cmd_all(args: argparse.Namespace) -> None:
    cmd_fetch(args)
    cmd_analyze(args)
    cmd_report(args)


def cmd_test(args: argparse.Namespace) -> None:
    from .matcher import Matcher
    from .normalize import normalize
    m = Matcher(config=args.config)
    text = normalize(" ".join(args.text))
    print("normalised:", text)
    for k, hits in m.match(text).items():
        print(f"  {k:14s} <- {hits}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="semitrend", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="keywords.json path")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download raw documents into data/raw/")
    f.add_argument("--source", nargs="+", default=["ieee", "arxiv"], choices=["arxiv", "ieee", "google_patents"])
    f.add_argument("--years", default="2014-2026", help="e.g. 2014-2026")
    f.add_argument("--max-per-year", type=int, default=0, help="0 = fetch all; N = random sample of N per source-year")
    f.add_argument("--seed", type=int, default=42)
    f.add_argument("--mailto", default="", help="contact email sent in the User-Agent / polite pool (or env OPENALEX_MAILTO)")
    f.add_argument("--via", default="s2", choices=["s2", "openalex", "direct"],
                   help="route for arxiv/ieee: s2 = Semantic Scholar (default, no key), openalex (small free daily budget), "
                        "direct = arXiv API / IEEE Xplore API (needs IEEE_API_KEY; arXiv API often 429s)")
    f.add_argument("--sample-pages", type=int, default=10, help="s2: pages x1000 of the domain query sampled per year")
    f.add_argument("--max-pages-concept", type=int, default=3, help="s2: pages x1000 fetched per concept-year")
    f.add_argument("--shard", default="0/1", help="s2: run concept queries for shard i of n (e.g. 1/3) in parallel processes")
    f.set_defaults(func=cmd_fetch)

    a = sub.add_parser("analyze", help="match keywords and build trend tables")
    a.add_argument("--sources", nargs="*", default=None, help="subset of sources (default: every raw file present)")
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a.add_argument("--recent-window", type=int, default=3, help="years in the 'recent' window for growth scoring")
    a.set_defaults(func=cmd_analyze)

    r = sub.add_parser("report", help="render output/report.html and charts from the tables")
    r.set_defaults(func=cmd_report)

    al = sub.add_parser("all", help="fetch + analyze + report")
    for parser in (al,):
        parser.add_argument("--source", nargs="+", default=["ieee", "arxiv"], choices=["arxiv", "ieee", "google_patents"])
        parser.add_argument("--years", default="2014-2026")
        parser.add_argument("--max-per-year", type=int, default=0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--mailto", default="")
        parser.add_argument("--via", default="s2", choices=["s2", "openalex", "direct"])
        parser.add_argument("--sample-pages", type=int, default=10)
        parser.add_argument("--max-pages-concept", type=int, default=3)
        parser.add_argument("--shard", default="0/1")
        parser.add_argument("--sources", nargs="*", default=None)
        parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        parser.add_argument("--recent-window", type=int, default=3)
    al.set_defaults(func=cmd_all)

    t = sub.add_parser("test-keywords", help="show which concepts a piece of text matches")
    t.add_argument("text", nargs="+")
    t.set_defaults(func=cmd_test)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
