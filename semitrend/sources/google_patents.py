"""Google Patents adapter.

Google Patents has no public REST API.  Two supported routes:

1. **BigQuery public dataset** ``patents-public-data.patents.publications``
   (title + abstract + CPC + dates for every publication world-wide).
   Needs ``pip install google-cloud-bigquery`` and Google Cloud credentials
   (``gcloud auth application-default login`` or ``GOOGLE_APPLICATION_CREDENTIALS``)
   plus a project id in ``GOOGLE_CLOUD_PROJECT``.  The free tier (1 TB scanned
   per month) is more than enough: the query below scans a few GB.

2. **CSV export from patents.google.com** - run a query on the website such as::

       (semiconductor OR transistor OR lithography) CPC=H01L after:priority:20200101 before:priority:20201231

   click *Download* (CSV) and drop the file(s) into ``data/raw/google_patents/``.
   The export has *titles and dates only* (no abstract), so keyword matching is
   title-only for this route - useful for rough trends, but expect lower hit
   rates than the abstract-bearing sources.  The importer figures out the year
   from the *priority date* column (falls back to filing / publication date).

``fetch()`` tries BigQuery first, otherwise imports any CSVs present.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path

from ..schema import Doc, RAW_DIR

CSV_DIR = RAW_DIR / "google_patents"

# CPC groups that define "semiconductor" for the patent corpus.
CPC_PREFIXES = ("H01L", "H10B", "H10D", "H10K", "H10N", "G03F", "H05K", "G11C", "B81C", "H01S5")

BQ_SQL = """
WITH base AS (
  SELECT
    p.publication_number,
    p.country_code,
    p.kind_code,
    p.priority_date,
    p.filing_date,
    p.publication_date,
    (SELECT text FROM UNNEST(p.title_localized) WHERE language = 'en' LIMIT 1) AS title,
    (SELECT text FROM UNNEST(p.abstract_localized) WHERE language = 'en' LIMIT 1) AS abstract,
    ARRAY(SELECT code FROM UNNEST(p.cpc)) AS cpc_codes
  FROM `patents-public-data.patents.publications` p
  WHERE p.priority_date BETWEEN @y0 AND @y1
    AND EXISTS (SELECT 1 FROM UNNEST(p.cpc) c WHERE REGEXP_CONTAINS(c.code, @cpc_re))
    AND EXISTS (SELECT 1 FROM UNNEST(p.abstract_localized) a WHERE a.language = 'en')
    -- one publication per family: keep the first English-abstract publication
    AND p.kind_code IN ('A1', 'A', 'B1', 'B2', 'A2')
)
SELECT * FROM base
WHERE title IS NOT NULL
  AND (@limit = 0 OR MOD(ABS(FARM_FINGERPRINT(publication_number)), @denom) = 0)
"""


def _bigquery_fetch(years: range, max_per_year: int, out: Path) -> int | None:
    try:
        from google.cloud import bigquery  # type: ignore
    except ImportError:
        return None
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return None
    try:
        client = bigquery.Client(project=project) if project else bigquery.Client()
    except Exception as e:  # noqa: BLE001 - no creds
        print(f"[google_patents] BigQuery client unavailable: {e}")
        return None

    cpc_re = "^(" + "|".join(CPC_PREFIXES) + ")"
    meta: dict = {"source": "google_patents", "route": "bigquery", "cpc": CPC_PREFIXES, "years": {}}
    n_total = 0
    with out.open("w", encoding="utf-8") as fh:
        for year in years:
            y0, y1 = int(f"{year}0101"), int(f"{year}1231")
            # count first so we can sample deterministically
            cnt_job = client.query(
                "SELECT COUNT(1) n FROM `patents-public-data.patents.publications` p "
                "WHERE p.priority_date BETWEEN @y0 AND @y1 "
                "AND EXISTS (SELECT 1 FROM UNNEST(p.cpc) c WHERE REGEXP_CONTAINS(c.code, @cpc_re)) "
                "AND EXISTS (SELECT 1 FROM UNNEST(p.abstract_localized) a WHERE a.language='en') "
                "AND p.kind_code IN ('A1','A','B1','B2','A2')",
                job_config=bigquery.QueryJobConfig(query_parameters=[
                    bigquery.ScalarQueryParameter("y0", "INT64", y0),
                    bigquery.ScalarQueryParameter("y1", "INT64", y1),
                    bigquery.ScalarQueryParameter("cpc_re", "STRING", cpc_re),
                ]))
            total = int(list(cnt_job.result())[0]["n"])
            denom = max(1, total // max_per_year) if max_per_year else 1
            job = client.query(BQ_SQL, job_config=bigquery.QueryJobConfig(query_parameters=[
                bigquery.ScalarQueryParameter("y0", "INT64", y0),
                bigquery.ScalarQueryParameter("y1", "INT64", y1),
                bigquery.ScalarQueryParameter("cpc_re", "STRING", cpc_re),
                bigquery.ScalarQueryParameter("limit", "INT64", max_per_year),
                bigquery.ScalarQueryParameter("denom", "INT64", denom),
            ]))
            fetched = 0
            for row in job.result():
                pd_ = str(row["priority_date"])
                doc = Doc(
                    source="google_patents", id=row["publication_number"], title=row["title"] or "",
                    abstract=row["abstract"] or "", year=int(pd_[:4]),
                    date=f"{pd_[:4]}-{pd_[4:6]}-{pd_[6:8]}" if len(pd_) == 8 else "",
                    venue=",".join(sorted({c[:4] for c in row["cpc_codes"]})), doc_type="patent",
                    url=f"https://patents.google.com/patent/{row['publication_number']}",
                    extra={"country": row["country_code"], "kind": row["kind_code"], "cpc": list(row["cpc_codes"])[:20]},
                )
                fh.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")
                fetched += 1
            meta["years"][str(year)] = {"total": total, "fetched": fetched}
            n_total += fetched
            print(f"[google_patents:bq] {year}: total={total} fetched={fetched}", flush=True)
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n_total


_DATE_COLS = ("priority date", "filing/creation date", "filing date", "publication date", "grant date")


def _csv_import(out: Path) -> int:
    files = sorted(CSV_DIR.glob("*.csv"))
    if not files:
        return 0
    n = 0
    years: dict[int, int] = {}
    seen: set[str] = set()
    with out.open("w", encoding="utf-8") as fh:
        for f in files:
            with f.open(encoding="utf-8-sig", newline="") as src:
                # Google's export starts with a 'search URL:' line before the real header;
                # skip it when present, otherwise rewind so the header is read normally.
                first = src.readline()
                if first.lower().startswith("id,"):
                    src.seek(0)
                reader = csv.DictReader(src)
                cols = {c.lower().strip(): c for c in reader.fieldnames or []}
                date_col = next((cols[c] for c in _DATE_COLS if c in cols), None)
                abs_col = cols.get("abstract")
                for row in reader:
                    pid = (row.get(cols.get("id", "id"), "") or "").strip()
                    if not pid or pid in seen:
                        continue
                    seen.add(pid)
                    raw_date = (row.get(date_col, "") if date_col else "") or ""
                    m = re.search(r"(\d{4})", raw_date)
                    if not m:
                        continue
                    year = int(m.group(1))
                    doc = Doc(
                        source="google_patents", id=pid, title=row.get(cols.get("title", "title"), "") or "",
                        abstract=(row.get(abs_col, "") if abs_col else "") or "", year=year, date=raw_date,
                        venue="csv-export", doc_type="patent",
                        url=row.get(cols.get("result link", "result link"), "") or f"https://patents.google.com/patent/{pid}",
                        extra={"assignee": row.get(cols.get("assignee", "assignee"), ""), "file": f.name},
                    )
                    fh.write(json.dumps(doc.__dict__, ensure_ascii=False) + "\n")
                    years[year] = years.get(year, 0) + 1
                    n += 1
    meta = {"source": "google_patents", "route": "csv", "files": [f.name for f in files],
            "years": {str(y): {"total": c, "fetched": c} for y, c in sorted(years.items())},
            "note": "CSV export has titles only unless an 'abstract' column is present"}
    out.with_name(out.stem + "_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return n


def fetch(years: range, domain_terms: list[str], max_per_year: int = 0) -> int:
    out = RAW_DIR / "google_patents.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = _bigquery_fetch(years, max_per_year, out)
    if n is not None:
        return n
    n = _csv_import(out)
    if n:
        print(f"[google_patents] imported {n} rows from CSV exports in {CSV_DIR}")
        return n
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    raise SystemExit(
        "google_patents: no BigQuery credentials (set GOOGLE_CLOUD_PROJECT + run `gcloud auth application-default login`, "
        f"pip install google-cloud-bigquery) and no CSV exports found in {CSV_DIR}. See semitrend/sources/google_patents.py."
    )
