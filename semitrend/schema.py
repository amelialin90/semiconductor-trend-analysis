"""Unified document schema shared by every data source, plus JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


@dataclass
class Doc:
    source: str          # "arxiv" | "ieee" | "google_patents"
    id: str              # source-native id (arXiv id, DOI, patent publication number)
    title: str
    abstract: str
    year: int
    date: str = ""       # ISO date if known
    venue: str = ""      # journal / conference / CPC main class
    doc_type: str = ""   # "paper" | "patent"
    url: str = ""
    extra: dict | None = None

    def text(self) -> str:
        return f"{self.title}. {self.abstract}"


def write_jsonl(path: str | Path, docs: Iterable[Doc], mode: str = "w") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open(mode, encoding="utf-8") as fh:
        for d in docs:
            fh.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[Doc]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Doc(**json.loads(line))
