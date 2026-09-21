"""Concept / synonym matcher.

Loads ``config/keywords.json`` and compiles every synonym into a regex.  Rules:

* Plain phrase containing lowercase letters  -> case-insensitive.  Each space or
  hyphen in the phrase becomes ``[\\s\\-]*`` so "gate all around" matches
  "gate-all-around" *and* "gateallaround"; "nano sheet" matches "nanosheet".
  A trailing ``(?:s|es)?`` allows plurals.
* Token with **no** lowercase letters ("GAA", "HBM3", "3D IC") -> case-sensitive
  whole-word match.  This avoids matching "gaa" inside ordinary words and keeps
  short acronyms (PIM, CIM, RET ...) from firing on unrelated lowercase text.
* ``re:`` prefix -> raw regex used as-is; a leading ``(?i)`` makes it
  case-insensitive.
* Optional per-concept ``exclude`` regex: a document that matches it is not
  counted for that concept.
* Optional per-concept ``require`` regex: the document must ALSO match it
  (context gate - e.g. "nanosheet" only counts as GAA when the text talks
  about transistors / FETs, not MoS2 nanosheets in a catalysis paper).

For speed every concept is compiled into ONE alternation regex with a named
group per synonym, so matching a document costs one ``finditer`` per concept.
Word boundaries use look-arounds on ``[A-Za-z0-9]`` instead of ``\\b`` so that
"MoS2" and "Ga2O3" behave sensibly.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "config" / "keywords.json"

_LB = r"(?<![A-Za-z0-9])"
_RB = r"(?![A-Za-z0-9])"


def synonym_regex(s: str) -> str:
    """Return a regex *fragment* (no anchors, flags scoped inline) for one synonym."""
    if s.startswith("re:"):
        body = s[3:]
        if body.startswith("(?i)"):
            return "(?i:" + body[4:] + ")"
        return "(?:" + body + ")"
    has_lower = any(c.islower() for c in s)
    parts = [p for p in re.split(r"[\s\-]+", s.strip()) if p]
    body = r"[\s\-]*".join(re.escape(p) for p in parts)
    if has_lower:
        return "(?i:" + _LB + body + r"(?:s|es)?" + _RB + ")"
    return "(?:" + _LB + body + r"s?" + _RB + ")"


def compile_synonym(s: str) -> re.Pattern:
    return re.compile(synonym_regex(s))


@dataclass
class Concept:
    key: str
    label: str
    category: str
    synonyms: list[str]
    combined: re.Pattern = field(repr=False, default=None)  # type: ignore[assignment]
    exclude: re.Pattern | None = field(repr=False, default=None)
    require: re.Pattern | None = field(repr=False, default=None)

    def build(self) -> "Concept":
        alts = [f"(?P<s{i}>{synonym_regex(s)})" for i, s in enumerate(self.synonyms)]
        self.combined = re.compile("|".join(alts))
        return self

    def hits(self, text: str) -> dict[str, int]:
        """{synonym: occurrences}; empty if excluded, if the context requirement fails, or no hit."""
        counts: dict[str, int] = {}
        for m in self.combined.finditer(text):
            g = m.lastgroup
            if g:
                syn = self.synonyms[int(g[1:])]
                counts[syn] = counts.get(syn, 0) + 1
        if not counts:
            return {}
        if self.exclude is not None and self.exclude.search(text):
            return {}
        if self.require is not None and not self.require.search(text):
            return {}
        return counts


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_concepts(path: str | Path = DEFAULT_CONFIG) -> list[Concept]:
    cfg = load_config(path)
    concepts: list[Concept] = []
    for cat, items in cfg["categories"].items():
        for key, spec in items.items():
            c = Concept(
                key=key, label=spec.get("label", key), category=cat, synonyms=list(spec["synonyms"]),
                exclude=compile_synonym(spec["exclude"]) if spec.get("exclude") else None,
                require=compile_synonym(spec["require"]) if spec.get("require") else None,
            ).build()
            concepts.append(c)
    return concepts


def domain_concept(path: str | Path = DEFAULT_CONFIG) -> Concept:
    cfg = load_config(path)
    return Concept(key="DOMAIN", label="in-domain gate", category="_", synonyms=list(cfg["domain_terms"])).build()


class Matcher:
    """Match a normalised text against all concepts.

    ``match(text)`` returns ``{concept_key: {synonym: hit_count}}`` for concepts
    with at least one hit.  Per-synonym counts let the report show *which*
    surface form dominates in each year (e.g. "nanosheet" vs "GAA").
    """

    def __init__(self, config: str | Path = DEFAULT_CONFIG):
        self.config = Path(config)
        self.concepts = load_concepts(config)
        self.by_key = {c.key: c for c in self.concepts}
        self.domain = domain_concept(config)

    def in_domain(self, text: str) -> bool:
        return self.domain.combined.search(text) is not None

    def match(self, text: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for c in self.concepts:
            h = c.hits(text)
            if h:
                out[c.key] = h
        return out

    def keys(self) -> list[str]:
        return [c.key for c in self.concepts]


if __name__ == "__main__":  # quick self-test
    from .normalize import normalize

    m = Matcher()
    samples = [
        "We demonstrate gate-all-around nanosheet FETs at the 2 nm node using EUV lithography.",
        "A 2.5D CoWoS package with HBM3E stacks and through‐silicon vias (TSVs).",
        "MoS₂ transistors and $\\mathrm{Ga_2O_3}$ Schottky diodes for ultra-wide-bandgap power electronics.",
        "The gaa protein binds to the pim kinase.",  # should NOT match GAA / PIM
    ]
    for s in samples:
        t = normalize(s)
        print(s, "-> domain:", m.in_domain(t), m.match(t))
