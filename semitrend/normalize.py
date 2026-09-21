"""Text normalisation for English technical text.

The goal is to make the many surface forms of the same technical term collapse
onto one canonical spelling *before* regex matching, so the synonym dictionary
stays small:

* Unicode dashes / hyphens (‐ ‑ ‒ – — −) -> ASCII '-'
* Unicode sub/superscript digits (MoS₂, Ga₂O₃)  -> ASCII digits (MoS2, Ga2O3)
* Ligatures (ﬁ, ﬂ) and NBSP / zero-width chars -> plain ASCII
* HTML / LaTeX leftovers common in arXiv abstracts ($\\mathrm{MoS}_2$ -> MoS2)
* Whitespace collapsed to single spaces
"""

from __future__ import annotations

import html
import re
import unicodedata

_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212\u2015\u00ad"), "-")
_SUBSUP = {
    **{ord(c): str(i) for i, c in enumerate("₀₁₂₃₄₅₆₇₈₉")},
    **{ord(c): str(i) for i, c in enumerate("⁰¹²³⁴⁵⁶⁷⁸⁹")},
}
_MISC = {
    ord("\u00a0"): " ",  # nbsp
    ord("\u200b"): "",   # zero-width space
    ord("\ufeff"): "",   # BOM
    ord("ﬁ"): "fi",
    ord("ﬂ"): "fl",
    ord("’"): "'",
    ord("‘"): "'",
    ord("“"): '"',
    ord("”"): '"',
    ord("/"): "/",
}
_TABLE = {**_DASHES, **_SUBSUP, **_MISC}

# LaTeX-ish patterns that show up in arXiv abstracts.
_LATEX_CMD = re.compile(r"\\(?:mathrm|mathbf|textrm|text|rm|it|emph|textbf|textit|ce|chem)\s*\{([^{}]*)\}")
_SUBSCRIPT = re.compile(r"_\{?(\d+(?:\.\d+)?)\}?")          # Ga_2O_3 / Ga_{2}O_{3} -> Ga2O3
_SUPERSCRIPT = re.compile(r"\^\{?([^{}\s]{1,6})\}?")          # 10^{12} -> 10 12 (keep readable)
_DOLLAR = re.compile(r"\$+")
_TAGS = re.compile(r"<[^>]{1,40}>")
_WS = re.compile(r"\s+")
_LATEX_SYM = {
    r"\beta": "beta", r"\alpha": "alpha", r"\gamma": "gamma", r"\mu": "u",
    r"\AA": "A", r"\%": "%", r"\&": "&", r"\_": "_", r"\,": " ", r"\;": " ", r"\!": "",
}


def normalize(text: str | None) -> str:
    """Return a normalised copy of *text* suitable for keyword matching."""
    if not text:
        return ""
    t = html.unescape(text)
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(_TABLE)
    t = _TAGS.sub(" ", t)
    for k, v in _LATEX_SYM.items():
        t = t.replace(k, v)
    # Apply the \mathrm{} unwrap a couple of times to handle nesting.
    for _ in range(2):
        t = _LATEX_CMD.sub(r"\1", t)
    t = _SUBSCRIPT.sub(r"\1", t)
    t = _SUPERSCRIPT.sub(r" \1 ", t)
    t = _DOLLAR.sub("", t)
    t = t.replace("{", "").replace("}", "")
    t = _WS.sub(" ", t).strip()
    return t


def tokenize(text: str) -> list[str]:
    """Lower-cased word tokens; keeps things like 'mos2', '3d', 'gan', 'risc-v'."""
    return re.findall(r"[a-z0-9][a-z0-9\-\.]*[a-z0-9]|[a-z0-9]", text.lower())
