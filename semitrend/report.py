"""Render the report from the tables.

Writes
  output/report.html           self-contained page (charts embedded as PNG, light + dark renders)
  output/report_artifact.html  the same page as a <title>+<style>+content fragment for publishing as an Artifact
  output/charts/*.{light,dark}.png
"""

from __future__ import annotations

import base64
import contextlib
import html
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .matcher import DEFAULT_CONFIG, load_config  # noqa: E402
from .schema import OUTPUT_DIR  # noqa: E402

TABLES = OUTPUT_DIR / "tables"
CHARTS = OUTPUT_DIR / "charts"
SOURCE_LABEL = {"arxiv": "arXiv", "ieee": "IEEE Xplore", "google_patents": "Google Patents", "all": "All publishers"}
HEADLINE = ["GAA", "CoWoS", "EUV", "Chiplet", "HBM", "HybridBonding", "CIM", "AIAccel"]
SYNONYM_SHOWCASE = ["GAA", "3DIC", "CIM", "EUV", "Chiplet", "RRAM"]

# ----------------------------------------------------------------------------- chart themes
# Validated categorical palette (fixed order, CVD-safe on adjacent pairs); >8 series fold into "Other".
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  palette=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
                  other="#898781", good="#0ca30c", critical="#d03b3b",
                  seq=["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 palette=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
                 other="#898781", good="#0ca30c", critical="#d03b3b",
                 seq=["#1a1a19", "#0d366b", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"]),
}
T = THEMES["light"]


@contextlib.contextmanager
def _theme(name: str):
    global T
    T = THEMES[name]
    with plt.rc_context({
        "font.family": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"], "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": T["axis"], "axes.labelcolor": T["ink2"],
        "xtick.color": T["muted"], "ytick.color": T["muted"], "text.color": T["ink"], "axes.titlecolor": T["ink"],
        "axes.grid": True, "grid.color": T["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"], "savefig.facecolor": T["surface"],
        "figure.dpi": 110, "legend.frameon": False, "legend.labelcolor": T["ink2"],
        "lines.linewidth": 2.0, "lines.markersize": 5,
    }):
        yield
    T = THEMES["light"]


def _colors(n: int) -> list[str]:
    return [T["palette"][i] if i < len(T["palette"]) else T["other"] for i in range(n)]


def _render(fn, name: str, *args, **kw) -> dict[str, str]:
    """Run a chart function under both themes; save PNGs; return {'light': b64, 'dark': b64}."""
    CHARTS.mkdir(parents=True, exist_ok=True)
    out = {}
    for th in ("light", "dark"):
        with _theme(th):
            fig = fn(*args, **kw)
            if fig is None:
                return {}
            fig.savefig(CHARTS / f"{name}.{th}.png", bbox_inches="tight")
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            out[th] = base64.b64encode(buf.getvalue()).decode()
    return out


def _img(pair: dict[str, str], alt: str) -> str:
    if not pair:
        return ""
    a = html.escape(alt)
    return (f'<img class="light" alt="{a}" src="data:image/png;base64,{pair["light"]}">'
            f'<img class="dark" alt="{a}" src="data:image/png;base64,{pair["dark"]}">')


def _partial_band(ax, partial_year: int) -> None:
    ax.axvspan(partial_year - 0.5, partial_year + 0.5, color=T["muted"], alpha=0.10, lw=0)


# ----------------------------------------------------------------------------- charts (each returns a Figure)
def chart_lines(cy, concepts, sources, title, partial_year, labels):
    vol = cy[cy["concept"].isin(concepts)].groupby("concept")["docs"].sum()
    concepts = sorted(concepts, key=lambda k: -float(vol.get(k, 0)))[:len(T["palette"])]
    colors = _colors(len(concepts))
    ncols = len(sources)
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 3.4), squeeze=False)
    for ax, src in zip(axes[0], sources):
        sub = cy[(cy["source"] == src) & (cy["concept"].isin(concepts))]
        ends: list[tuple[float, int, str]] = []
        for i, k in enumerate(concepts):
            s = sub[sub["concept"] == k].sort_values("year")
            if s.empty:
                continue
            full = s[s["year"] != partial_year]
            part = s[s["year"] >= partial_year - 1]
            ax.plot(full["year"], full["share_pct"], marker="o", color=colors[i], label=labels.get(k, k),
                    markeredgecolor=T["surface"], markeredgewidth=1.2)
            if len(part) == 2:
                ax.plot(part["year"], part["share_pct"], ls="--", lw=1.4, color=colors[i], alpha=0.75)
            if len(full) and full["share_pct"].iloc[-1] > 0:
                ends.append((float(full["share_pct"].iloc[-1]), int(full["year"].iloc[-1]), labels.get(k, k)[:16]))
        # direct-label the four largest series at their last complete year (legend carries the rest)
        for v, yr, lbl in sorted(ends, reverse=True)[:4]:
            ax.annotate(lbl, (yr, v), xytext=(4, 0), textcoords="offset points", fontsize=6.5, color=T["ink2"], va="center")
        ax.set_title(SOURCE_LABEL.get(src, src), fontsize=10, loc="left", fontweight="bold")
        ax.set_ylabel("% of in-domain documents")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _partial_band(ax, partial_year)
    handles, lbls = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, lbls = h, l
    fig.legend(handles, lbls, loc="upper center", ncol=min(len(concepts), 4), bbox_to_anchor=(0.5, 1.13), fontsize=8)
    fig.suptitle(title, y=1.2, fontsize=11)
    return fig


def chart_heatmap(cy, src, labels, partial_year):
    sub = cy[cy["source"] == src]
    piv = sub.pivot_table(index="concept", columns="year", values="share_pct", aggfunc="sum").fillna(0)
    vol = sub.groupby("concept")["docs"].sum()
    order = vol[vol > 0].sort_values(ascending=False).index
    piv = piv.loc[[k for k in order if k in piv.index]]
    norm = piv.div(piv.max(axis=1).replace(0, 1), axis=0)
    fig, ax = plt.subplots(figsize=(0.55 * len(piv.columns) + 4, 0.28 * len(piv) + 1.2))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", T["seq"])
    im = ax.imshow(norm.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([f"{labels.get(k, k)}  ({int(sub[sub.concept == k].docs.sum()):,})" for k in piv.index], fontsize=7.5)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"{y}*" if y == partial_year else str(y) for y in piv.columns], fontsize=8)
    ax.grid(False)
    for i in range(len(piv)):
        for j in range(len(piv.columns)):
            v = piv.values[i, j]
            if v > 0:
                ax.text(j, i, f"{v:.1f}" if v < 10 else f"{v:.0f}", ha="center", va="center", fontsize=6,
                        color=T["surface"] if norm.values[i, j] > 0.55 else T["ink2"])
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("share / row peak", fontsize=8, color=T["ink2"])
    cb.ax.tick_params(colors=T["muted"])
    ax.set_title(f"Concept share (%) by year - {SOURCE_LABEL.get(src, src)}   cell = % of in-domain docs, colour = relative to the row's peak, * partial year",
                 fontsize=8.5, loc="left")
    return fig


def chart_growth(growth, src, labels, n=14):
    g = growth[(growth["source"] == src) & (growth["docs_recent"] + growth["docs_prior"] >= 20)].sort_values("growth_ratio")
    if g.empty:
        return None
    top = pd.concat([g.head(n // 2), g.tail(n)])
    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(top) + 1))
    cols = [T["critical"] if r < 1 else T["good"] for r in top["growth_ratio"]]
    ax.barh([labels.get(k, k) for k in top["concept"]], np.log2(top["growth_ratio"]), color=cols, height=0.7)
    ax.axvline(0, color=T["axis"], lw=0.8)
    ax.set_xlabel(f"log2(growth ratio)   share {top['recent_years'].iloc[0]} vs {top['prior_years'].iloc[0]}")
    for i, (_, r) in enumerate(top.iterrows()):
        x = np.log2(r["growth_ratio"])
        ax.text(x + (0.06 if x >= 0 else -0.06), i, f"{r['share_prior_pct']:.2f}% -> {r['share_recent_pct']:.2f}%",
                va="center", ha="left" if x >= 0 else "right", fontsize=7, color=T["ink2"])
    ax.set_title(f"Fastest rising / declining concepts - {SOURCE_LABEL.get(src, src)}", loc="left", fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    return fig


def chart_synonyms(syn, concept, label, partial_year):
    sub = syn[(syn["concept"] == concept) & (syn["source"] == "all")] if (syn["source"] == "all").any() else syn[syn["concept"] == concept]
    if sub.empty:
        return None
    agg = sub.groupby(["synonym", "year"], as_index=False)["docs"].sum()
    top_syn = agg.groupby("synonym")["docs"].sum().sort_values(ascending=False).head(7).index.tolist()
    agg["synonym"] = agg["synonym"].where(agg["synonym"].isin(top_syn), "other")
    piv = agg.pivot_table(index="year", columns="synonym", values="docs", aggfunc="sum").fillna(0)
    piv = piv[[c for c in top_syn if c in piv.columns] + (["other"] if "other" in piv.columns else [])]
    share = piv.div(piv.sum(axis=1).replace(0, 1), axis=0) * 100
    colors = [T["other"] if c == "other" else col for c, col in zip(piv.columns, _colors(len(piv.columns)))]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    for ax, data, ttl in ((axes[0], piv, f"{label}: documents by surface form"), (axes[1], share, "mix of surface forms (%)")):
        bottom = np.zeros(len(data))
        for i, c in enumerate(data.columns):
            lbl = c[3:] if c.startswith("re:") else c
            ax.bar(data.index, data[c], bottom=bottom, color=colors[i], label=lbl[:38], width=0.8, edgecolor=T["surface"], linewidth=1)
            bottom += data[c].values
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        _partial_band(ax, partial_year)
    axes[0].set_ylabel("documents")
    axes[0].legend(fontsize=6.5, loc="upper left")
    return fig


def chart_volume(totals, partial_year):
    fig, ax = plt.subplots(figsize=(7.5, 3))
    for i, (src, s) in enumerate(totals.groupby("source")):
        s = s.sort_values("year")
        ax.plot(s["year"], s["in_domain"], marker="o", color=_colors(i + 1)[i], markeredgecolor=T["surface"], markeredgewidth=1.2,
                label=f"{SOURCE_LABEL.get(src, src)} - in-domain corpus (denominator)")
        ax.plot(s["year"], s["source_total"], ls=":", lw=1.4, color=_colors(i + 1)[i], alpha=0.8,
                label=f"{SOURCE_LABEL.get(src, src)} - raw query total")
    _partial_band(ax, partial_year)
    ax.set_ylabel("documents / year")
    ax.set_yscale("log")
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_title("Corpus size per source (log scale)", loc="left", fontsize=10)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return fig


# ----------------------------------------------------------------------------- html
CSS = """
:root{--bg:#f4f6f8;--surface:#fcfcfb;--ink:#14181f;--ink2:#525a66;--muted:#7d8591;--line:#dfe3e8;--line-strong:#b8c0c9;
--accent:#2a78d6;--copper:#a85a24;--copper-soft:#f6e9de;--hl:#fff3c4;--good:#0a7a0a;--critical:#c03434;--show-dark:none;--show-light:block}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0f1214;--surface:#1a1a19;--ink:#eef0f2;--ink2:#b9c0c8;--muted:#8b939d;
--line:#2c3136;--line-strong:#444b53;--accent:#5a9bea;--copper:#d98c5a;--copper-soft:#33241a;--hl:#3d3512;--good:#4cc24c;--critical:#f07070;--show-dark:block;--show-light:none}}
:root[data-theme="dark"]{--bg:#0f1214;--surface:#1a1a19;--ink:#eef0f2;--ink2:#b9c0c8;--muted:#8b939d;--line:#2c3136;--line-strong:#444b53;
--accent:#5a9bea;--copper:#d98c5a;--copper-soft:#33241a;--hl:#3d3512;--good:#4cc24c;--critical:#f07070;--show-dark:block;--show-light:none}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:"IBM Plex Sans","Noto Sans TC","Segoe UI","PingFang TC","Microsoft JhengHei",sans-serif;font-size:15px;line-height:1.6}
.mono{font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace}
main{max-width:1120px;margin:0 auto;padding:0 24px 72px}
header.masthead{padding:40px 0 20px;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--copper);margin:0 0 10px}
h1{font-size:34px;line-height:1.15;margin:0 0 10px;font-weight:600;letter-spacing:-.01em;text-wrap:balance}
.dek{color:var(--ink2);max-width:72ch;margin:0;font-size:16px}
nav.toc{position:sticky;top:0;z-index:5;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;margin-bottom:24px;overflow-x:auto;white-space:nowrap}
nav.toc a{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--ink2);text-decoration:none;margin-right:18px}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--accent);outline:none;text-decoration:underline}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:0 24px;margin:20px 0 8px;border-top:1px solid var(--line-strong)}
.kpi{padding:12px 0 14px;border-bottom:1px solid var(--line)}
.kpi b{display:block;font-size:26px;font-weight:600;letter-spacing:-.01em;font-variant-numeric:tabular-nums;line-height:1.1}
.kpi span{display:block;color:var(--muted);font-size:12.5px;margin-top:4px}
h2{font-size:22px;font-weight:600;margin:48px 0 8px;padding-top:18px;border-top:1px solid var(--line-strong);letter-spacing:-.01em;text-wrap:balance}
h2 .num{font-family:"IBM Plex Mono",ui-monospace,monospace;color:var(--copper);font-weight:500;font-size:14px;margin-right:12px;vertical-align:middle}
h3{font-size:16px;font-weight:600;margin:26px 0 6px}
p{max-width:72ch}
figure{margin:12px 0 26px}
figure img{max-width:100%;height:auto;display:block;border:1px solid var(--line);border-radius:3px;background:var(--surface)}
img.light{display:var(--show-light)}img.dark{display:var(--show-dark)}
figcaption{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11.5px;color:var(--muted);margin-top:8px;max-width:90ch;line-height:1.5}
.wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;margin:10px 0 18px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:6px 10px 6px 0;border-bottom:1px solid var(--line);vertical-align:top}
th{font-family:"IBM Plex Mono",ui-monospace,monospace;font-weight:500;font-size:11.5px;color:var(--muted);letter-spacing:.04em;text-transform:uppercase;border-bottom:1px solid var(--line-strong)}
td.n{text-align:right;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px}
tr.new td{background:var(--hl)}
.up{color:var(--good)}.down{color:var(--critical)}
.note{border-left:3px solid var(--copper);background:var(--copper-soft);padding:10px 16px;margin:16px 0;max-width:80ch;font-size:14px}
.lead{font-size:16.5px;max-width:70ch}
.summary li{margin:4px 0}
details{margin:8px 0;max-width:100%}summary{cursor:pointer;color:var(--accent);font-weight:500}
summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
details ul{padding-left:18px}details li{margin:3px 0;font-size:13.5px}
a{color:var(--accent)}a:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.tag{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:11px;color:var(--muted);margin-left:6px}
/* first-appearance ruler */
.ruler{margin:14px 0 8px;font-size:12.5px}
.ruler .row{display:grid;grid-template-columns:200px 1fr;align-items:center;gap:12px;height:26px}
.ruler .row .lbl{color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ruler .track{position:relative;height:14px;border-bottom:1px dashed var(--line)}
.ruler .bar{position:absolute;top:3px;height:8px;background:var(--accent);opacity:.55;border-radius:2px}
.ruler .peak{position:absolute;top:0;width:14px;height:14px;border-radius:50%;background:var(--copper);border:2px solid var(--bg);margin-left:-7px}
.ruler .axis{display:grid;grid-template-columns:200px 1fr;gap:12px}
.ruler .axis .ticks{position:relative;height:18px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10.5px;color:var(--muted)}
.ruler .axis .ticks span{position:absolute;transform:translateX(-50%)}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.9em;background:var(--surface);border:1px solid var(--line);padding:0 4px;border-radius:3px}
@media (prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
@media (max-width:700px){.ruler .row,.ruler .axis{grid-template-columns:120px 1fr}h1{font-size:28px}}
"""

FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Noto+Sans+TC:wght@400;500;700&display=swap">'


def _tbl(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset(), row_cls: list[str] | None = None) -> str:
    out = ['<div class="wrap"><table><thead><tr>' + "".join(f"<th>{html.escape(h)}</th>" for h in headers) + "</tr></thead><tbody>"]
    for i, r in enumerate(rows):
        cls = f' class="{row_cls[i]}"' if row_cls and row_cls[i] else ""
        out.append(f"<tr{cls}>" + "".join(f'<td{" class=n" if j in numeric else ""}>{c}</td>' for j, c in enumerate(r)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def _ruler(growth_main: pd.DataFrame, labels: dict[str, str], years: list[int], keys: list[str]) -> str:
    y0, y1 = years[0], years[-1]
    span = max(1, y1 - y0)
    rows = []
    for k in keys:
        if k not in growth_main.index:
            continue
        r = growth_main.loc[k]
        if pd.isna(r.first_year_ge3):
            continue
        fy, py = int(r.first_year_ge3), int(r.peak_year) if pd.notna(r.peak_year) else int(r.first_year_ge3)
        left = 100 * (fy - y0) / span
        width = max(1.0, 100 * (y1 - fy) / span)
        peak = 100 * (py - y0) / span
        rows.append(f'<div class="row"><div class="lbl">{html.escape(labels.get(k, k))}</div><div class="track">'
                    f'<div class="bar" style="left:{left:.1f}%;width:{width:.1f}%"></div>'
                    f'<div class="peak" style="left:{peak:.1f}%" title="peak share {py}"></div></div></div>')
    ticks = "".join(f'<span style="left:{100 * (y - y0) / span:.1f}%">{y}</span>' for y in years if (y - y0) % 2 == 0 or y == y1)
    return f'<div class="ruler">{"".join(rows)}<div class="axis"><div></div><div class="ticks">{ticks}</div></div></div>'


def _fmt_pct(v: float) -> str:
    return f"{v:.2f}%"


def build(config: str | Path = DEFAULT_CONFIG) -> Path:
    cfg = load_config(config)
    labels = {k: v.get("label", k) for cat in cfg["categories"].values() for k, v in cat.items()}
    cats = {cat: list(items) for cat, items in cfg["categories"].items()}

    totals = pd.read_csv(TABLES / "totals.csv")
    cy = pd.read_csv(TABLES / "concept_year.csv")
    syn = pd.read_csv(TABLES / "synonym_year.csv")
    growth = pd.read_csv(TABLES / "growth.csv")
    co = pd.read_csv(TABLES / "cooccurrence.csv")
    ex = pd.read_csv(TABLES / "examples.csv") if (TABLES / "examples.csv").stat().st_size > 5 else pd.DataFrame()
    em = pd.read_csv(TABLES / "emerging.csv") if (TABLES / "emerging.csv").stat().st_size > 5 else pd.DataFrame()
    summary = json.loads((TABLES / "summary.json").read_text(encoding="utf-8"))
    partial_year = summary["partial_year"]
    have = set(cy["source"])
    sources = [s for s in ["arxiv", "ieee", "google_patents"] if s in have]
    main_src = "all" if "all" in have else sources[0]
    panels = ([main_src] if main_src == "all" else []) + sources
    years = summary["years"]
    s2 = summary.get("mode") == "s2"
    g_main = growth[growth.source == main_src].set_index("concept")
    win = g_main.iloc[0]

    P: list[str] = []
    add = P.append

    # ---------------- masthead
    add('<header class="masthead">')
    add(f'<p class="eyebrow">Semiconductor literature · {years[0]}–{years[-1]} · {len(labels)} concepts · {"、".join(SOURCE_LABEL[s] for s in sources)}</p>')
    add("<h1>半導體技術詞趨勢圖譜</h1>")
    route = "Semantic Scholar 學術圖譜（含摘要；IEEE 以 DOI 前綴 10.1109 或出版者辨識、arXiv 以 arXiv id 辨識）" if s2 else "OpenAlex（含摘要）"
    add(f'<p class="dek">GAA、CoWoS、EUV、chiplet、HBM、hybrid bonding……{len(labels)} 個技術概念在 {years[0]}–{years[-1]} 年論文中出現比例的變化，'
        f'以同義詞字典把各種英文寫法收斂到同一概念後計算。資料經 {route}；{partial_year} 年為部分年度。產生於 {summary["run_at"][:16].replace("T", " ")}。</p>')
    add("</header>")
    add('<nav class="toc"><a href="#s1">01 語料</a><a href="#s2">02 焦點技術</a><a href="#s3">03 各領域</a><a href="#s4">04 熱圖</a>'
        '<a href="#s5">05 成長排行</a><a href="#s6">06 同義詞遷移</a><a href="#s7">07 共現</a><a href="#s8">08 新興詞</a><a href="#s9">09 代表文件</a><a href="#s10">10 方法</a></nav>')

    # ---------------- KPIs
    add('<div class="kpis">')
    add(f'<div class="kpi"><b>{summary["corpus_in_domain_est"]:,.0f}</b><span>{"估計的半導體相關文獻總量（所有出版者）" if s2 else "通過相關性閘門的文件（分母）"}</span></div>')
    for s in sources:
        t = totals[totals.source == s]
        add(f'<div class="kpi"><b>{int(t.in_domain.sum()):,}</b><span>{SOURCE_LABEL[s]} 相關文獻{"（估計）" if s2 else ""}</span></div>')
    if s2:
        n_num = int(cy[cy.source == "all"].docs_raw.sum())
        add(f'<div class="kpi"><b>{n_num:,}</b><span>概念查詢命中並經本地比對確認的文件</span></div>')
    add(f'<div class="kpi"><b>{summary["docs_analysed"]:,}</b><span>{"隨機抽樣並逐篇分析的文件" if s2 else "抓取並分析的文件"}</span></div>')
    add("</div>")

    # ---------------- headline summary (answer first)
    ok = g_main[(g_main.docs_recent + g_main.docs_prior) >= 20]
    risers = ok.sort_values("growth_ratio", ascending=False).head(5)
    fallers = ok.sort_values("growth_ratio").head(4)
    add('<div class="summary"><p class="lead"><b>一句話結論。</b>'
        f'以 {win.recent_years} 對比 {win.prior_years} 的份額，成長最快的是 '
        + "、".join(f"{labels[k]}（{r.share_prior_pct:.2f}%→{r.share_recent_pct:.2f}%，{r.growth_ratio:.1f}×）" for k, r in risers.iterrows())
        + "；衰退最明顯的是 "
        + "、".join(f"{labels[k]}（{r.share_prior_pct:.2f}%→{r.share_recent_pct:.2f}%）" for k, r in fallers.iterrows())
        + "。</p></div>")
    add('<div class="note">份額 (share) = 該年提到此概念的文件數 ÷ 該年半導體相關文件數。分母逐年變動，份額比原始篇數更適合比較趨勢。'
        + ("分母由每年一萬筆隨機樣本推估，分子來自逐概念完整查詢並經本地同義詞正規式再確認。" if s2 else "")
        + "完整數字在 <code>output/tables/concept_year.csv</code>。</div>")

    # ---------------- 1 corpus
    add('<h2 id="s1"><span class="num">01</span>語料規模與摘要可得率</h2>')
    add(f'<figure>{_img(_render(chart_volume, "00_volume", totals, partial_year), "corpus volume")}'
        '<figcaption>實線：通過本地相關性閘門的文件數（份額的分母）；點線：搜尋引擎對領域查詢回傳的原始總量。差距來自搜尋引擎的詞幹化／模糊比對，本地閘門以大小寫敏感的縮寫規則過濾。</figcaption></figure>')
    if "abstract_rate" in totals.columns:
        ar = totals.pivot_table(index="source", columns="year", values="abstract_rate")
        rows = [[SOURCE_LABEL.get(s, s)] + [f"{100 * v:.0f}%" if pd.notna(v) else "–" for v in ar.loc[s]] for s in ar.index if s in panels]
        add("<h3>摘要可得率</h3><p>索引服務對舊年份的摘要覆蓋率較低；沒有摘要的文件只能靠標題比對，會讓早期年份的份額略被低估。解讀 2014–2017 的絕對水準時請把這點放在心上，年與年之間的相對趨勢仍可比較。</p>")
        add(_tbl(["來源"] + [str(y) for y in ar.columns], rows, numeric=set(range(1, len(ar.columns) + 1))))

    # ---------------- 2 headline
    add('<h2 id="s2"><span class="num">02</span>焦點技術：GAA、CoWoS、EUV 與先進封裝</h2>')
    add(f'<figure>{_img(_render(chart_lines, "01_headline", cy, HEADLINE, panels, "Headline concepts - share of in-domain documents", partial_year, labels), "headline")}'
        '<figcaption>每條線為該概念在各年份文件中的出現比例（%）。虛線段為部分年度；圖例依總量排序。</figcaption></figure>')
    rows, cls = [], []
    for k in HEADLINE:
        if k in g_main.index:
            r = g_main.loc[k]
            arrow = f'<span class="{"up" if r.growth_ratio >= 1 else "down"}">{r.growth_ratio:.2f}×</span>'
            rows.append([html.escape(labels[k]), f"{int(r.docs_total):,}", str(int(r.first_year_ge3)) if pd.notna(r.first_year_ge3) else "–",
                         _fmt_pct(r.share_prior_pct), _fmt_pct(r.share_recent_pct), arrow, f"{r.slope_pp_per_year:+.3f}",
                         str(int(r.peak_year)) if pd.notna(r.peak_year) else "–"])
    add(_tbl(["概念", "文件數", "首次 ≥3 篇", f"份額 {win.prior_years}", f"份額 {win.recent_years}", "成長倍數", "斜率 pp/yr", "高峰年"], rows, numeric={1, 2, 3, 4, 5, 6, 7}))
    add("<h3>何時進入文獻、何時到達高峰</h3><p>藍色橫條從該概念首次達到 3 篇的年份延伸至今；銅色圓點是份額最高的年份。</p>")
    add(_ruler(g_main, labels, years, HEADLINE + ["FinFET", "3DIC", "HighNA", "BSPDN", "CFET", "RuMo", "GlassSubstrate", "UCIe"]))

    # ---------------- 3 categories
    add('<h2 id="s3"><span class="num">03</span>各技術領域趨勢</h2>')
    for i, (cat, keys) in enumerate(cats.items()):
        keys = [k for k in keys if cy[(cy.concept == k) & (cy.source == main_src)].docs.sum() > 0]
        if not keys:
            continue
        add(f"<h3>{html.escape(cat)}</h3>")
        add(f'<figure>{_img(_render(chart_lines, f"1{i}_{cat.split()[0].lower()}", cy, keys, panels, cat, partial_year, labels), cat)}</figure>')

    # ---------------- 4 heatmap
    add('<h2 id="s4"><span class="num">04</span>全景熱圖</h2>')
    add(f'<figure>{_img(_render(chart_heatmap, "20_heatmap", cy, main_src, labels, partial_year), "heatmap")}'
        '<figcaption>每格數字為該年份額（%），顏色為該列相對於自身高峰的比例，因此小眾但快速成長的主題（CoWoS、hybrid bonding）也看得出高峰落在哪一年。括號為總文件數。</figcaption></figure>')

    # ---------------- 5 growth
    add('<h2 id="s5"><span class="num">05</span>成長最快與衰退最快</h2>')
    for s in panels:
        pair = _render(chart_growth, f"30_growth_{s}", growth, s, labels)
        if pair:
            add(f"<figure>{_img(pair, 'growth ' + s)}</figure>")
    add('<p class="mono" style="font-size:12px;color:var(--muted)">growth ratio = (recent share + ε) / (prior share + ε)，ε 為先前窗口一篇文件的份額；僅列兩窗口合計 ≥ 20 篇的概念。</p>')

    # ---------------- 6 synonyms
    add('<h2 id="s6"><span class="num">06</span>同義詞與表面形式的遷移</h2>')
    add("<p>同一技術在不同年代常用不同名稱。把每個概念的命中依「哪一個同義詞命中」拆開，可看出用語如何遷移，例如 nanosheet 逐漸取代 gate-all-around 成為主流稱呼。</p>")
    for k in SYNONYM_SHOWCASE:
        if k in labels:
            pair = _render(chart_synonyms, f"40_syn_{k}", syn, k, labels[k], partial_year)
            if pair:
                add(f"<figure>{_img(pair, k)}</figure>")

    # ---------------- 7 co-occurrence
    add('<h2 id="s7"><span class="num">07</span>概念共現</h2>')
    add("<p>同一篇文件同時提到兩個概念的次數；lift &gt; 1 表示比隨機更常一起出現。可觀察技術組合，例如 chiplet × hybrid bonding、GAA × node。</p>")
    for s in panels:
        c = co[co.source == s].head(15)
        if c.empty:
            continue
        add(f"<h3>{SOURCE_LABEL[s]}</h3>")
        add(_tbl(["概念 A", "概念 B", "共現", "A", "B", "lift", "Jaccard"],
                 [[html.escape(labels.get(r.concept_a, r.concept_a)), html.escape(labels.get(r.concept_b, r.concept_b)), f"{int(r.docs_both):,}",
                   f"{int(r.docs_a):,}", f"{int(r.docs_b):,}", f"{r.lift:.1f}", f"{r.jaccard:.3f}"] for _, r in c.iterrows()], numeric={2, 3, 4, 5, 6}))

    # ---------------- 8 emerging
    add('<h2 id="s8"><span class="num">08</span>新興詞彙探勘：同義詞字典的盲點檢查</h2>')
    add("<p>不靠字典，直接統計 1–3 gram 在近期窗口相對於先前窗口的文件份額成長。<b>黃底</b>為目前字典尚未涵蓋的詞，是候選的新同義詞或新主題；加進 <code>config/keywords.json</code> 後重跑 <code>analyze</code> 即可納入。</p>")
    if not em.empty:
        for s in panels:
            e = em[em.source == s].head(40)
            if e.empty:
                continue
            add(f"<h3>{SOURCE_LABEL[s]}<span class='tag'>{e.recent_years.iloc[0]} vs {e.prior_years.iloc[0]}</span></h3>")
            rows = [[html.escape(r.term), f"{int(r.recent_df):,}", f"{int(r.prior_df):,}", f"{r.prior_share_pct:.2f}% → {r.recent_share_pct:.2f}%",
                     f"{r.growth_ratio:.1f}×", html.escape(str(r.covered_by)) if isinstance(r.covered_by, str) and r.covered_by else "–"] for _, r in e.iterrows()]
            cls = ["new" if not (isinstance(r.covered_by, str) and r.covered_by) else "" for _, r in e.iterrows()]
            add(_tbl(["詞", "近期文件", "先前文件", "份額變化", "倍數", "已涵蓋於概念"], rows, numeric={1, 2, 4}, row_cls=cls))

    # ---------------- 9 examples
    if not ex.empty:
        add('<h2 id="s9"><span class="num">09</span>代表文件（近四年引用最多）</h2>')
        for s in panels:
            e = ex[ex.source == s]
            if e.empty:
                continue
            add(f"<details><summary>{SOURCE_LABEL[s]}：展開各概念的代表文件</summary>")
            for k, grp in e.groupby("concept"):
                items = "".join(f'<li>[{int(r.year)}] <a href="{html.escape(str(r.url))}" target="_blank" rel="noopener">{html.escape(str(r.title))}</a>'
                                f'<span class="tag">{int(r.cited)} cites</span></li>' for _, r in grp.iterrows())
                add(f"<p><b>{html.escape(labels.get(k, k))}</b></p><ul>{items}</ul>")
            add("</details>")

    # ---------------- 10 method
    add('<h2 id="s10"><span class="num">10</span>方法與限制</h2>')
    add("<ul>"
        "<li><b>語料定義</b>：以 <code>domain_terms</code>（約 60 個領域詞的 OR）向資料源查詢，再於本地以同一組詞、同一套比對規則做相關性閘門，讓不同來源共用一致的分母。</li>"
        "<li><b>同義詞處理</b>：每個概念維護同義詞清單；含小寫字母的片語不分大小寫、空白與連字號可互換或省略（gate-all-around／gate all around／nanosheet／nano-sheet 皆命中）；"
        "純大寫縮寫（GAA、PIM、TSV）採大小寫敏感整詞比對避免誤判；<code>exclude</code> 排除歧義（PCM = pulse-code modulation）。</li>"
        "<li><b>文字正規化</b>：Unicode 連字號、上下標（MoS₂→MoS2）、LaTeX（\\mathrm{Ga_2O_3}→Ga2O3）、HTML 實體先轉成 ASCII 再比對。</li>"
        "<li><b>計數單位</b>：文件層級，一篇文件不論提到幾次都算一次。</li>"
        + ("<li><b>抽樣與加權</b>：Semantic Scholar 對領域查詢每年約 10 萬筆，每年取 1 萬筆（依 paper id 排序，等同隨機）估計各來源的分母；"
           "分子以逐概念查詢（同義詞 OR 列表）取得，超過 3,000 筆的概念年以 total/fetched 加權回推。所有分子文件都經本地正規式再確認並依 DOI／arXiv id 歸類。</li>" if s2 else "")
        + "<li><b>限制</b>：arXiv 與 IEEE 的涵蓋率受索引服務影響；arXiv 預印本後來刊登於 IEEE 時同時計入兩者；舊年份摘要可得率低。"
        "Google Patents 需 BigQuery 憑證或網站 CSV 匯出（見 README），本次未包含。最後一年為部分年度。</li></ul>")

    body = "".join(P)
    head = f"<title>半導體技術詞趨勢圖譜</title>{FONTS}<style>{CSS}</style>"
    (OUTPUT_DIR / "report_artifact.html").write_text(f"{head}<main>{body}</main>", encoding="utf-8")
    doc = ("<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
           f"{head}</head><body><main>{body}</main></body></html>")
    out = OUTPUT_DIR / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out
