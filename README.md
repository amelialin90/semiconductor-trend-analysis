# semitrend — 半導體專利／論文技術關鍵字趨勢分析

追蹤 GAA、CoWoS、EUV、chiplet、HBM、hybrid bonding……等 47 個技術概念在
arXiv、IEEE Xplore（以及可選的 Google Patents）文獻中，逐年出現比例的變化。
重點在於**英文技術詞與同義詞的處理**：每個概念維護一份同義詞清單，並以一套
可預期的比對規則（大小寫、連字號、複數、LaTeX／Unicode 正規化、排除規則）
把各種寫法收斂到同一個概念。

`output/tables/*.csv`、`output/report.html` 這些**分析完的結果**
都有放進來，可以直接打開看，不需要重新抓資料就能瀏覽。

```
config/keywords.json      概念 × 同義詞字典（可自由編輯；改完只需重跑 analyze）
semitrend/
  normalize.py            文字正規化（Unicode 連字號、MoS₂→MoS2、LaTeX、HTML）
  matcher.py              同義詞 → 正規式；每概念一支合併正規式；領域相關性閘門
  discover.py             新興 n-gram 探勘（字典盲點檢查）
  analyze.py              建表：份額、成長、同義詞組成、共現、代表文件、新興詞
  report.py               產生 output/report.html（自含式，圖表內嵌）
  sources/
    semanticscholar.py    預設路徑：Semantic Scholar 批次搜尋（免金鑰，含摘要）
    openalex.py           OpenAlex（免費額度很小，可續跑）
    arxiv_api.py          arXiv 官方 API（--via direct；2026 年起常回 429）
    ieee_xplore.py        IEEE Xplore Metadata API（--via direct；需 IEEE_API_KEY）
    google_patents.py     BigQuery 公開資料集（需 GCP 憑證）或 patents.google.com CSV 匯出
data/raw/                 抓下來的原始 JSONL（可重複使用，不需重抓）
data/processed/           每篇文件的比對結果
output/tables/*.csv       所有分析表
output/charts/*.png       圖表
output/report.html        報告
```

## 安裝與執行

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# 1. 抓資料（預設走 Semantic Scholar，可中斷後續跑；三個分片並行約 20-30 分鐘）
.venv\Scripts\python -m semitrend fetch --source ieee arxiv --years 2014-2026 --shard 0/3
.venv\Scripts\python -m semitrend fetch --source ieee arxiv --years 2014-2026 --shard 1/3
.venv\Scripts\python -m semitrend fetch --source ieee arxiv --years 2014-2026 --shard 2/3

# 2. 比對 + 建表（逐年串流；記憶體 < 2 GB 時用 --workers 2，約 40 分鐘）
#    年份切檔與 n-gram 快取放在 data/processed/，改字典後重跑只需重做正規式比對
.venv\Scripts\python -m semitrend analyze --workers 2

# 3. 產生報告
.venv\Scripts\python -m semitrend report

# 測試某段文字會命中哪些概念
.venv\Scripts\python -m semitrend test-keywords "gate-all-around nanosheet FETs at the 2 nm node using EUV"
```

## 資料來源與取得方式

| 來源 | 路徑 | 需要 | 備註 |
|---|---|---|---|
| arXiv | `--via s2`（預設） | 無 | Semantic Scholar 記錄帶 arXiv id 者 |
| IEEE Xplore | `--via s2`（預設） | 無 | DOI 前綴 `10.1109` 或出版者含 IEEE 者，含摘要 |
| arXiv | `--via direct` | 無 | 官方 API；目前多數網路環境回 429，僅作備援 |
| IEEE Xplore | `--via direct` | `IEEE_API_KEY` | 免費金鑰 200 次/日 × 200 筆 |
| arXiv / IEEE | `--via openalex` | 無 | 每日免費額度僅約 250 次請求 |
| Google Patents | `--source google_patents` | GCP 憑證 + `pip install google-cloud-bigquery` | `patents-public-data.patents.publications`，CPC H01L/H10*/G03F/H05K/G11C |
| Google Patents | `--source google_patents` | 把網站匯出的 CSV 放到 `data/raw/google_patents/` | 只有標題與日期，命中率較低 |

### Semantic Scholar 的「抽樣分母、精準分子」設計

領域查詢（`domain_terms`，約 60 個詞的 OR）在 S2 每年約 10 萬筆，全量下載不切實際，因此：

1. **分母**：每年取前 10 頁 × 1000 筆（S2 依 paper id 排序，id 是雜湊值，等同隨機抽樣），在本地做相關性閘門並依 DOI／arXiv id 歸類來源，再乘上總量回推各來源的語料規模。
2. **分子**：對每個概念用它的同義詞組 OR 查詢，逐年完整抓取（上限 3 頁；超過者依 total/fetched 加權）。每篇文件都以本地正規式再確認，S2 的模糊搜尋只影響召回率，不影響精確度。
3. 抽樣池另外用來做**新興詞探勘**，找出字典還沒收的新寫法。

## 同義詞比對規則（`config/keywords.json`）

| 寫法 | 規則 | 範例 |
|---|---|---|
| 含小寫字母的片語 | 不分大小寫；空白／連字號可互換或省略；允許複數 | `gate all around` 命中 gate-all-around、GateAllAround；`nano sheet` 命中 nanosheet |
| 全大寫縮寫 | 大小寫敏感、整詞比對 | `GAA` 不會命中 "gaa protein"；`PIM` 不會命中 "pim kinase" |
| `re:` 前綴 | 原始正規式（開頭 `(?i)` 表示不分大小寫） | `re:\bHBM[2-4]?E?\b` |
| `exclude` | 命中此正規式的文件不計入該概念 | PCM 排除 pulse-code modulation |
| `require` | 文件必須**同時**命中此正規式（上下文閘門） | GAA 的 nanosheet 只在 transistor/FET 語境計入，避免 MoS₂ 奈米片；GaN 只在 power/RF 語境 |
| `s2_extra` | 只加入 S2 搜尋查詢的純文字詞（正規式無法送給搜尋引擎） | HBM → HBM2/HBM3/HBM3E |

比對前的正規化：Unicode 連字號→`-`、上下標數字→ASCII（MoS₂→MoS2）、`\mathrm{Ga_2O_3}`→Ga2O3、HTML 實體、多重空白。
計數單位是**文件**（一篇不論提到幾次都算一次），避免長摘要灌水。

## 輸出表格

| 檔案 | 內容 |
|---|---|
| `totals.csv` | 每來源每年：分析文件數、in-domain 語料（分母）、查詢總量 |
| `concept_year.csv` | 每概念每來源每年：文件數（加權）、原始命中數、份額 %、估計絕對篇數 |
| `synonym_year.csv` | 每個概念的命中是由哪個同義詞貢獻（看用語遷移，如 nanosheet 取代 GAA） |
| `growth.csv` | 近期窗口 vs 先前窗口的份額倍數、變化（pp）、線性斜率、首次≥3篇年份、高峰年 |
| `cooccurrence.csv` | 概念共現、lift、Jaccard |
| `examples.csv` | 各概念近四年被引用最多的代表文件 |
| `emerging.csv` | 份額暴增的 1–3 gram；`covered_by` 空白者為字典尚未涵蓋的候選新詞 |

## 擴充字典的建議流程

1. 看 `output/report.html` 第 8 節或 `emerging.csv` 中黃底（未涵蓋）的詞。
2. 把它加進對應概念的 `synonyms`（或新增概念）。
3. 重跑 `analyze` + `report`（不必重新抓取；若要讓 S2 查詢也涵蓋新詞，再跑一次 `fetch`，會只補抓缺的概念年）。
