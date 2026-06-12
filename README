# ohlc2intraday

📘 [English](#english) ・ [日本語](#日本語)

---

## English

Generate statistically plausible **intraday price paths from daily OHLC data**,
based on Riedel (2021), and measure how well existing volatility findings can
be reproduced from free, low-frequency data.

### Motivation

Most empirical finance research relies on intraday data (TAQ, Bloomberg, etc.)
that costs hundreds of thousands of yen per year or requires an institutional
license. Individual researchers and students can typically only access **free
daily OHLC data** (open, high, low, close) from sources like Yahoo Finance.

This package asks: *given only daily OHLC, how far can we statistically
reconstruct intraday dynamics, and which published findings remain reproducible
on the reconstructed data?* Perfect reconstruction is information-theoretically
impossible (4 numbers cannot recover ~390), so the goal is to **map where the
reconstruction works and where it breaks**.

### Method

The core is Riedel (2021), which gives closed-form conditional moments of a
Brownian motion given its high, low, and close. We implement this incrementally:

- **Section 2** — Brownian bridge conditioned on close only (baseline).
- **Section 3** — conditioned on close **and high** (Theorem 3.4), with an
  exact path sampler via Bessel(3)-bridge (Williams) decomposition.
- **Section 4** — full OHLC conditioning (planned).

Synthetic paths are validated against **FirstRateData SPX 1-minute data**
(2008–2026) as ground truth.

### Status (Phase 0)

- [x] Project structure
- [x] Section 2: Brownian bridge (close-only)
- [x] Section 3: close + high (Theorem 3.4 + Bessel(3) sampler)
- [x] Long-term validation (250 days, FirstRateData ground truth)
- [ ] Section 4: full OHLC conditioning
- [ ] Replication matrix on existing finance papers (HAR, SHAR, ...)
- [ ] PyPI release

### Key Results

Section 3 vs Section 2 over 250 trading days of SPX 1-min data.
Ratio = synthetic return std / true return std (1.0 = perfect).

| Method | median ratio | IQR |
|--------|--------------|-----|
| Section 2 (close only) | 0.787 | [0.320, 1.194] |
| Section 3 (close + high) | 0.879 | [0.629, 1.114] |

- Section 3 is closer to the true volatility on **69.6%** of days.
- Section 3 mainly helps by **reducing variability** (much tighter IQR),
  not just shifting the median.
- Section 3 low-violation rate: median 22.5%, mean 32.4%,
  which motivates Section 4 (adding the low constraint).

### Repository layout
ohlc2intraday/
├── ohlc2intraday/
│   ├── brownian_bridge.py          # Section 2
│   ├── conditional_close_high.py   # Section 3 moments (Theorem 3.4)
│   └── path_sampler.py             # Section 3 exact path sampler
├── notebooks/
│   ├── 01_section2_basic.py
│   ├── 02_section2_evaluation.py
│   ├── 03_section2_distribution.py
│   ├── 04_section3_evaluation.py   # 5-day pilot
│   └── 05_long_term_evaluation.py  # 250-day validation
├── data/                           # FirstRateData (not tracked)
└── results/

### Reference

Riedel, K. (2021). *The Value of the High, Low and Close in the Estimation of
Brownian Motion.* Statistical Inference for Stochastic Processes.
[arXiv:1911.05280](https://arxiv.org/abs/1911.05280)

### License

MIT

---

## 日本語

Riedel (2021) に基づき、**日次OHLCデータから統計的に妥当な日中価格パス**を生成し、
無料・低頻度データから既存のボラティリティ研究の発見をどこまで再現できるかを検証する
パッケージ。

### 動機

実証ファイナンス研究の多くは、年間数十万円〜数百万円の費用や機関契約が必要な
高頻度データ (TAQ、Bloomberg等) に依存している。個人研究者や学生が使えるのは、
通常 Yahoo Finance 等から得られる**無料の日次OHLCデータ** (始値・高値・安値・終値)
だけである。

本パッケージが問うのは、*日次OHLCだけから、日中の動きをどこまで統計的に再構成でき、
再構成したデータ上でどの既存研究の発見が再現できるか* という点である。完全な再構成は
情報理論的に不可能 (4つの数字から約390個は復元できない) なので、目標は
**「再構成が機能する場面と破綻する場面の地図を作ること」**にある。

### 手法

中核は Riedel (2021) で、高値・安値・終値を条件としたブラウン運動の条件付きモーメントを
閉形式で与える。これを段階的に実装する:

- **Section 2** — 終値のみを条件とするブラウン橋 (ベースライン)。
- **Section 3** — 終値**と高値**を条件 (Theorem 3.4)。Bessel(3)橋 (Williams) 分解に
  よる厳密なパスサンプラーを実装。
- **Section 4** — OHLC全条件 (予定)。

生成したパスは、真値として **FirstRateData の SPX 1分足データ** (2008–2026) で検証する。

### 進捗 (Phase 0)

- [x] プロジェクト構造
- [x] Section 2: ブラウン橋 (終値のみ)
- [x] Section 3: 終値+高値 (Theorem 3.4 + Bessel(3)サンプラー)
- [x] 長期検証 (250日、FirstRateData真値)
- [ ] Section 4: OHLC全条件
- [ ] 既存研究の再現性マトリクス (HAR, SHAR, ...)
- [ ] PyPI 公開

### 主要な結果

SPX 1分足 250営業日での Section 3 vs Section 2 の比較。
比率 = 合成リターンの標準偏差 / 真値リターンの標準偏差 (1.0 が完全一致)。

| 手法 | 比率の中央値 | 四分位範囲 (IQR) |
|------|------------|-----------------|
| Section 2 (終値のみ) | 0.787 | [0.320, 1.194] |
| Section 3 (終値+高値) | 0.879 | [0.629, 1.114] |

- Section 3 は **69.6%** の日で真値ボラティリティに近い。
- Section 3 の改善は主に**バラつきの縮小** (IQRが大幅に狭い) によるもので、
  中央値の改善だけではない。
- Section 3 の安値違反率: 中央値 22.5%、平均 32.4%。
  これが Section 4 (安値制約の追加) の動機となる。

### リポジトリ構成
ohlc2intraday/
├── ohlc2intraday/
│   ├── brownian_bridge.py          # Section 2
│   ├── conditional_close_high.py   # Section 3 モーメント (Theorem 3.4)
│   └── path_sampler.py             # Section 3 厳密パスサンプラー
├── notebooks/
│   ├── 01_section2_basic.py
│   ├── 02_section2_evaluation.py
│   ├── 03_section2_distribution.py
│   ├── 04_section3_evaluation.py   # 5日パイロット
│   └── 05_long_term_evaluation.py  # 250日検証
├── data/                           # FirstRateData (Git管理外)
└── results/
### 参考文献

Riedel, K. (2021). *The Value of the High, Low and Close in the Estimation of
Brownian Motion.* Statistical Inference for Stochastic Processes.
[arXiv:1911.05280](https://arxiv.org/abs/1911.05280)

### ライセンス

MIT