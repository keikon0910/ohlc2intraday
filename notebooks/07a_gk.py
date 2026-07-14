# notebook 07: Phase 1 第一歩
# 「経路でしか作れない量」の再現性を S4 合成経路で検証する
# 検証対象(情報要求度レベル順):
#   レベル0(土台):    RV (実現ボラ)
#   レベル1(非自明):  MaxDD (最大ドローダウン)
#   レベル1':         RS+/RS- の大きさ
#   レベル2(境界候補): RS-/RS+ 比 (非対称性)
#   レベル3(比較用):  実現歪度 (対称ブリッジで再現不可の予測)
# 扱わないもの: レンジで作れる量・板情報が要る量・ジャンプ系
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd

from ohlc2intraday.brownian_bridge import brownian_bridge_path
from ohlc2intraday.path_sampler_ohlc import make_sampler_ohlc

DATA_PATH = "../data/SPX_full_1min.txt"
N_DAYS    = 250
N_PATHS   = 100
SEED      = 42
LOW_TOL   = 1e-7


# ────────────────────────────────────────────────
# 量の計算関数
# ────────────────────────────────────────────────

def calc_rv(path):
    """RV: 1分リターン2乗和"""
    return float(np.sum(np.diff(path)**2))

def calc_maxdd(path):
    """最大ドローダウン: 天井からの最大下落幅 (log空間)"""
    runmax = np.maximum.accumulate(path)
    return float(np.max(runmax - path))

def calc_semivar(path):
    """RS+, RS-: 上昇/下落リターンの2乗和"""
    r = np.diff(path)
    rsp = float(np.sum(r[r > 0]**2))
    rsm = float(np.sum(r[r < 0]**2))
    return rsp, rsm

def calc_skew(path):
    """実現歪度: 3次実現モーメントを分散^(3/2) で正規化"""
    r = np.diff(path)
    m2 = np.sum(r**2)
    if m2 < 1e-20:
        return 0.0
    return float(np.sum(r**3) / (m2**1.5))

def calc_all(path):
    """1本の経路から全量を計算して dict で返す"""
    rsp, rsm = calc_semivar(path)
    return {
        "rv":    calc_rv(path),
        "maxdd": calc_maxdd(path),
        "rsp":   rsp,
        "rsm":   rsm,
        "skew":  calc_skew(path),
    }

def mean_stats(paths):
    """複数経路の各量の平均を dict で返す"""
    all_stats = [calc_all(p) for p in paths]
    keys = all_stats[0].keys()
    return {k: float(np.mean([s[k] for s in all_stats])) for k in keys}


# ────────────────────────────────────────────────
# データ読み込み
# ────────────────────────────────────────────────

def load_firstrate(path):
    df = pd.read_csv(
        path, header=None,
        names=["datetime","open","high","low","close"],
        usecols=[0,1,2,3,4], parse_dates=["datetime"],
    )
    return df.set_index("datetime").sort_index()


# ────────────────────────────────────────────────
# 1日の評価
# ────────────────────────────────────────────────

def eval_day(day_df, n_paths, day_seed):
    prices = day_df["close"].values.astype(float)
    if len(prices) < 100:
        return None

    O = float(day_df["open"].iloc[0])
    H = float(day_df["high"].max())
    L = float(day_df["low"].min())
    C = float(day_df["close"].iloc[-1])
    if not (H >= max(O, C) and L <= min(O, C) and O > 0 and L > 0):
        return None

    lo = np.log(O)
    lH, lL, lC = np.log(H)-lo, np.log(L)-lo, np.log(C)-lo
    true_log = np.log(prices) - lo
    true_ret = np.diff(true_log)
    n = len(true_log) - 1
    if true_ret.std() == 0:
        return None

    # 独立した子 RNG (手法間で順序依存しない)
    ss = np.random.SeedSequence(day_seed)
    rng_s2, rng_s4_prep, rng_s4_path = [
        np.random.default_rng(s) for s in ss.spawn(3)
    ]

    # ── 本物の量を計算 ──
    true_stats = calc_all(true_log)

    # ── S2 ベースライン (終値のみ) ──
    s2_sig2 = max(lC**2, 1e-16)   # 終値の対数の2乗(GK版のS2)
    s2_paths  = np.array([
        brownian_bridge_path(0.0, lC, n_steps=n, sigma2=s2_sig2, rng=rng_s2)
        for _ in range(n_paths)
    ])
    s2_stats = mean_stats(s2_paths)

    # ── S4 (OHLC全条件) ──
    gk      = 0.5*(lH-lL)**2 - (2*np.log(2)-1)*lC**2
    s4_sig2 = max(gk, (2*lH-lC)**2/3.0, 1e-16)
    try:
        sampler, _ = make_sampler_ohlc(
            0.0, lH, lL, lC, n_steps=n, sigma2=s4_sig2,
            prepass_kwargs=dict(
                seed=int(rng_s4_prep.integers(1<<31)),
                target_pool=300, batch=30_000, total=5_000_000,
                tol_sigma=0.15,   # ★σ比例、Step4 で収束確認して確定
            ),
        )
        s4_paths = np.array([sampler(rng_s4_path) for _ in range(n_paths)])
    except RuntimeError:
        return None

    s4_low_viol = (s4_paths.min(1) < lL - LOW_TOL).mean() * 100
    s4_stats    = mean_stats(s4_paths)

    # ── 非対称比 RS-/RS+ ──
    def asym(st):
        rsp, rsm = st["rsp"], st["rsm"]
        return rsm / rsp if rsp > 1e-20 else np.nan

    row = {"date": None}
    for key in ["rv", "maxdd", "rsp", "rsm", "skew"]:
        row[f"true_{key}"] = true_stats[key]
        row[f"s2_{key}"]   = s2_stats[key]
        row[f"s4_{key}"]   = s4_stats[key]

    row["true_asym"]   = asym(true_stats)
    row["s2_asym"]     = asym(s2_stats)
    row["s4_asym"]     = asym(s4_stats)
    row["s4_low_viol"] = s4_low_viol

    return row


# ────────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────────

def main():
    df = load_firstrate(DATA_PATH)
    print(f"loaded: {len(df):,} rows  {df.index.min()} -> {df.index.max()}")

    dates   = sorted(set(df.index.date))[-N_DAYS:]
    rows    = []
    skipped = 0

    for i, d in enumerate(dates):
        day_df   = df[df.index.date == d]
        day_seed = SEED * 1_000_000 + i
        row      = eval_day(day_df, N_PATHS, day_seed)
        if row is not None:
            row["date"] = d
            rows.append(row)
        else:
            skipped += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(dates)} 日処理 (スキップ {skipped}日)")

    res = pd.DataFrame(rows)
    print(f"\nevaluated: {len(res)} days, skipped: {skipped}\n")

    def spearman(a, b):
        from scipy.stats import spearmanr
        r, _ = spearmanr(a, b)
        return r

    print("=" * 60)
    print("レベル別 再現性サマリー")
    print("=" * 60)

    metrics = [
        ("レベル0  RV",           "rv",    "比率"),
        ("レベル1  MaxDD",         "maxdd", "比率"),
        ("レベル1' RS+ (大きさ)",  "rsp",   "比率"),
        ("レベル1' RS- (大きさ)",  "rsm",   "比率"),
        ("レベル2  RS-/RS+ 比",    "asym",  "値"),
        ("レベル3  実現歪度",       "skew",  "値"),
    ]

    for label, key, mode in metrics:
        tv  = res[f"true_{key}"].dropna()
        s4v = res[f"s4_{key}"].dropna()
        idx = tv.index.intersection(s4v.index)
        tv, s4v = tv[idx], s4v[idx]

        if mode == "比率":
            ratio = np.median(s4v / tv)
            corr  = spearman(tv, s4v)
            print(f"\n{label}")
            print(f"  本物 中央値={np.median(tv):.5f}")
            print(f"  S4   中央値={np.median(s4v):.5f}")
            print(f"  S4/本物 比率={ratio:.3f}  スピアマン相関={corr:.3f}")
        else:
            s2v = res[f"s2_{key}"].dropna()
            print(f"\n{label}")
            print(f"  本物 中央値={np.median(tv):.4f}  mean={np.mean(tv):.4f}")
            print(f"  S2   中央値={np.median(s2v):.4f}")
            print(f"  S4   中央値={np.median(s4v):.4f}  mean={np.mean(s4v):.4f}")
            if key == "asym":
                print(f"  RS->RS+が多い日: 本物={( tv>1.0).mean()*100:.0f}%  "
                      f"S4={( s4v>1.0).mean()*100:.0f}%")
                print(f"  ★本物は >1 (下落優勢)、S4 は ≈1 (対称) なら境界確定")

    print(f"\n=== S4 low違反 ===")
    print(f"  mean={res['s4_low_viol'].mean():.1f}%  "
          f"0%の日={(res['s4_low_viol'] < 1e-6).sum()}/{len(res)}")

    print("\n" + "="*60)
    print("★ 境界の判定 ★")
    print("="*60)
    asym_t = res["true_asym"].dropna()
    asym_s = res["s4_asym"].dropna()
    idx    = asym_t.index.intersection(asym_s.index)
    gap_asym = np.median(asym_t[idx]) - np.median(asym_s[idx])
    gap_skew = np.median(res["true_skew"]) - np.median(res["s4_skew"])
    print(f"RS-/RS+ ギャップ (本物-S4): {gap_asym:+.4f}")
    print(f"  正 = 本物の方が下落優勢が強い = S4 で再現できてない = 境界")
    print(f"\n実現歪度 ギャップ (本物-S4): {gap_skew:+.4f}")
    print(f"  負 = 本物は負の歪度、S4 は≈0 = 理論通り")

    # ────────────────────────────────
    # アブレーションB: S2 vs S4 の比較
    # H/L情報を足した効果を量ごとに定量化
    # ────────────────────────────────
    print("\n" + "="*60)
    print("★ アブレーションB: 終値のみ(S2) vs OHLC全部(S4) ★")
    print("="*60)
    print("H/L情報を足すと、各量の再現度がどう変わるか\n")

    print(f"{'量':10} {'S2比率':>8} {'S2相関':>8} | {'S4比率':>8} {'S4相関':>8} | 改善")
    print("-"*70)

    for label, key in [("RV","rv"), ("MaxDD","maxdd"),
                       ("RS+","rsp"), ("RS-","rsm")]:
        tv  = res[f"true_{key}"].dropna()
        s2v = res[f"s2_{key}"].dropna()
        s4v = res[f"s4_{key}"].dropna()
        idx = tv.index.intersection(s2v.index).intersection(s4v.index)
        tv, s2v, s4v = tv[idx], s2v[idx], s4v[idx]

        s2_ratio = np.median(s2v/tv)
        s4_ratio = np.median(s4v/tv)
        s2_corr  = spearman(tv, s2v)
        s4_corr  = spearman(tv, s4v)

        # 相関の改善(1.0からの距離が縮んだか)
        corr_gain = s4_corr - s2_corr
        # 比率の改善(1.0からの距離が縮んだか)
        ratio_gain = abs(s2_ratio - 1.0) - abs(s4_ratio - 1.0)

        print(f"{label:10} {s2_ratio:8.3f} {s2_corr:8.3f} | "
              f"{s4_ratio:8.3f} {s4_corr:8.3f} | "
              f"相関+{corr_gain:.3f} 比率{ratio_gain:+.3f}")

    # RS-/RS+ 比と歪度は既に上で S2 も表示されてる
    print(f"\nRS-/RS+ 比:")
    print(f"  S2 mean={res['s2_asym'].mean():.3f}, "
          f"S4 mean={res['s4_asym'].mean():.3f}, "
          f"本物 mean={res['true_asym'].mean():.3f}")

    print(f"\n実現歪度:")
    print(f"  S2 median={res['s2_skew'].median():+.4f}, "
          f"S4 median={res['s4_skew'].median():+.4f}, "
          f"本物 median={res['true_skew'].median():+.4f}")

    import os; os.makedirs("../results", exist_ok=True)
    res.to_csv("../results/phase1_boundary_gk.csv", index=False)
    print(f"\nSaved: ../results/phase1_boundary_gk.csv")


if __name__ == "__main__":
    main()