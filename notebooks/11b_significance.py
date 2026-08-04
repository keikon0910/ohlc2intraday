# notebooks/11b_significance.py
# 到達ρと天井ρの差の有意性検定(Williams–Steiger)
# 11番と同じ共通日集合の上で、各指標について
#   ρ(本物, S4)   … 到達
#   ρ(本物, 予測) … 天井(同期間)
# の差を検定する。両者は「本物」を共有する従属相関なので、
# 共有変数つきの従属相関の比較(Williams 1959 / Steiger 1980)を使う。
# スピアマン版: 3変数を順位変換してからピアソン相関を取り、同検定を適用。
#
# 使い方: python 11b_significance.py [地図CSVパス]
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, t as t_dist

PRED_PATH = "../results/ceiling_predictions_set2.csv"
MAP_CANDIDATES = [
    "../results/phase1_boundary_gk.csv",
    "../results/phase1_boundary_07a.csv",
    "../results/phase1_boundary.csv",
]

# (表示名, 地図のtrue列, 地図のs4列, 10番のターゲット名)
TARGETS = [
    ("RV",        "true_rv",    "s4_rv",    "t_rv"),
    ("MaxDD",     "true_maxdd", "s4_maxdd", "t_maxdd"),
    ("RS+",       "true_rsp",   "s4_rsp",   "t_rsp"),
    ("RS-",       "true_rsm",   "s4_rsm",   "t_rsm"),
    ("RS-/RS+比", "true_asym",  "s4_asym",  "t_asym"),
    ("実現歪度",   "true_skew",  "s4_skew",  "t_skew"),
]


def williams_steiger(x_true, x_a, x_b):
    """共有変数(x_true)つき従属相関 r(true,a) vs r(true,b) の差の検定。
    スピアマン版: 順位変換後のピアソン相関に Williams の t を適用する。
    戻り値: (rho_a, rho_b, t値, 両側p, n)"""
    m = np.isfinite(x_true) & np.isfinite(x_a) & np.isfinite(x_b)
    xt, xa, xb = (rankdata(x_true[m]), rankdata(x_a[m]), rankdata(x_b[m]))
    n = int(m.sum())
    if n < 10:
        return np.nan, np.nan, np.nan, np.nan, n

    r_ta = np.corrcoef(xt, xa)[0, 1]   # 到達(順位相関)
    r_tb = np.corrcoef(xt, xb)[0, 1]   # 天井(順位相関)
    r_ab = np.corrcoef(xa, xb)[0, 1]   # S4と予測の相関

    detR = 1 - r_ta**2 - r_tb**2 - r_ab**2 + 2*r_ta*r_tb*r_ab
    rbar = (r_ta + r_tb) / 2.0
    denom = 2 * (n - 1) / (n - 3) * detR + rbar**2 * (1 - r_ab)**3
    if denom <= 0:
        return r_ta, r_tb, np.nan, np.nan, n
    t_val = (r_ta - r_tb) * np.sqrt((n - 1) * (1 + r_ab) / denom)
    p = 2 * t_dist.sf(abs(t_val), df=n - 3)
    return r_ta, r_tb, float(t_val), float(p), n


def find_map_csv():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for c in MAP_CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit("地図CSVが見つからない。引数でパスを渡して")


def main():
    mp = pd.read_csv(find_map_csv())
    mp["date"] = mp["date"].astype(str)
    pr = pd.read_csv(PRED_PATH)
    pr["date"] = pr["date"].astype(str)
    j = mp.merge(pr, on="date", how="inner")
    print(f"共通日数: {len(j)}")

    print(f"\n{'指標':10s} {'到達ρ':>8} {'天井ρ':>8} {'差':>8} "
          f"{'t':>7} {'p値':>9}  判定")
    print("-" * 62)
    rows = []
    for label, tcol, scol, tg in TARGETS:
        pc = f"pred_{tg}"
        if not all(c in j.columns for c in (tcol, scol, pc)):
            print(f"{label:10s} 列なし→スキップ")
            continue
        ra, rb, tv, p, n = williams_steiger(
            j[tcol].values.astype(float),
            j[scol].values.astype(float),
            j[pc].values.astype(float))
        verdict = ("n.s.(差を検出できない)" if p >= 0.05 else "有意")
        print(f"{label:10s} {ra:+8.3f} {rb:+8.3f} {rb-ra:+8.3f} "
              f"{tv:7.2f} {p:9.4f}  {verdict}")
        rows.append(dict(target=label, n=n, achieved=ra, ceiling=rb,
                         diff=rb - ra, t=tv, p=p))

    out = "../results/ceiling_significance.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print("注: 天井は下からの近似のため、n.s. は「改良余地ゼロの証明」ではなく")
    print("    「この標本サイズでは差を検出できない」の意。")


if __name__ == "__main__":
    main()