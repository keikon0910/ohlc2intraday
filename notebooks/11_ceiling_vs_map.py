# notebooks/11_ceiling_vs_map.py
# 穴埋めスクリプト:
#  穴2: 歪度・RS-/RS+比の「到達ρ」を地図CSV(07a出力)から計算
#  穴1: 天井(10番のCV予測)と到達(地図)を「同一の日集合」の上で比較
# 使い方:
#   python 11_ceiling_vs_map.py [地図CSVパス]
#   地図CSVパス省略時は ../results/ 内の候補を自動探索
import sys
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PRED_PATH = "../results/ceiling_predictions_set2.csv"
MAP_CANDIDATES = [
    "../results/phase1_boundary_gk.csv",
    "../results/phase1_boundary_07a.csv",
    "../results/phase1_boundary.csv",
]

# (表示名, 地図CSVのtrue列, 地図CSVのs4列, 10番のターゲット名)
TARGETS = [
    ("RV",        "true_rv",    "s4_rv",    "t_rv"),
    ("MaxDD",     "true_maxdd", "s4_maxdd", "t_maxdd"),
    ("RS+",       "true_rsp",   "s4_rsp",   "t_rsp"),
    ("RS-",       "true_rsm",   "s4_rsm",   "t_rsm"),
    ("RS-/RS+比", "true_asym",  "s4_asym",  "t_asym"),
    ("実現歪度",   "true_skew",  "s4_skew",  "t_skew"),
]


def spear(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10:
        return np.nan, int(m.sum())
    return spearmanr(a[m], b[m])[0], int(m.sum())


def find_map_csv():
    if len(sys.argv) > 1:
        return sys.argv[1]
    for c in MAP_CANDIDATES:
        if os.path.exists(c):
            return c
    sys.exit("地図CSVが見つからない。引数でパスを渡して: "
             "python 11_ceiling_vs_map.py ../results/xxx.csv")


def main():
    map_path = find_map_csv()
    mp = pd.read_csv(map_path)
    mp["date"] = mp["date"].astype(str)
    print(f"地図CSV: {map_path}  ({len(mp)}日)")

    # ── 穴2: 地図単独で計算できる「到達ρ」(歪度・asym含む全指標)──
    print("\n" + "=" * 72)
    print("穴2: 到達ρ(地図の全指標、S4 vs 本物のスピアマン)")
    print("=" * 72)
    achieved = {}
    for label, tcol, scol, _ in TARGETS:
        if tcol not in mp.columns or scol not in mp.columns:
            print(f"  {label:10s} 列なし({tcol}/{scol})→スキップ")
            continue
        rho, n = spear(mp[tcol].values.astype(float),
                       mp[scol].values.astype(float))
        achieved[label] = rho
        print(f"  {label:10s} 到達ρ={rho:+.3f}  (n={n})")

    # ── 穴1: 同一の日集合での 到達ρ vs 天井ρ ──
    if not os.path.exists(PRED_PATH):
        print(f"\n{PRED_PATH} が無い。先に修正版10番を実行して"
              "CV予測を保存してから再実行して。")
        return
    pr = pd.read_csv(PRED_PATH)
    pr["date"] = pr["date"].astype(str)
    j = mp.merge(pr, on="date", how="inner")
    print("\n" + "=" * 72)
    print(f"穴1: 同一期間比較(日付の共通部分 n={len(j)}日)")
    print("=" * 72)

    # 整合性チェック: 同じ日の同じ指標の「本物の値」が両スクリプトで一致するか
    # (半日除外基準の違い等で日集合はズレてよいが、値はズレてはいけない)
    for label, tcol, _, tg in TARGETS:
        tc, cc = f"true_{tg}", tcol
        if tc in j.columns and cc in j.columns:
            rho, n = spear(j[cc].values.astype(float),
                           j[tc].values.astype(float))
            if np.isfinite(rho) and rho < 0.999:
                print(f"  ⚠ 整合性警告: {label} の本物の値が"
                      f"2スクリプト間でズレてる(ρ={rho:.4f})。要調査")

    print(f"\n{'指標':10s} {'到達ρ':>8} {'天井ρ(同期間)':>14} {'伸びしろ':>10}")
    print("-" * 48)
    rows = []
    for label, tcol, scol, tg in TARGETS:
        pc = f"pred_{tg}"
        if not all(c in j.columns for c in (tcol, scol, pc)):
            continue
        a, _ = spear(j[tcol].values.astype(float),
                     j[scol].values.astype(float))
        c, n = spear(j[tcol].values.astype(float),
                     j[pc].values.astype(float))
        gap = c - a if np.isfinite(a) and np.isfinite(c) else np.nan
        print(f"{label:10s} {a:+8.3f} {c:+14.3f} {gap:+10.3f}")
        rows.append(dict(target=label, n=n, achieved=a,
                         ceiling_sameperiod=c, headroom=gap))

    pd.DataFrame(rows).to_csv("../results/ceiling_vs_map_sameperiod.csv",
                              index=False)
    j[["date"]].to_csv("../results/ceiling_common_days.csv", index=False)
    print("\nSaved: ../results/ceiling_vs_map_sameperiod.csv")
    print("\n読み方: 天井は下からの近似。到達>天井は「回帰側が天井に未達」の意。")
    print("地図を全期間に拡張して07aを再実行すれば、この表がそのまま全期間版になる。")
    


if __name__ == "__main__":
    main()