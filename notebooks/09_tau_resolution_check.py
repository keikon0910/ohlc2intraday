# notebooks/09_tau_resolution_check.py
# tol_sigma=0.15 固定で n_steps 150 vs 390 を比較
# → 「収束した」が本物か、τの解像度(1/150)の壁かを分ける
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from ohlc2intraday.path_sampler_ohlc import prepass_argtimes

DATA_PATH = "../data/SPX_full_1min.txt"
PREV_CSV  = "../results/tol_convergence.csv"
TOL_SIGMA = 0.15
TOTAL     = 500_000
SEED      = 0

CONFIGS = [(150, 100_000), (390, 50_000)]   # (n_steps, batch)


def load_firstrate(path):
    df = pd.read_csv(path, header=None,
        names=["datetime","open","high","low","close"],
        usecols=[0,1,2,3,4], parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def main():
    df   = load_firstrate(DATA_PATH)
    prev = pd.read_csv(PREV_CSV, parse_dates=["date"])
    days = (prev[prev["tol_sigma"] == TOL_SIGMA]
            [["label", "date"]].drop_duplicates())
    print(f"tol_sigma={TOL_SIGMA} 固定、n_steps を {[c[0] for c in CONFIGS]} で比較")
    print(f"1条件あたり {TOTAL:,} サンプル\n")

    recs = []
    for _, r in days.iterrows():
        d  = r["date"].date()
        dd = df[df.index.date == d]
        O = float(dd["open"].iloc[0]);  H = float(dd["high"].max())
        L = float(dd["low"].min());     C = float(dd["close"].iloc[-1])
        lo = np.log(O)
        lH, lL, lC = np.log(H)-lo, np.log(L)-lo, np.log(C)-lo
        gk = 0.5*(lH-lL)**2 - (2*np.log(2)-1)*lC**2
        s2 = max(gk, (2*lH-lC)**2/3.0, 1e-16)

        out = {}
        for n_steps, batch in CONFIGS:
            pool = prepass_argtimes(
                lC, lH, lL, s2,
                n_steps=n_steps, total=TOTAL, batch=batch,
                tol_sigma=TOL_SIGMA, target_pool=10**12, seed=SEED,
            )
            if pool is None or len(pool) < 100:
                out[n_steps] = None
                continue
            tauL, tauH = pool[:, 0], pool[:, 1]
            out[n_steps] = dict(n=len(pool),
                                p_low=(tauL < tauH).mean(),
                                med_L=np.median(tauL),
                                med_H=np.median(tauH))

        a, b = out[150], out[390]
        if a is None or b is None:
            print(f"■ {r['label']:12s} {d}  → 採択不足、スキップ")
            continue

        dp  = abs(b["p_low"] - a["p_low"])
        dL  = abs(b["med_L"] - a["med_L"])
        dH  = abs(b["med_H"] - a["med_H"])
        print(f"■ {r['label']:12s} {d}")
        print(f"   n_steps=150: n={a['n']:>7d}  P(L先)={a['p_low']:.3f}  "
              f"τL={a['med_L']:.3f}  τH={a['med_H']:.3f}")
        print(f"   n_steps=390: n={b['n']:>7d}  P(L先)={b['p_low']:.3f}  "
              f"τL={b['med_L']:.3f}  τH={b['med_H']:.3f}")
        print(f"   差:          ΔP={dp:.3f}  ΔτL={dL:.3f}  ΔτH={dH:.3f}\n")

        recs.append(dict(label=r["label"], date=d,
                         dp=dp, dL=dL, dH=dH))

    res = pd.DataFrame(recs)
    print("="*60)
    print(f"比較日数={len(res)}")
    print(f"|ΔP(L先)|  平均={res['dp'].mean():.3f}  最大={res['dp'].max():.3f}")
    print(f"|ΔτL中央|  平均={res['dL'].mean():.3f}  最大={res['dL'].max():.3f}")
    print(f"|ΔτH中央|  平均={res['dH'].mean():.3f}  最大={res['dH'].max():.3f}")
    print()
    print("判定:")
    print("  ΔP(L先) < 0.02        → 順序は解像度に鈍感、150で十分")
    print("  Δτ中央 < 0.007 (=1/150) → 格子の粒度以下、差とみなせない")
    print("  それ以上動く → 150 の解像度が制約になっている")

    import os; os.makedirs("../results", exist_ok=True)
    res.to_csv("../results/tau_resolution_check.csv", index=False)
    print("\nSaved: ../results/tau_resolution_check.csv")


if __name__ == "__main__":
    main()