# tol_sigma の収束確認
# prepass だけ回してプール統計が動かなくなる点を探す
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from ohlc2intraday.path_sampler_ohlc import prepass_argtimes

DATA_PATH   = "../data/SPX_full_1min.txt"
N_RECENT    = 250
TOL_SIGMAS  = [0.3, 0.15, 0.075, 0.0375]
TOTAL       = 2_000_000     # 早期停止させないため target_pool は巨大に
SEED        = 0


def load_firstrate(path):
    df = pd.read_csv(path, header=None,
        names=["datetime","open","high","low","close"],
        usecols=[0,1,2,3,4], parse_dates=["datetime"])
    return df.set_index("datetime").sort_index()


def day_ohlc(day_df):
    O = float(day_df["open"].iloc[0])
    H = float(day_df["high"].max())
    L = float(day_df["low"].min())
    C = float(day_df["close"].iloc[-1])
    if not (H >= max(O,C) and L <= min(O,C) and O > 0 and L > 0):
        return None
    lo = np.log(O)
    return np.log(H)-lo, np.log(L)-lo, np.log(C)-lo


def select_shaped_days(df):
    """形の違う10日を意図的に選ぶ(レンジ広/狭、C≈H寄り/C≈L寄り/中立)"""
    dates = sorted(set(df.index.date))[-N_RECENT:]
    rows = []
    for d in dates:
        dd = df[df.index.date == d]
        if len(dd) < 100:
            continue
        o = day_ohlc(dd)
        if o is None:
            continue
        lH, lL, lC = o
        rng_w = lH - lL
        c_pos = (lC - lL) / rng_w if rng_w > 0 else 0.5   # 0=L寄り 1=H寄り
        rows.append(dict(date=d, lH=lH, lL=lL, lC=lC, rng=rng_w, cpos=c_pos))

    s = pd.DataFrame(rows)
    picks = {}
    picks["レンジ最小"]   = s.nsmallest(1, "rng").iloc[0]
    picks["レンジ最大"]   = s.nlargest(1, "rng").iloc[0]
    picks["レンジ小(25%)"] = s.sort_values("rng").iloc[len(s)//4]
    picks["レンジ大(75%)"] = s.sort_values("rng").iloc[3*len(s)//4]
    picks["C最もL寄り"]   = s.nsmallest(1, "cpos").iloc[0]
    picks["C最もH寄り"]   = s.nlargest(1, "cpos").iloc[0]
    picks["C中立"]        = s.iloc[(s["cpos"]-0.5).abs().argsort().iloc[0]]
    rng = np.random.default_rng(42)
    for i, j in enumerate(rng.choice(len(s), size=3, replace=False)):
        picks[f"ランダム{i+1}"] = s.iloc[j]

    # 重複除去(同じ日が複数ラベルに来ることがある)
    out, seen = [], set()
    for label, row in picks.items():
        if row["date"] in seen:
            continue
        seen.add(row["date"])
        out.append((label, row))
    return out


def main():
    df = load_firstrate(DATA_PATH)
    print(f"loaded: {len(df):,} rows\n")
    days = select_shaped_days(df)
    print(f"選んだ日: {len(days)}日  /  tol_sigma: {TOL_SIGMAS}")
    print(f"1条件あたり {TOTAL:,} サンプル生成(早期停止なし)\n")

    recs = []
    for label, row in days:
        lH, lL, lC = row["lH"], row["lL"], row["lC"]
        gk = 0.5*(lH-lL)**2 - (2*np.log(2)-1)*lC**2
        s2 = max(gk, (2*lH-lC)**2/3.0, 1e-16)

        print(f"■ {label}  {row['date']}  "
              f"レンジ={row['rng']*100:.2f}%  C位置={row['cpos']:.2f}  "
              f"σ={np.sqrt(s2)*100:.2f}%")
        print(f"   {'tol_σ':>8} {'採択数':>8} {'採択率':>9} "
              f"{'P(L先)':>8} {'τL中央':>8} {'τH中央':>8}")

        for ts in TOL_SIGMAS:
            pool = prepass_argtimes(
                lC, lH, lL, s2,
                total=TOTAL, batch=100_000,
                tol_sigma=ts, target_pool=10**12,   # 早期停止させない
                seed=SEED,
            )
            n = 0 if pool is None else len(pool)
            rate = n / TOTAL * 100
            if n >= 30:
                tauL, tauH = pool[:,0], pool[:,1]
                p_low  = (tauL < tauH).mean()
                med_L, med_H = np.median(tauL), np.median(tauH)
                print(f"   {ts:>8.4f} {n:>8d} {rate:>8.4f}% "
                      f"{p_low:>8.3f} {med_L:>8.3f} {med_H:>8.3f}")
            else:
                p_low = med_L = med_H = np.nan
                print(f"   {ts:>8.4f} {n:>8d} {rate:>8.4f}% "
                      f"{'不足':>8} {'-':>8} {'-':>8}")
            recs.append(dict(label=label, date=row["date"], tol_sigma=ts,
                             n=n, rate=rate, p_low=p_low,
                             med_L=med_L, med_H=med_H))
        print()

    res = pd.DataFrame(recs)

    print("="*64)
    print("収束チェック: 隣り合う tol_sigma 間で P(L先) がどれだけ動くか")
    print("="*64)
    for a, b in zip(TOL_SIGMAS[:-1], TOL_SIGMAS[1:]):
        sub = res[res["tol_sigma"].isin([a,b])].dropna(subset=["p_low"])
        piv = sub.pivot_table(index="label", columns="tol_sigma", values="p_low")
        piv = piv.dropna()
        if len(piv) == 0:
            print(f"{a} → {b}: 比較可能な日なし")
            continue
        d = (piv[b] - piv[a]).abs()
        print(f"{a} → {b}:  比較日数={len(piv)}  "
              f"|ΔP(L先)| 平均={d.mean():.3f}  最大={d.max():.3f}")

    print("\n判定の目安:")
    print("  |ΔP(L先)| 平均 < 0.02 なら、その手前の tol_sigma で収束とみなす")
    print("  採択数 < 300 の条件は統計が薄い(本番でスキップになる予兆)")

    import os; os.makedirs("../results", exist_ok=True)
    res.to_csv("../results/tol_convergence.csv", index=False)
    print("\nSaved: ../results/tol_convergence.csv")


if __name__ == "__main__":
    main()