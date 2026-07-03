# notebook 06: OHLC全条件サンプラー (Section 4) の250日評価
# notebook 05 と同じ枠組みで S2/S3/S4 を並べ、S4 の low違反 0% を確認する。
# - S3 と S4 は同じ GK sigma2 を使う → 「low違反だけが S4 の差分」になる設計
# - 各手法は独立した子RNGで回す → 結果が手法の呼び出し順に依存しない
# - low違反判定は 1e-7 の許容誤差付き (縮退ガードが安値を eps0≈8e-9 ずらすため)
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd

from ohlc2intraday.brownian_bridge import brownian_bridge_path
from ohlc2intraday.path_sampler import generate_path_ohc
from ohlc2intraday.path_sampler_ohlc import make_sampler_ohlc

DATA_PATH = "../data/SPX_full_1min.txt"
N_DAYS = 250
N_PATHS = 100
SEED = 42
LOW_TOL = 1e-7  # 縮退ガード(安値=終値の日に安値を eps0 ずらす)の誤差を許容


def load_firstrate(path):
    df = pd.read_csv(
        path, header=None,
        names=["datetime", "open", "high", "low", "close"],
        usecols=[0, 1, 2, 3, 4], parse_dates=["datetime"],
    )
    return df.set_index("datetime").sort_index()


def eval_day(day_df, n_paths, day_seed):
    # 1日分の OHLC と1分足から S2/S3/S4 を評価。log空間で統一。
    # 各手法に独立した RNG を割り当てて呼び出し順に依存しないようにする。
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
    lH, lL, lC = np.log(H) - lo, np.log(L) - lo, np.log(C) - lo
    true_log = np.log(prices) - lo
    true_ret = np.diff(true_log)
    n = len(true_log) - 1
    t_std = true_ret.std()
    if t_std == 0:
        return None

    # 子RNGを4つ作る (S2, S3, S4 prepass, S4 paths)
    ss = np.random.SeedSequence(day_seed)
    rng_s2, rng_s3, rng_s4_prep, rng_s4_path = [
        np.random.default_rng(s) for s in ss.spawn(4)
    ]

    # S2: 終値のみ, sigma2 = c^2
    s2_sig2 = max(lC ** 2, 1e-16)
    s2 = np.array([
        brownian_bridge_path(0.0, lC, n_steps=n, sigma2=s2_sig2, rng=rng_s2)
        for _ in range(n_paths)
    ])
    s2_std = np.diff(s2, axis=1).std()

    # S3: 終値+高値, sigma2 = Garman-Klass (GKが負なら (2h-c)^2/3 にフォールバック)
    gk = 0.5 * (lH - lL) ** 2 - (2 * np.log(2) - 1) * lC ** 2
    s3_sig2 = max(gk, (2 * lH - lC) ** 2 / 3.0, 1e-16)
    s3 = np.array([
        generate_path_ohc(0.0, lH, lC, n_steps=n, sigma2=s3_sig2, rng=rng_s3)
        for _ in range(n_paths)
    ])
    s3_std = np.diff(s3, axis=1).std()
    s3_low_viol = (s3.min(axis=1) < lL - LOW_TOL).mean() * 100.0

    # S4: OHLC全条件 (sigma2 は S3 と同じGKで揃える → 「low違反だけが差分」)
    try:
        sampler, _ = make_sampler_ohlc(
            0.0, lH, lL, lC, n_steps=n, sigma2=s3_sig2,
            prepass_kwargs=dict(seed=int(rng_s4_prep.integers(1 << 31)),
                                target_pool=300, batch=30_000, total=500_000),
        )
        s4 = np.array([sampler(rng_s4_path) for _ in range(n_paths)])
        s4_std = np.diff(s4, axis=1).std()
        s4_low_viol = (s4.min(axis=1) < lL - LOW_TOL).mean() * 100.0
    except RuntimeError:
        # prepass が集まらなかった日 (極端なOHLC) はスキップ
        return None

    return {
        "true_std": t_std,
        "s2_ratio": s2_std / t_std,
        "s3_ratio": s3_std / t_std,
        "s4_ratio": s4_std / t_std,
        "s3_low_viol": s3_low_viol,
        "s4_low_viol": s4_low_viol,
    }


def main():
    df = load_firstrate(DATA_PATH)
    print(f"loaded: {len(df):,} rows, {df.index.min()} -> {df.index.max()}")

    dates = sorted(set(df.index.date))[-N_DAYS:]
    rows = []
    skipped = 0
    for i, d in enumerate(dates):
        day_df = df[df.index.date == d]
        day_seed = SEED * 1_000_000 + i
        r = eval_day(day_df, N_PATHS, day_seed)
        if r is not None:
            r["date"] = d
            rows.append(r)
        else:
            skipped += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(dates)} 日処理 (スキップ {skipped}日)")
    res = pd.DataFrame(rows)
    print(f"\nevaluated: {len(res)} days, skipped: {skipped} days\n")

    def closer_to_one(a, b):
        return np.abs(np.log(a)) < np.abs(np.log(b))

    s4_beats_s3 = closer_to_one(res["s4_ratio"], res["s3_ratio"]).mean() * 100
    s4_beats_s2 = closer_to_one(res["s4_ratio"], res["s2_ratio"]).mean() * 100

    print("=== ratio to true sigma (1.0 = perfect) ===")
    for col in ["s2_ratio", "s3_ratio", "s4_ratio"]:
        q = res[col].quantile([0.25, 0.5, 0.75])
        print(f"  {col}: median={q[0.5]:.3f}  IQR=[{q[0.25]:.3f}, {q[0.75]:.3f}]")
    print(f"\n  S4 closer to 1.0 than S3: {s4_beats_s3:.1f}% of days")
    print(f"  S4 closer to 1.0 than S2: {s4_beats_s2:.1f}% of days")

    print("\n=== low violation (合成パスが安値を突き破る率) ===")
    print(f"  S3: median={res['s3_low_viol'].median():.1f}%  mean={res['s3_low_viol'].mean():.1f}%")
    print(f"  S4: median={res['s4_low_viol'].median():.1f}%  mean={res['s4_low_viol'].mean():.1f}%")

    import os
    os.makedirs("../results", exist_ok=True)
    res.to_csv("../results/long_term_s2_s3_s4.csv", index=False)
    print("\nSaved to ../results/long_term_s2_s3_s4.csv")


if __name__ == "__main__":
    main()