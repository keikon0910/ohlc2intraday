"""
Notebook 05: Long-term evaluation of Section 2 vs Section 3
using FirstRateData SPX 1-min as ground truth.

- aggregates daily OHLC from the 1-min bars themselves
  (true OHLC and true intraday returns come from the same source)
- runs S2 (close-only) and S3 (close+high) on each day
- reports distributional summaries over many days
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd

from ohlc2intraday.brownian_bridge import brownian_bridge_path
from ohlc2intraday.path_sampler import generate_path_ohc

# ===== settings =====
DATA_PATH = "../data/SPX_full_1min.txt"
N_DAYS = 250        # 直近何営業日を評価するか
N_PATHS = 100       # 1日あたりの生成パス数
SEED = 42


def load_firstrate(path):
    """FirstRateData 1-min file -> DataFrame with DatetimeIndex."""
    df = pd.read_csv(
        path,
        header=None,
        names=["datetime", "open", "high", "low", "close"],
        usecols=[0, 1, 2, 3, 4],
        parse_dates=["datetime"],
    )
    return df.set_index("datetime").sort_index()


def eval_day(day_df, n_paths, rng):
    """1日分の1分足から S2/S3 を評価。log空間で統一。"""
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

    # --- S2: close-only, MLE sigma2 = c^2 ---
    s2_sig2 = max(lC ** 2, 1e-16)
    s2 = np.array([
        brownian_bridge_path(0.0, lC, n_steps=n, sigma2=s2_sig2, rng=rng)
        for _ in range(n_paths)
    ])
    s2_std = np.diff(s2, axis=1).std()

    # --- S3: close+high, MLE sigma2 = (2h-c)^2/3 ---
    s3_sig2 = max((2 * lH - lC) ** 2 / 3.0, 1e-16)
    s3 = np.array([
        generate_path_ohc(0.0, lH, lC, n_steps=n, sigma2=s3_sig2, rng=rng)
        for _ in range(n_paths)
    ])
    s3_std = np.diff(s3, axis=1).std()
    low_viol = (s3.min(axis=1) < lL).mean() * 100.0

    return {
        "true_std": t_std,
        "s2_ratio": s2_std / t_std,
        "s3_ratio": s3_std / t_std,
        "low_viol": low_viol,
    }


def main():
    rng = np.random.default_rng(SEED)
    df = load_firstrate(DATA_PATH)
    print(f"loaded: {len(df):,} rows, {df.index.min()} -> {df.index.max()}")

    dates = sorted(set(df.index.date))[-N_DAYS:]
    rows = []
    for d in dates:
        day_df = df[df.index.date == d]
        r = eval_day(day_df, N_PATHS, rng)
        if r is not None:
            r["date"] = d
            rows.append(r)
    res = pd.DataFrame(rows)
    print(f"evaluated: {len(res)} days\n")

    def closer_to_one(a, b):
        return np.abs(np.log(a)) < np.abs(np.log(b))

    s3_wins = closer_to_one(res["s3_ratio"], res["s2_ratio"]).mean() * 100

    print("=== ratio to true sigma (1.0 = perfect) ===")
    for col in ["s2_ratio", "s3_ratio"]:
        q = res[col].quantile([0.25, 0.5, 0.75])
        print(f"  {col}: median={q[0.5]:.3f}  IQR=[{q[0.25]:.3f}, {q[0.75]:.3f}]")
    print(f"\n  S3 closer to 1.0 than S2: {s3_wins:.1f}% of days")
    print(f"  low violation: median={res['low_viol'].median():.1f}%  "
          f"mean={res['low_viol'].mean():.1f}%")

    import os
    os.makedirs("../results", exist_ok=True)
    res.to_csv("../results/long_term_s2_vs_s3.csv", index=False)
    print("\nSaved to ../results/long_term_s2_vs_s3.csv")


if __name__ == "__main__":
    main()