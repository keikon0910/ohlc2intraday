"""
Notebook 04: Section 2 (close-only) vs Section 3 (close+high) on SPY.

Reflects the overnight review:
- log prices (robust for crisis/historical data)
- MLE sigma^2:  Section2 = c^2,  Section3 = (2h-c)^2/3
- two comparison modes: (a) practical (each method's own MLE),
                        (b) info-only (same sigma^2 for both)
- also report the LOW-violation rate (motivates Section 4)
"""
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

from ohlc2intraday.brownian_bridge import brownian_bridge_path
from ohlc2intraday.path_sampler import generate_path_ohc


def _f(x):
    """robust scalar extraction from a possibly-MultiIndex row."""
    return float(np.asarray(x).flatten()[0])


def eval_day(O, H, L, C, true_path_prices, n_paths=200, rng=None):
    """
    All computation in LOG space, shifted so log(open)=0.
    true_path_prices : 1d array of intraday prices (the true 1-min path).
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # --- to log, shift by log(open) ---
    lo = np.log(O)
    lO, lH, lL, lC = 0.0, np.log(H) - lo, np.log(L) - lo, np.log(C) - lo
    true_log = np.log(np.asarray(true_path_prices).astype(float)) - lo
    true_ret = np.diff(true_log)
    n = len(true_log) - 1
    true_sigma2 = true_ret.var() * n  # total daily variance (log)

    # === Section 2 (close-only) ===
    s2_sig2_mle = max(lC**2, 1e-16)               # MLE: E[c^2]=sigma^2
    s2_paths = np.array([
        brownian_bridge_path(lO, lC, n_steps=n, sigma2=s2_sig2_mle, rng=rng)
        for _ in range(n_paths)
    ])
    s2_ret = np.diff(s2_paths, axis=1).flatten()

    # === Section 3 (close+high), MLE sigma^2 = (2h-c)^2/3 ===
    s3_sig2_mle = max((2 * lH - lC) ** 2 / 3.0, 1e-16)
    s3_paths = np.array([
        generate_path_ohc(lO, lH, lC, n_steps=n, sigma2=s3_sig2_mle, rng=rng)
        for _ in range(n_paths)
    ])
    s3_ret = np.diff(s3_paths, axis=1).flatten()

    # === info-only mode: both use the TRUE sigma^2 ===
    s2b = np.array([
        brownian_bridge_path(lO, lC, n_steps=n, sigma2=true_sigma2, rng=rng)
        for _ in range(n_paths)
    ])
    s3b = np.array([
        generate_path_ohc(lO, lH, lC, n_steps=n, sigma2=true_sigma2, rng=rng)
        for _ in range(n_paths)
    ])
    s2b_ret = np.diff(s2b, axis=1).flatten()
    s3b_ret = np.diff(s3b, axis=1).flatten()

    # === low-violation rate (Section 3 ignores L) ===
    low_viol = (s3_paths.min(axis=1) < lL).mean() * 100.0

    # KS tests (practical mode)
    ks2 = stats.ks_2samp(true_ret, s2_ret)
    ks3 = stats.ks_2samp(true_ret, s3_ret)

    return {
        "true_sig": true_ret.std(),
        "s2_sig": s2_ret.std(), "s3_sig": s3_ret.std(),
        "s2b_sig": s2b_ret.std(), "s3b_sig": s3b_ret.std(),
        "ks2_p": ks2.pvalue, "ks3_p": ks3.pvalue,
        "low_viol": low_viol,
    }


def main():
    spy_daily = yf.download("SPY", period="5d", interval="1d")
    spy_1m = yf.download("SPY", period="5d", interval="1m")
    if isinstance(spy_daily.columns, pd.MultiIndex):
        spy_daily.columns = spy_daily.columns.droplevel(1)
    if isinstance(spy_1m.columns, pd.MultiIndex):
        spy_1m.columns = spy_1m.columns.droplevel(1)

    rng = np.random.default_rng(42)
    rows = []
    for i in range(len(spy_daily)):
        date = spy_daily.index[i].date()
        row = spy_daily.iloc[i]
        O, H, L, C = _f(row["Open"]), _f(row["High"]), _f(row["Low"]), _f(row["Close"])
        true_path = spy_1m[spy_1m.index.date == date]["Close"].values
        true_path = np.asarray(true_path).flatten().astype(float)
        if len(true_path) < 100:
            continue
        r = eval_day(O, H, L, C, true_path, n_paths=200, rng=rng)
        r["date"] = date
        rows.append(r)

    df = pd.DataFrame(rows)
    print("\n=== Practical mode (each method's MLE sigma^2) ===")
    print(f"{'date':>12} {'trueσ':>9} {'S2σ':>9} {'S3σ':>9} {'S2/t':>6} {'S3/t':>6} {'low%':>6}")
    for _, x in df.iterrows():
        print(f"{str(x['date']):>12} {x['true_sig']:>9.6f} {x['s2_sig']:>9.6f} "
              f"{x['s3_sig']:>9.6f} {x['s2_sig']/x['true_sig']:>6.2f} "
              f"{x['s3_sig']/x['true_sig']:>6.2f} {x['low_viol']:>6.1f}")

    print("\n=== Info-only mode (both use TRUE sigma^2) ===")
    print(f"{'date':>12} {'S2/t':>6} {'S3/t':>6}  (closer to 1.0 = high info helps shape)")
    for _, x in df.iterrows():
        print(f"{str(x['date']):>12} {x['s2b_sig']/x['true_sig']:>6.2f} "
              f"{x['s3b_sig']/x['true_sig']:>6.2f}")

    print("\n=== KS p-values (vs true return dist) ===")
    for _, x in df.iterrows():
        print(f"{str(x['date']):>12}  S2 p={x['ks2_p']:.4f}  S3 p={x['ks3_p']:.4f}")

    import os
    os.makedirs("../results", exist_ok=True)
    df.to_csv("../results/section3_evaluation.csv", index=False)
    print("\nSaved to ../results/section3_evaluation.csv")


if __name__ == "__main__":
    main()