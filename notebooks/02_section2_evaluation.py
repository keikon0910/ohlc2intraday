"""
Notebook 02: Evaluate Section 2 Brownian bridge on multiple days.

Statistical validation:
- KS test (returns distribution)
- Wasserstein distance
- Visual comparison
"""
import sys
sys.path.insert(0, '..')

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from ohlc2intraday.brownian_bridge import brownian_bridge_path


def evaluate_one_day(daily_row, true_path, n_paths=100, seed=42):
    """Generate synthetic paths and compute metrics for one day."""
    # MultiIndex 対応 + 強制的に float に
    O = float(np.asarray(daily_row["Open"]).flatten()[0])
    C = float(np.asarray(daily_row["Close"]).flatten()[0])
    
    # true_path を 1次元の float 配列に強制
    true_path = np.asarray(true_path).flatten().astype(float)
    
    n_steps = len(true_path) - 1
    
    # generate synthetic paths
    rng = np.random.default_rng(seed)
    synthetic = np.array([
        brownian_bridge_path(O, C, n_steps=n_steps, rng=rng)
        for _ in range(n_paths)
    ])
    
    # compute returns (必ず1次元にする)
    true_returns = np.diff(np.log(true_path))
    synthetic_returns = np.diff(np.log(synthetic), axis=1).flatten()
    
    # 念のため 1次元化
    true_returns = np.asarray(true_returns).flatten()
    synthetic_returns = np.asarray(synthetic_returns).flatten()
    
    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(true_returns, synthetic_returns)
    
    # Wasserstein
    wasserstein = stats.wasserstein_distance(true_returns, synthetic_returns)
    
    return {
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "wasserstein": wasserstein,
        "true_std": true_returns.std(),
        "synth_std": synthetic_returns.std(),
        "true_mean": true_returns.mean(),
        "synth_mean": synthetic_returns.mean(),
    }


def main():
    # Get last 5 days of SPY data
    spy_daily = yf.download("SPY", period="5d", interval="1d")
    spy_1m = yf.download("SPY", period="5d", interval="1m")
    
    # MultiIndex を潰す
    if isinstance(spy_daily.columns, pd.MultiIndex):
        spy_daily.columns = spy_daily.columns.droplevel(1)
    if isinstance(spy_1m.columns, pd.MultiIndex):
        spy_1m.columns = spy_1m.columns.droplevel(1)
    
    print(f"Daily shape: {spy_daily.shape}")
    print(f"1-min shape: {spy_1m.shape}")
    print(f"Daily columns: {spy_daily.columns.tolist()}")
    
    results = []
    for i in range(len(spy_daily)):
        date = spy_daily.index[i].date()
        daily_row = spy_daily.iloc[i]
        true_path = spy_1m[spy_1m.index.date == date]["Close"].values
        
        print(f"\n--- Day {i}: {date} ---")
        print(f"  O: {float(np.asarray(daily_row['Open']).flatten()[0]):.2f}, "
              f"C: {float(np.asarray(daily_row['Close']).flatten()[0]):.2f}")
        print(f"  true_path shape: {np.asarray(true_path).shape}")
        
        if len(np.asarray(true_path).flatten()) < 100:
            print("  Skipping (too short)")
            continue
        
        metrics = evaluate_one_day(daily_row, true_path)
        metrics["date"] = date
        results.append(metrics)
        
        print(f"  KS={metrics['ks_stat']:.3f}, "
              f"p={metrics['ks_pvalue']:.3f}, "
              f"Wasserstein={metrics['wasserstein']:.5f}")
    
    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df[["date", "ks_stat", "ks_pvalue", "wasserstein", 
              "true_std", "synth_std"]].to_string(index=False))
    
    # save
    import os
    os.makedirs("../results", exist_ok=True)
    df.to_csv("../results/section2_evaluation.csv", index=False)
    print("\nResults saved to ../results/section2_evaluation.csv")


if __name__ == "__main__":
    main()