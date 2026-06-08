# notebooks/03_section2_distribution.py
"""Plot return distributions: true vs synthetic."""
import sys
sys.path.insert(0, '..')
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from ohlc2intraday.brownian_bridge import brownian_bridge_path


def main():
    spy_daily = yf.download("SPY", period="5d", interval="1d")
    spy_1m = yf.download("SPY", period="5d", interval="1m")
    
    if isinstance(spy_daily.columns, pd.MultiIndex):
        spy_daily.columns = spy_daily.columns.droplevel(1)
    if isinstance(spy_1m.columns, pd.MultiIndex):
        spy_1m.columns = spy_1m.columns.droplevel(1)
    
    fig, axes = plt.subplots(1, len(spy_daily), figsize=(20, 4))
    
    for i in range(len(spy_daily)):
        date = spy_daily.index[i].date()
        daily_row = spy_daily.iloc[i]
        true_path = np.asarray(
            spy_1m[spy_1m.index.date == date]["Close"].values
        ).flatten().astype(float)
        
        if len(true_path) < 100:
            continue
        
        O = float(np.asarray(daily_row["Open"]).flatten()[0])
        C = float(np.asarray(daily_row["Close"]).flatten()[0])
        
        rng = np.random.default_rng(42)
        synthetic = np.array([
            brownian_bridge_path(O, C, n_steps=len(true_path) - 1, rng=rng)
            for _ in range(100)
        ])
        
        true_returns = np.diff(np.log(true_path))
        synth_returns = np.diff(np.log(synthetic), axis=1).flatten()
        
        ax = axes[i]
        ax.hist(true_returns, bins=30, alpha=0.5, label="True", color="red", density=True)
        ax.hist(synth_returns, bins=30, alpha=0.5, label="Synth", color="blue", density=True)
        ax.set_title(f"{date}")
        ax.set_xlabel("Log return")
        ax.legend()
    
    plt.tight_layout()
    
    import os
    os.makedirs("../figures", exist_ok=True)
    plt.savefig("../figures/03_return_distributions.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()