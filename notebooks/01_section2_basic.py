"""
Notebook 01: Test basic Brownian bridge on SPY 1-day data.

Compare 100 generated paths to the true SPY 1-min path.
"""
import sys
sys.path.insert(0, '..')  # parent dir (ohlc2intraday)

import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from ohlc2intraday.brownian_bridge import brownian_bridge_path


def main():
    # 1. Get SPY daily OHLC
    spy = yf.download("SPY", period="5d", interval="1d")
    day = spy.iloc[-1]
    print("Day:", spy.index[-1].date())
    print("Open:", day["Open"].item())
    print("Close:", day["Close"].item())
    
    O = day["Open"].item()
    C = day["Close"].item()
    
    # 2. Get 1-min ground truth (last 7 days only, free from Yahoo)
    spy_1m = yf.download("SPY", period="5d", interval="1m")
    target_date = spy.index[-1].date()
    true_path = spy_1m[spy_1m.index.date == target_date]["Close"].values
    print(f"True path length: {len(true_path)}")
    
    # 3. Generate 100 synthetic paths
    rng = np.random.default_rng(42)
    synthetic_paths = np.array([
        brownian_bridge_path(O, C, n_steps=len(true_path) - 1, rng=rng)
        for _ in range(100)
    ])
    
    # 4. Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    for path in synthetic_paths:
        ax.plot(path, color="blue", alpha=0.05)
    ax.plot(true_path, color="red", linewidth=2, label="True (Yahoo 1-min)")
    ax.set_title(f"SPY {target_date}: 100 synthetic paths vs truth")
    ax.set_xlabel("Minute of day")
    ax.set_ylabel("Price")
    ax.legend()
    
    # save
    import os
    os.makedirs("../figures", exist_ok=True)
    plt.savefig("../figures/01_section2_smoke_test.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()