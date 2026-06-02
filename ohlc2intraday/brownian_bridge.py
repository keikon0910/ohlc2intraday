"""
Basic Brownian bridge implementation (Riedel 2021, Section 2).

Generate intraday paths conditioned on:
- open price O
- close price C

This is the baseline: no high/low information used.
"""
import numpy as np


def brownian_bridge_path(
    open_price: float,
    close_price: float,
    n_steps: int = 390,
    sigma: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate one intraday path from open to close.
    
    Parameters
    ----------
    open_price : float
        Open price (= start of the day)
    close_price : float
        Close price (= end of the day)
    n_steps : int
        Number of intraday steps (390 = 1-min bars in regular session)
    sigma : float, optional
        Volatility per unit time. If None, estimated from |C - O|.
    rng : np.random.Generator, optional
        Random number generator.
    
    Returns
    -------
    path : np.ndarray
        Length n_steps + 1, starting at open and ending at close.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if sigma is None:
        # rough estimate from the day's drift
        sigma = abs(close_price - open_price) / np.sqrt(n_steps)
    
    # time grid 0, 1/n, 2/n, ..., 1
    t = np.linspace(0, 1, n_steps + 1)
    
    # standard Brownian path
    dW = rng.normal(0, np.sqrt(1.0 / n_steps), size=n_steps)
    W = np.concatenate([[0], np.cumsum(dW)])
    
    # Brownian bridge: B(t) = W(t) - t * W(1)
    # then scale and shift to start at O, end at C
    bridge = W - t * W[-1]
    path = open_price + sigma * bridge * np.sqrt(n_steps) + t * (close_price - open_price)
    
    return path


if __name__ == "__main__":
    # quick smoke test
    rng = np.random.default_rng(42)
    path = brownian_bridge_path(
        open_price=430.0,
        close_price=431.5,
        n_steps=390,
        rng=rng,
    )
    print(f"Path length: {len(path)}")
    print(f"Start: {path[0]:.2f}, End: {path[-1]:.2f}")
    print(f"Min: {path.min():.2f}, Max: {path.max():.2f}")