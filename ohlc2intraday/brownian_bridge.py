"""
Basic Brownian bridge implementation (Riedel 2021, Section 2).

Generate intraday paths conditioned on:
- open price O
- close price C

This is the baseline: no high/low information used.

Convention (matches path_sampler.py): sigma2 is the TOTAL variance
of the bridge over the unit interval [0, 1].
"""
import numpy as np


def brownian_bridge_path(
    open_price: float,
    close_price: float,
    n_steps: int = 390,
    sigma2: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate one Brownian-bridge intraday path from open to close.

    Parameters
    ----------
    open_price : float
        Open price (= start of the day, B(0)).
    close_price : float
        Close price (= end of the day, B(1)).
    n_steps : int
        Number of intraday steps (390 = 1-min bars).
    sigma2 : float, optional
        TOTAL variance over [0, 1]. If None, estimated from (C - O)^2.
    rng : np.random.Generator, optional

    Returns
    -------
    path : np.ndarray of length n_steps + 1, with path[0]=open, path[-1]=close.
    """
    if rng is None:
        rng = np.random.default_rng()

    if sigma2 is None:
        sigma2 = max((close_price - open_price) ** 2, 1e-16)

    ts = np.linspace(0.0, 1.0, n_steps + 1)

    # standard Brownian motion with total variance sigma2 over [0,1]
    dt = np.diff(ts)
    dW = rng.normal(0.0, np.sqrt(sigma2 * dt))
    W = np.concatenate([[0.0], np.cumsum(dW)])

    # Brownian bridge pinned to 0 at both ends, then shifted to O -> C
    bridge = W - ts * W[-1]
    path = open_price + bridge + ts * (close_price - open_price)
    return path


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    path = brownian_bridge_path(
        open_price=430.0,
        close_price=431.5,
        n_steps=390,
        sigma2=1.0,
        rng=rng,
    )
    print(f"Path length: {len(path)}")
    print(f"Start: {path[0]:.2f}, End: {path[-1]:.2f}")
    print(f"Min: {path.min():.2f}, Max: {path.max():.2f}")