"""
Exact intraday path sampler conditioned on Open, High, Close.
Based on the Bessel(3)-bridge decomposition at the path maximum
(Williams decomposition), consistent with Riedel (2021) Section 3.

Validated against the closed-form moments of Theorem 3.4
(see conditional_close_high.py).
"""
import numpy as np


def _std_brownian_bridge(ts, T, sigma2, rng):
    """Brownian bridge 0->0 on [0,T] sampled at times ts (incl. 0 and T)."""
    dt = np.diff(ts)
    incr = rng.normal(0, np.sqrt(sigma2 * dt))
    W = np.concatenate([[0.0], np.cumsum(incr)])
    return W - (ts / T) * W[-1]


def _bes3_bridge(a, b, ts, T, sigma2, rng):
    """Exact Bessel(3) bridge from a to b on [0,T] at times ts."""
    lin = a * (1 - ts / T) + b * (ts / T)
    w1 = _std_brownian_bridge(ts, T, sigma2, rng)
    w2 = _std_brownian_bridge(ts, T, sigma2, rng)
    w3 = _std_brownian_bridge(ts, T, sigma2, rng)
    return np.sqrt((lin + w1) ** 2 + w2 ** 2 + w3 ** 2)


def _sample_argmax_time(c, h, sigma2, rng, grid=4000):
    """Sample the time at which the maximum h is attained."""
    eps = 1e-9
    taus = np.linspace(eps, 1 - eps, grid)
    a = max(h, 1e-9)
    b = max(h - c, 1e-9)
    logf = (np.log(a) - 1.5 * np.log(taus) - a * a / (2 * sigma2 * taus)
            + np.log(b) - 1.5 * np.log(1 - taus) - b * b / (2 * sigma2 * (1 - taus)))
    logf -= logf.max()
    f = np.exp(logf)
    f /= f.sum()
    return rng.choice(taus, p=f)


def generate_path_ohc(open_price, high, close, n_steps=390,
                      sigma2=None, rng=None):
    """
    Generate one intraday path with B(0)=open, B(1)=close, max=high.

    Parameters
    ----------
    open_price, high, close : float
        Daily values. Requires high >= max(open_price, close).
    n_steps : int
        Number of intraday steps (390 = 1-min bars).
    sigma2 : float, optional
        Total variance over the day. If None, a rough estimate is used.
    rng : np.random.Generator, optional

    Returns
    -------
    path : np.ndarray of length n_steps + 1
    """
    if rng is None:
        rng = np.random.default_rng()
    c = close - open_price
    h = high - open_price
    if h < max(0.0, c):
        raise ValueError("high must be >= max(open, close)")
    if sigma2 is None:
        sigma2 = max(h ** 2 + (h - c) ** 2, 1e-12)

    tau = _sample_argmax_time(c, h, sigma2, rng)
    ts_all = np.linspace(0, 1, n_steps + 1)
    left = ts_all <= tau
    ts_L, ts_R = ts_all[left], ts_all[~left]

    D_L = _bes3_bridge(h, 0.0, ts_L, tau, sigma2, rng) if len(ts_L) > 1 else np.array([h])
    if len(ts_R) > 0:
        D_R = _bes3_bridge(0.0, h - c, ts_R - tau, 1 - tau, sigma2, rng)
    else:
        D_R = np.array([])

    path = h - np.concatenate([D_L, D_R])
    path[0] = 0.0
    path[-1] = c
    return path + open_price


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    O, H, C, sigma2 = 0.0, 1.3, 0.4, 1.0
    n_paths, n_steps = 3000, 390

    paths = np.array([
        generate_path_ohc(O, H, C, n_steps, sigma2, rng) for _ in range(n_paths)
    ])
    maxima = paths.max(axis=1)
    ret_std = np.diff(paths, axis=1).std(axis=1)
    dt = 1.0 / n_steps
    print(f"{n_paths} paths:")
    print(f"  end == close: max|err| = {np.abs(paths[:, -1] - C).max():.2e}")
    print(f"  max <= high: {(maxima <= H + 1e-9).mean() * 100:.1f}%  "
          f"mean max = {maxima.mean():.4f} (H={H})")
    print(f"  1-step return std: {ret_std.mean():.5f} "
          f"(theory ~ {np.sqrt(sigma2 * dt):.5f})")

    # cross-validation against Theorem 3.4 (now in a separate module)
    from conditional_close_high import conditional_mean_var
    print("\n t     empirical mean   Theorem3.4   diff")
    for t_check in [0.1, 0.25, 0.5, 0.75, 0.9]:
        i = int(round(t_check * n_steps))
        emp = paths[:, i].mean()
        th, _ = conditional_mean_var(t_check, H, C, O, sigma2)
        print(f" {t_check:.2f}   {emp:+.4f}        {th:+.4f}    {abs(emp - th):.4f}")