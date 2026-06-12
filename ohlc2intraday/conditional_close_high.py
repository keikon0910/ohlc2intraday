"""
Brownian bridge conditioned on Close and High prices.
Based on Riedel (2021) Section 3, Theorem 3.4.

Convention: shift so open is at 0.
  c = Close - Open,  h = High - Open  (requires h >= max(0, c))
"""
import numpy as np
from scipy.special import erf


def _phi(x, var):
    """Normal density N(0, var) at x."""
    return np.exp(-x * x / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def _moments_high_close(t, h, c, sigma2):
    """
    (M0, M1, M2) from Riedel (2021) Theorem 3.4, eqs (3.12)-(3.14).
    Uses open=0 convention. Valid for t in (0, 1), h >= max(0, c).
    """
    sig_t2 = t * (1.0 - t) * sigma2
    sig_t = np.sqrt(sig_t2)
    r = 2.0 * h - c
    s2 = sigma2

    M0 = (2.0 * (2.0 * h - c) / s2) * _phi(2.0 * h - c, s2)

    e1 = erf((c * t - h) / (np.sqrt(2.0) * sig_t))
    e2 = erf((h - r * t) / (np.sqrt(2.0) * sig_t))

    p_hrt = (1.0 - 2.0 * t) + 2.0 * r * (r * t - h) / s2
    M1 = (
        _phi(c, s2) * (1.0 + e1)
        + _phi(r, s2) * (2.0 * h * r / s2 - 1.0 + p_hrt * e2)
        - 4.0 * r * t * (1.0 - t) * _phi(r, s2) * _phi(h - r * t, sig_t2)
    )

    q1 = -(r * t**2 + (1.0 - t) * (2.0 * h - r * t)) + r * (h**2 + (h - r * t)**2) / s2
    q2 = (2.0 * h * (1.0 - t) - r * t) + 2.0 * h * r * (r * t - h) / s2
    M2 = (
        2.0 * (2.0 * h - c * t) * _phi(c, s2) * (1.0 + e1)
        + 2.0 * _phi(r, s2) * (
            (r * t * (1.0 - t) + q1 + q2 * e2)
            - 4.0 * r * h * t * (1.0 - t) * _phi(h - r * t, sig_t2)
        )
    )
    return M0, M1, M2


def conditional_mean_var(t, high, close, open_price=0.0, sigma2=1.0):
    """
    E[B(t) | High, Close] and Var[B(t) | High, Close]
    in original price coordinates.
    """
    h = high - open_price
    c = close - open_price
    M0, M1, M2 = _moments_high_close(t, h, c, sigma2)
    mean_shifted = M1 / M0
    var = M2 / M0 - mean_shifted**2
    return mean_shifted + open_price, var


if __name__ == "__main__":
    sigma2 = 1.0
    O, H, C = 0.0, 1.3, 0.4
    print("t->1 should give mean->close=0.4, var->0:")
    for t in [0.5, 0.9, 0.99, 0.999]:
        m, v = conditional_mean_var(t, H, C, O, sigma2)
        print(f"  t={t:.3f}  mean={m:+.4f}  var={v:.5f}")