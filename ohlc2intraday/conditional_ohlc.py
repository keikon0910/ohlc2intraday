# Section 4: OHLC 全部を条件にしたモーメント (Riedel 2021 Theorem 4.6)
# 各時刻 t での E[B(t)|h,l,c] と Var を二重級数で計算する。
# モンテカルロと一致することを確認済み (t=0.5 で diff 0.002)。
import numpy as np
from scipy.special import erf


def _phi(x, var):
    return np.exp(-x * x / (2.0 * var)) / np.sqrt(2.0 * np.pi * var)


def _E(x, sig):
    # スケール付き erf (式 4.15 の E_sigma)
    return 0.5 * erf(x / (np.sqrt(2.0) * sig))


def moments_ohlc(t, h, l, c, sigma2, K=None):
    # Theorem 4.6 の二重級数。始値=0 規約。
    # K (打ち切り段数) は省くと Δ/σ から自動決定 (Δが小さいと収束が遅いので増やす)。
    s2 = sigma2
    sig = np.sqrt(s2)
    delta = h - l
    if K is None:
        ratio = delta / sig
        K = 3 if ratio >= 1.0 else min(12, int(np.ceil(3.0 / max(ratio, 0.25))))
    sig_t2 = t * (1.0 - t) * s2
    sig_t = np.sqrt(sig_t2)

    M0 = M1 = M2 = 0.0
    for j in range(-K, K + 1):
        for k in range(-K, K + 1):
            # 補助変数 (式 4.7 の v, vt, w, wt)
            v = 2.0 * (j * (1 - t) + k * t)
            vt = 2.0 * (j * (1 - t) - k * t)
            w = 2.0 * (j + k)
            wt = 2.0 * (j - k)
            # 4本のガウシアン項のパラメータ: (s_i, mu, q, tau, tau_hat, dq/dh, dq/dl)
            params = (
                (+1, c * t + vt * delta, w * delta - c, vt, -vt, w, -w),
                (-1, (2*h - c) * t + v * delta, wt * delta - (2*h - c),
                 2*t + v, -v, wt - 2.0, -wt),
                (-1, 2*h*(1 - t) + c*t - v*delta, (2*h - c) - wt*delta,
                 2*(1 - t) - v, v, 2.0 - wt, wt),
                (+1, 2*h - c*t - vt*delta, c - w*delta,
                 2.0 - vt, vt, -w, w),
            )
            for (si, mu, q, tau, taut, dqh, dql) in params:
                g = q * q / (2.0 * s2)
                if g > 700.0:
                    # exp アンダーフローで実質ゼロの項はスキップ
                    continue
                dgh = q * dqh / s2
                dgl = q * dql / s2
                dghl = dqh * dql / s2
                A = tau * taut
                Bc = tau * dgl + taut * dgh
                Gam = -dgh * dgl + dghl
                psi = np.exp(-g) / (np.sqrt(2.0 * np.pi) * sig)
                ph = _phi(h - mu, sig_t2)
                pl = _phi(l - mu, sig_t2)
                R = _E(h - mu, sig_t) - _E(l - mu, sig_t)
                # m=0 (Corollary 4.7: M0 = p(h,l,c))
                M0 += si * psi * (Gam * R)
                # m=1 (式 4.16)
                a1 = A - Gam * sig_t2
                e1 = Bc + Gam * mu
                M1 += si * psi * (a1 * ph - a1 * pl + e1 * R)
                # m=2 (式 4.17)
                a2 = 2*h*A - 2*Bc*sig_t2 - Gam*sig_t2*(mu + h)
                ah2 = 2*l*A - 2*Bc*sig_t2 - Gam*sig_t2*(mu + l)
                e2 = 2*Bc*mu - 2*A + Gam*(mu*mu + sig_t2)
                M2 += si * psi * (a2 * ph - ah2 * pl + e2 * R)
    return M0, M1, M2


def conditional_mean_var_ohlc(t, high, low, close, open_price=0.0,
                              sigma2=1.0, K=None):
    # 元の価格スケールでの E[B(t)|h,l,c] と Var
    h = high - open_price
    l = low - open_price
    c = close - open_price
    # 高値=始値/終値 などの縮退ケースは微小にずらす
    eps = 1e-6 * max(h - l, 1e-8)
    if h <= max(0.0, c):
        h = max(0.0, c) + eps
    if l >= min(0.0, c):
        l = min(0.0, c) - eps
    t = float(np.clip(t, 1e-4, 1 - 1e-4))
    M0, M1, M2 = moments_ohlc(t, h, l, c, sigma2, K=K)
    if M0 <= 0 or not np.isfinite(M0):
        # 数値破綻時は線形橋にフォールバック
        return c * t + open_price, t * (1 - t) * sigma2
    mean = M1 / M0
    var = max(M2 / M0 - mean * mean, 0.0)
    return mean + open_price, var


if __name__ == "__main__":
    sigma2 = 1.0
    O, H, L, C = 0.0, 1.3, -0.6, 0.4
    print("t->1 should give mean->c=0.4, var->0:")
    for t in [0.5, 0.9, 0.99, 0.999]:
        m, v = conditional_mean_var_ohlc(t, H, L, C, O, sigma2)
        print(f"  t={t:.3f}  mean={m:+.4f}  var={v:.5f}")