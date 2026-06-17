# Section 2: 終値だけを条件にしたブラウン橋 (Riedel 2021 Sec.2)
# 始値->終値をつなぐ最もシンプルな経路生成。高値・安値は使わない。
# sigma2 は1日全体の分散 (path_sampler 等と共通の規約)。
import numpy as np


def brownian_bridge_path(
    open_price: float,
    close_price: float,
    n_steps: int = 390,
    sigma2: float = None,
    rng: np.random.Generator = None,
) -> np.ndarray:
    # 始値から終値へのブラウン橋を1本生成。長さ n_steps+1。
    # sigma2 を省くと (終値-始値)^2 で代用する。
    if rng is None:
        rng = np.random.default_rng()
    if sigma2 is None:
        sigma2 = max((close_price - open_price) ** 2, 1e-16)

    ts = np.linspace(0.0, 1.0, n_steps + 1)
    # 全分散 sigma2 のブラウン運動を作り、両端を0に固定してから始値->終値へずらす
    dt = np.diff(ts)
    dW = rng.normal(0.0, np.sqrt(sigma2 * dt))
    W = np.concatenate([[0.0], np.cumsum(dW)])
    bridge = W - ts * W[-1]
    return open_price + bridge + ts * (close_price - open_price)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    path = brownian_bridge_path(430.0, 431.5, n_steps=390, sigma2=1.0, rng=rng)
    print(f"Path length: {len(path)}")
    print(f"Start: {path[0]:.2f}, End: {path[-1]:.2f}")
    print(f"Min: {path.min():.2f}, Max: {path.max():.2f}")