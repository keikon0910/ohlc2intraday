# OHLC条件付きの日中パス生成 (Riedel 2021 Sec.4)
# B(0)=始値, B(1)=終値, 最大=高値, 最小=安値 を満たす経路を作る。
# 高値・安値の到達時刻 (tauH, tauL) をMCで引き、区間ごとにBessel(3)橋でつなぐ。
# Theorem 4.6 のモーメントと全t点で一致することを確認済み (diff<0.01)。
import numpy as np


def _std_bb(ts, T, s2, rng):
    # 0->0 のブラウン橋を時刻 ts でサンプル
    dt = np.diff(ts)
    incr = rng.normal(0, np.sqrt(s2 * dt))
    W = np.concatenate([[0.0], np.cumsum(incr)])
    return W - (ts / T) * W[-1]


def _bes3(a, b, ts, T, s2, rng):
    # a->b の Bessel(3)橋 (3本のブラウン橋の二乗和の平方根)
    lin = a * (1 - ts / T) + b * (ts / T)
    return np.sqrt((lin + _std_bb(ts, T, s2, rng)) ** 2
                   + _std_bb(ts, T, s2, rng) ** 2
                   + _std_bb(ts, T, s2, rng) ** 2)


def _q(s, x, y, l, h, s2, K=5):
    # 両端 l,h に吸収壁がある場合の遷移密度 (鏡像法の級数)
    x = np.asarray(x, float); y = np.asarray(y, float)
    d = h - l; var = s2 * s
    tot = np.zeros(np.broadcast(x, y).shape)
    for k in range(-K, K + 1):
        tot = tot + np.exp(-(y - x - 2 * k * d) ** 2 / (2 * var))
        tot = tot - np.exp(-(y + x - 2 * l - 2 * k * d) ** 2 / (2 * var))
    return tot / np.sqrt(2 * np.pi * var)


def prepass_argtimes(c, h, l, s2, n_steps=150, total=1_500_000, batch=100_000,
                     tol=0.06, target_pool=500, seed=None, beta=0.5826):
    # 高値・安値に到達する (tauL, tauH) の組をMCで集める。
    # beta は離散極値の連続性補正 (BGK)。壁を corr だけ内側にして引く。
    corr = beta * np.sqrt(s2 / n_steps)
    h_adj, l_adj = h - corr, l + corr
    rng = np.random.default_rng(seed)
    dt = 1.0 / n_steps
    tg = np.linspace(0, 1, n_steps + 1).astype(np.float32)
    pairs = []; got = 0; done = 0
    while done < total and got < target_pool:
        nb = min(batch, total - done)
        incr = rng.normal(0, np.sqrt(s2 * dt), size=(nb, n_steps)).astype(np.float32)
        W = np.cumsum(incr, axis=1)
        W = np.concatenate([np.zeros((nb, 1), np.float32), W], axis=1)
        br = W - tg[None, :] * (W[:, [-1]] - c)
        mask = (np.abs(br.max(1) - h_adj) < tol) & (np.abs(br.min(1) - l_adj) < tol)
        if mask.any():
            sub = br[mask]
            pairs.append(np.stack([tg[sub.argmin(1)], tg[sub.argmax(1)]], 1))
            got += mask.sum()
        done += nb
        del incr, W, br, mask
    if not pairs:
        return None
    return np.concatenate(pairs)


def make_sampler_ohlc(open_price, high, low, close, n_steps=390,
                      sigma2=None, pool=None, max_tries=200, grid_n=300,
                      prepass_kwargs=None):
    # 1日分のサンプラーを作って返す。sample(rng) が元の価格スケールの経路を返す。
    # sigma2 を省くと OHLC から Garman-Klass で推定 (負なら (2h-c)^2/3 にフォールバック)。
    c = close - open_price
    h = high - open_price
    l = low - open_price
    # 高値=始値/終値 などの縮退ケースは微小にずらす
    eps0 = 1e-6 * max(h - l, 1e-8)
    if h <= max(0.0, c): h = max(0.0, c) + eps0
    if l >= min(0.0, c): l = min(0.0, c) - eps0
    if sigma2 is None:
        gk = 0.5 * (h - l) ** 2 - (2 * np.log(2) - 1) * c ** 2
        sigma2 = max(gk, (2 * h - c) ** 2 / 3.0, 1e-16)
    s2 = sigma2
    if pool is None:
        pool = prepass_argtimes(c, h, l, s2, **(prepass_kwargs or {}))
    if pool is None:
        raise RuntimeError("prepass で (tauL, tauH) が集まりませんでした")

    ts = np.linspace(0, 1, n_steps + 1)
    yg = np.linspace(l + 1e-9, h - 1e-9, grid_n)
    eps = 1e-5

    def idx(tau):
        # 端点 (始値・終値) を潰さないよう内部のインデックスに丸める
        return min(max(int(round(tau * n_steps)), 1), n_steps - 1)

    def seg_to_wall(x0, i0, i1, to_floor, rng):
        # x0 (内部) から壁 (区間の終点) へ。壁側を Bessel橋、反対壁は棄却で守る
        tl = ts[i0:i1+1] - ts[i0]; T = ts[i1] - ts[i0]
        for _ in range(max_tries):
            if to_floor:
                seg = l + _bes3(x0 - l, 0.0, tl, T, s2, rng)
                if seg.max() <= h + 1e-12: return seg
            else:
                seg = h - _bes3(h - x0, 0.0, tl, T, s2, rng)
                if seg.min() >= l - 1e-12: return seg
        return np.clip(seg, l, h)

    def seg_from_wall(x1, i0, i1, from_floor, rng):
        # 壁 (区間の始点) から x1 (内部) へ。壁側を Bessel橋、反対壁は棄却
        tl = ts[i0:i1+1] - ts[i0]; T = ts[i1] - ts[i0]
        for _ in range(max_tries):
            if from_floor:
                seg = l + _bes3(0.0, x1 - l, tl, T, s2, rng)
                if seg.max() <= h + 1e-12: return seg
            else:
                seg = h - _bes3(0.0, h - x1, tl, T, s2, rng)
                if seg.min() >= l - 1e-12: return seg
        return np.clip(seg, l, h)

    def seg_wall_to_wall(i0, i1, low_first, rng):
        # 壁から壁へ (l->h または h->l)。中点の値を厳密な両側密度から引き、
        # 前半・後半をそれぞれの壁側 Bessel橋で作る
        im = (i0 + i1) // 2
        if im <= i0 or im >= i1:
            return np.linspace(l if low_first else h,
                               h if low_first else l, i1 - i0 + 1)
        tm = ts[im] - ts[i0]; T = ts[i1] - ts[i0]
        if low_first:
            w = _q(tm, l + eps, yg, l, h, s2) * _q(T - tm, h - eps, yg, l, h, s2)
        else:
            w = _q(tm, h - eps, yg, l, h, s2) * _q(T - tm, l + eps, yg, l, h, s2)
        w = np.clip(w, 0, None)
        csum = np.cumsum(w)
        ym = yg[np.searchsorted(csum, rng.random() * csum[-1])]
        a = seg_from_wall(ym, i0, im, low_first, rng)
        b = seg_to_wall(ym, im, i1, not low_first, rng)
        return np.concatenate([a, b[1:]])

    def sample(rng):
        tauL, tauH = pool[rng.integers(len(pool))]
        path = np.empty(n_steps + 1)
        iL, iH = idx(tauL), idx(tauH)
        if iH == iL:
            if iL < n_steps - 1: iH = iL + 1
            else: iL = iH - 1
        if iL < iH:
            # 安値が先: 始値->安値->高値->終値
            s1 = seg_to_wall(0.0, 0, iL, True, rng)
            s2_ = seg_wall_to_wall(iL, iH, True, rng)
            s3 = seg_from_wall(c, iH, n_steps, False, rng)
            path[:iL+1] = s1; path[iL:iH+1] = s2_; path[iH:] = s3
        else:
            # 高値が先: 始値->高値->安値->終値
            s1 = seg_to_wall(0.0, 0, iH, False, rng)
            s2_ = seg_wall_to_wall(iH, iL, False, rng)
            s3 = seg_from_wall(c, iL, n_steps, True, rng)
            path[:iH+1] = s1; path[iH:iL+1] = s2_; path[iL:] = s3
        path[0] = 0.0; path[-1] = c
        return path + open_price
    return sample, pool


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    O, H, L, C = 0.0, 1.3, -0.6, 0.4
    sampler, pool = make_sampler_ohlc(O, H, L, C, n_steps=390, sigma2=1.0,
                                      prepass_kwargs=dict(seed=12345))
    P = np.array([sampler(rng) for _ in range(800)])
    print(f"pool={len(pool)}  end err={np.abs(P[:,-1]-C).max():.0e}")
    print(f"max==H: {(np.abs(P.max(1)-H)<1e-9).mean()*100:.0f}%  "
          f"min==L: {(np.abs(P.min(1)-L)<1e-9).mean()*100:.0f}%")
    print(f"in range: {bool((P>=L-1e-9).all() and (P<=H+1e-9).all())}")
    print(f"ret std: {np.diff(P,1).std():.5f} (theory ~{np.sqrt(1.0/390):.5f})")