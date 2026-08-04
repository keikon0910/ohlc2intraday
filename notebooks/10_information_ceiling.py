# notebooks/10_information_ceiling.py
# 情報の天井測定: 各指標について、OHLC由来の特徴量だけで
# どこまで当てられるかを交差検証で測る。
# 得られる CV R² / Spearman が「OHLC-only手法の到達可能な上限(の下からの近似)」。
# サンプラーは一切使わない(τ順序の問題と独立)。
#
# 設計メモ:
# - 時系列連続ブロックCV + embargo(境界±20日を学習から除外)でリーク防止
# - スケール量(RV/MaxDD/RS±)は log で回帰、報告は Spearman 併記(地図と直接比較可)
# - 特徴量は情報セット別に段階評価(①当日OHLCのみ / ②+隣接日)
#   → 段差が「情報の限界効用」としてそのまま読める
# - ネガコン: ターゲットをシャッフルして R²≈0 を確認(リーク検出器)
# - 天井ρには年ブロック・ブートストラップの90%CIを付す
# - 歪度はレベル回帰に加えて符号分類のAUCも測る(感度の保険)
#
# 解釈上の注意(必読):
# - ここで出る天井は「観測した特徴量での天井」であり、OHLCの真の情報量の
#   天井そのものではない。ρ≈0 は情報限界の強い証拠だが証明ではない。
# - 天井は下からの近似なので「到達ρ > 天井ρ」が出ても矛盾ではなく
#   「回帰側が天井に届いていない」だけ。
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (HistGradientBoostingRegressor,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

DATA_PATH = "../data/SPX_full_1min.txt"
MIN_BARS  = 350      # 半日立会を除外(nの違いが交絡するため)
N_FOLDS   = 5
EMBARGO   = 20       # CVブロック境界の±この日数を学習から除外
N_BOOT    = 300      # ブートストラップ反復
SEED      = 0
LN2       = np.log(2)


# ────────────────────────────────────────────────
# 1日ごとに「真の指標」と「OHLC特徴量」を作る
# ────────────────────────────────────────────────

def day_record(g, d):
    p = g["close"].values.astype(float)
    if len(p) < MIN_BARS:
        return None
    O = float(g["open"].iloc[0]); H = float(g["high"].max())
    L = float(g["low"].min());    C = float(g["close"].iloc[-1])
    if not (H >= max(O, C) and L <= min(O, C) and O > 0 and L > 0):
        return None

    lo = np.log(O)
    lH, lL, lC = np.log(H)-lo, np.log(L)-lo, np.log(C)-lo
    rng = lH - lL
    if rng <= 0:
        return None

    # ── 真の指標(1分足からのみ計算。特徴量には絶対に入れない)──
    # 実測経路の先頭に始値(=0)を追加する(07a/07bと同じ修正、2026-07-25)。
    # 従来はバー終値のみで、始値→1分目のリターンが真値から欠けていた。
    path = np.concatenate([[0.0], np.log(p) - lo])
    r    = np.diff(path)
    if r.std() == 0:
        return None
    rv    = float(np.sum(r**2))
    rsp   = float(np.sum(r[r > 0]**2))
    rsm   = float(np.sum(r[r < 0]**2))
    runmax = np.maximum.accumulate(path)
    maxdd = float(np.max(runmax - path))
    skew  = float(np.sum(r**3) / (np.sum(r**2)**1.5))

    # ── OHLC特徴量(日次OHLCだけから作れるもの)──
    gk = max(0.5*rng**2 - (2*LN2-1)*lC**2, (2*lH-lC)**2/3.0, 1e-16)
    return dict(
        date=d, C_abs=C,
        # targets
        t_rv=rv, t_maxdd=maxdd, t_rsp=rsp, t_rsm=rsm,
        t_asym=(rsm/rsp if rsp > 1e-20 else np.nan), t_skew=skew,
        # features: 当日OHLC(セット①)
        f_lH=lH, f_lL=lL, f_lC=lC, f_rng=rng,
        f_cpos=(lC-lL)/rng,
        f_distratio=lH/rng,                       # (H-O)/(H-L) のlog版
        f_up=lH, f_down=-lL,
        # 比は対数化+クリップ(lL≈0の日の爆発でRidgeが壊れるのを防ぐ)
        f_updown=float(np.clip(np.log(max(lH, 1e-12)/max(-lL, 1e-12)), -6, 6)),
        f_gk=gk, f_park=rng**2/(4*LN2),
        f_rs=lH*(lH-lC) + lL*(lL-lC),             # Rogers-Satchell
        f_absC=abs(lC), f_signC=float(np.sign(lC)),
        f_dow=float(pd.Timestamp(d).dayofweek),
    )


def build_table(df):
    df = df.copy()
    df["date"] = df.index.date
    recs = [r for d, g in df.groupby("date")
            if (r := day_record(g, d)) is not None]
    t = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)

    # ── ラグ特徴量(前日までの情報のみ。セット②で追加)──
    # 前日終値→当日始値のギャップ: O_t を C_abs と lC から復元
    O_abs = t["C_abs"] / np.exp(t["f_lC"])
    t["g_gap"]      = np.log(O_abs) - np.log(t["C_abs"].shift(1))
    t["g_prev_rng"] = t["f_rng"].shift(1)
    t["g_prev_gk"]  = t["f_gk"].shift(1)
    t["g_prev_lC"]  = t["f_lC"].shift(1)
    t["g_gk_ma5"]   = t["f_gk"].shift(1).rolling(5).mean()
    t["g_gk_ma20"]  = t["f_gk"].shift(1).rolling(20).mean()
    t["g_rng_ma20"] = t["f_rng"].shift(1).rolling(20).mean()

    return t.dropna().reset_index(drop=True)


# ────────────────────────────────────────────────
# 連続ブロック交差検証(embargo付き)
# ────────────────────────────────────────────────

def blocked_cv_predict(X, y, model_fn, n_folds=N_FOLDS, embargo=EMBARGO,
                       classifier=False):
    n = len(y)
    edges = np.linspace(0, n, n_folds+1).astype(int)
    pred  = np.full(n, np.nan)
    for k in range(n_folds):
        lo_, hi_ = edges[k], edges[k+1]
        te = np.zeros(n, bool); te[lo_:hi_] = True
        tr = ~te
        # embargo: テストブロックの前後 embargo 日を学習から外す
        tr[max(lo_-embargo, 0):lo_] = False
        tr[hi_:min(hi_+embargo, n)] = False
        m = model_fn()
        m.fit(X[tr], y[tr])
        if classifier:
            pred[te] = m.predict_proba(X[te])[:, 1]
        else:
            pred[te] = m.predict(X[te])
    return pred


def gbdt():
    return HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.06, max_depth=4,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)

def gbdt_clf():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.06, max_depth=4,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEED)

def ridge():
    # 標準化を挟む(gk~1e-4 と dow~0-4 のスケール差でRidgeが歪むのを防ぐ)
    return make_pipeline(StandardScaler(),
                         RidgeCV(alphas=np.logspace(-4, 3, 20)))


def r2(y, p):
    return 1.0 - np.sum((y-p)**2) / np.sum((y-y.mean())**2)


def spear(a, b):
    return spearmanr(a, b)[0]


def boot_ci_spearman(y_raw, pred, blocks, n_boot=N_BOOT, seed=SEED):
    """四半期ブロック・ブートストラップで Spearman ρ の90%CI"""
    rs = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    stats = []
    for _ in range(n_boot):
        pick = rs.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(blocks == yv)[0] for yv in pick])
        if len(np.unique(y_raw[idx])) < 3:
            continue
        st = spear(y_raw[idx], pred[idx])
        if np.isfinite(st):
            stats.append(st)
    if len(stats) < 20:   # 予測が定数など、CIが計算不能な場合
        return np.nan, np.nan
    lo_, hi_ = np.percentile(stats, [5, 95])
    return lo_, hi_


# ────────────────────────────────────────────────
# 評価本体
# ────────────────────────────────────────────────

def evaluate(t, target, log_target, feats, label, setname, blocks,
             pred_store=None):
    y_raw = t[target].values.astype(float)
    ok = np.isfinite(y_raw)
    if log_target:
        ok &= (y_raw > 0)
    X  = t.loc[ok, feats].values.astype(float)
    dates_ok = t.loc[ok, "date"].values
    yr = blocks[ok]
    y_raw = y_raw[ok]
    y = np.log(y_raw) if log_target else y_raw

    # GBDT / Ridge
    p_g = blocked_cv_predict(X, y, gbdt)
    p_r = blocked_cv_predict(X, y, ridge)
    r2_g, sp_g = r2(y, p_g), spear(y_raw, p_g)
    r2_r, sp_r = r2(y, p_r), spear(y_raw, p_r)
    ci_lo, ci_hi = boot_ci_spearman(y_raw, p_g, yr)
    if pred_store is not None:
        # 予測はlogスケールのままでよい(スピアマンは単調変換に不変)
        pred_store[target] = pd.DataFrame(
            {"date": dates_ok, f"true_{target}": y_raw,
             f"pred_{target}": p_g})

    # ネガコン: 同一のシャッフルで学習と評価(リーク検出器)
    rs = np.random.default_rng(SEED)
    y_perm = y[rs.permutation(len(y))]
    p_null = blocked_cv_predict(X, y_perm, gbdt)
    r2_null = r2(y_perm, p_null)

    print(f"  {label:14s} n={len(y):5d} | "
          f"GBDT R²={r2_g:6.3f} ρ={sp_g:6.3f} CI[{ci_lo:5.3f},{ci_hi:5.3f}] | "
          f"Ridge R²={r2_r:6.3f} ρ={sp_r:6.3f} | "
          f"ネガコン R²={r2_null:+.3f}")
    return dict(set=setname, target=label, n=len(y),
                r2_gbdt=r2_g, sp_gbdt=sp_g, sp_lo=ci_lo, sp_hi=ci_hi,
                r2_ridge=r2_r, sp_ridge=sp_r, r2_null=r2_null)


def evaluate_skew_sign(t, feats, setname, blocks):
    """歪度の符号分類(レベル回帰が全滅でも弱い信号を拾う保険)"""
    y_raw = t["t_skew"].values.astype(float)
    ok = np.isfinite(y_raw)
    X  = t.loc[ok, feats].values.astype(float)
    y  = (y_raw[ok] > 0).astype(int)
    if len(np.unique(y)) < 2:
        return None
    p = blocked_cv_predict(X, y, gbdt_clf, classifier=True)
    auc = roc_auc_score(y, p)
    print(f"  {'歪度の符号(AUC)':14s} n={len(y):5d} | AUC={auc:.3f} "
          f"(0.5=情報なし)")
    return dict(set=setname, target="歪度符号AUC", n=len(y),
                r2_gbdt=np.nan, sp_gbdt=auc, sp_lo=np.nan, sp_hi=np.nan,
                r2_ridge=np.nan, sp_ridge=np.nan, r2_null=np.nan)


def main():
    df = pd.read_csv(DATA_PATH, header=None,
        names=["datetime","open","high","low","close"],
        usecols=[0,1,2,3,4], parse_dates=["datetime"])
    df = df.set_index("datetime").sort_index()
    print(f"loaded: {len(df):,} rows  {df.index.min()} -> {df.index.max()}")

    t = build_table(df)
    blocks = pd.PeriodIndex(pd.to_datetime(t["date"]), freq="Q").astype(str).values

    set1 = [c for c in t.columns if c.startswith("f_")]   # 当日OHLCのみ
    set2 = set1 + [c for c in t.columns if c.startswith("g_")]  # +隣接日
    print(f"評価日数: {len(t)}  期間: {t['date'].iloc[0]} 〜 {t['date'].iloc[-1]}")
    print(f"セット①(当日OHLC): {len(set1)}個 / セット②(+隣接日): {len(set2)}個")
    print(f"CV: 連続{N_FOLDS}ブロック + embargo±{EMBARGO}日\n")

    specs = [
        ("t_rv",    True,  "RV(log)"),
        ("t_maxdd", True,  "MaxDD(log)"),
        ("t_rsp",   True,  "RS+(log)"),
        ("t_rsm",   True,  "RS-(log)"),
        ("t_asym",  True,  "RS-/RS+(log)"),
        ("t_skew",  False, "実現歪度"),
    ]

    res = []
    pred_store = {}
    for setname, feats in [("①当日OHLCのみ", set1), ("②+隣接日", set2)]:
        print("="*104)
        print(f"情報セット {setname}({len(feats)}特徴量)")
        print("="*104)
        store = pred_store if setname == "②+隣接日" else None
        for tg, lg, lb in specs:
            res.append(evaluate(t, tg, lg, feats, lb, setname, blocks,
                                pred_store=store))
        sk = evaluate_skew_sign(t, feats, setname, blocks)
        if sk is not None:
            res.append(sk)
        print()

    # ── 地図との比較(セット②のGBDT ρを天井として)──
    print("="*104)
    print("地図の到達点との比較(天井=セット②GBDT ρ、到達=GK版250日スピアマン)")
    print("  注: 天井は下からの近似。到達>天井は「回帰が天井に届いていない」の意")
    print("="*104)
    # 表示用の参考値(始値修正後のGK版250日)。正式な同一期間比較は11番で行う
    achieved = {"RV(log)": 0.880, "MaxDD(log)": 0.939,
                "RS+(log)": 0.875, "RS-(log)": 0.867}
    print(f"{'指標':14s} {'天井ρ':>8} {'90%CI':>16} {'到達ρ':>8} {'伸びしろ':>10}")
    for r_ in res:
        if r_["set"] != "②+隣接日" or r_["target"] == "歪度符号AUC":
            continue
        a = achieved.get(r_["target"])
        ci = f"[{r_['sp_lo']:.3f},{r_['sp_hi']:.3f}]"
        if a is None:
            print(f"{r_['target']:14s} {r_['sp_gbdt']:8.3f} {ci:>16} {'—':>8} {'—':>10}")
        else:
            print(f"{r_['target']:14s} {r_['sp_gbdt']:8.3f} {ci:>16} {a:8.3f} "
                  f"{r_['sp_gbdt']-a:+10.3f}")

    import os; os.makedirs("../results", exist_ok=True)
    pd.DataFrame(res).to_csv("../results/information_ceiling.csv", index=False)
    # セット②のCV予測を日次で保存(11番が地図と同一期間比較するために使う)
    merged = None
    for tg, dfp in pred_store.items():
        dfp = dfp.copy(); dfp["date"] = dfp["date"].astype(str)
        merged = dfp if merged is None else merged.merge(dfp, on="date", how="outer")
    merged.to_csv("../results/ceiling_predictions_set2.csv", index=False)
    print("Saved: ../results/ceiling_predictions_set2.csv")
    print("\nSaved: ../results/information_ceiling.csv")


if __name__ == "__main__":
    main()