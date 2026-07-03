# notebooks/08_oco_step12_count.py
# ステップ1-2: OCO先着順応用の実験成立を確認
# 各kで「H・L両方がライン越えた日」を数える
import sys
sys.path.insert(0, '..')

import numpy as np
import pandas as pd

DATA_PATH = "../data/SPX_full_1min.txt"
N_DAYS = 250  # nb07と同じ最後の250日
K_VALUES = [0.5, 1.0, 1.5, 2.0]

def load_firstrate(path):
    df = pd.read_csv(
        path, header=None,
        names=["datetime","open","high","low","close"],
        usecols=[0,1,2,3,4], parse_dates=["datetime"],
    )
    return df.set_index("datetime").sort_index()

def eval_day_oco(day_df, k_values):
    """1日について、各kで H・L両方がライン越えたか判定"""
    prices = day_df["close"].values.astype(float)
    if len(prices) < 100: return None
    
    O = float(day_df["open"].iloc[0])
    H = float(day_df["high"].max())
    L = float(day_df["low"].min())
    C = float(day_df["close"].iloc[-1])
    if not (H >= max(O,C) and L <= min(O,C) and O > 0 and L > 0):
        return None
    
    # log空間
    lo = np.log(O)
    lH, lL, lC = np.log(H)-lo, np.log(L)-lo, np.log(C)-lo
    
    # σ² は GK で推定
    gk = 0.5*(lH-lL)**2 - (2*np.log(2)-1)*lC**2
    sig2 = max(gk, (2*lH-lC)**2/3.0, 1e-16)
    sigma = np.sqrt(sig2)  # 日次σ
    
    result = {"O": O, "H": H, "L": L, "C": C, "sigma": sigma}
    
    # 各kについて、ストップと利確を設定
    # ストップ = 始値 - k*σ (log空間で -k*σ、価格空間で O*exp(-k*σ))
    for k in k_values:
        stop_log = -k * sigma   # 始値からlog差分
        take_log = +k * sigma
        
        # H・L両方が越えたか
        stop_hit = lL <= stop_log  # 安値がストップを下回った
        take_hit = lH >= take_log  # 高値が利確を上回った
        both_hit = stop_hit and take_hit
        
        result[f"k{k}_stop_hit"] = stop_hit
        result[f"k{k}_take_hit"] = take_hit
        result[f"k{k}_both_hit"] = both_hit
    
    return result

def main():
    df = load_firstrate(DATA_PATH)
    print(f"loaded: {len(df):,} rows  {df.index.min()} -> {df.index.max()}\n")
    
    dates = sorted(set(df.index.date))[-N_DAYS:]
    rows = []
    for d in dates:
        day_df = df[df.index.date == d]
        r = eval_day_oco(day_df, K_VALUES)
        if r is not None:
            r["date"] = d
            rows.append(r)
    
    res = pd.DataFrame(rows)
    print(f"評価日数: {len(res)}\n")
    
    print("="*60)
    print("kごとの対象日数(H・L両方がライン越えた日)")
    print("="*60)
    print(f"{'k':>6} {'ストップ':>10} {'利確':>10} {'両方':>10} {'両方%':>10}")
    print("-"*50)
    for k in K_VALUES:
        stop = res[f"k{k}_stop_hit"].sum()
        take = res[f"k{k}_take_hit"].sum()
        both = res[f"k{k}_both_hit"].sum()
        pct = both / len(res) * 100
        print(f"{k:>6.1f} {stop:>10d} {take:>10d} {both:>10d} {pct:>9.1f}%")
    
    print("\n判定:")
    print("  両方の日数が100以上あれば実験成立(キャリブレーションに十分)")
    print("  50-100 なら要検討(k を緩めるか期間延ばす)")
    print("  50未満 なら k の設計をやり直す")
    
    # CSV保存
    import os; os.makedirs("../results", exist_ok=True)
    res.to_csv("../results/oco_step12_count.csv", index=False)
    print(f"\nSaved: ../results/oco_step12_count.csv")

if __name__ == "__main__":
    main()