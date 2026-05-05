# core/features.py  ─ 피처 생성 공통 모듈
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ret1","ret2","ret3","ret5","ret10","ret20",
    "dist_ma3","dist_ma5","dist_ma10","dist_ma20","dist_ma60","dist_ma120",
    "volume_ratio5","volume_ratio20","trading_value_ratio5","trading_value_ratio20",
    "range_pct","open_gap","upper_shadow","lower_shadow","body_ratio",
    "rsi7","rsi14","rsi21",
    "macd_gap","macd_hist_slope",
    "bb_pos","bb_width",
    "atr_ratio",
    "stoch_k","stoch_d",
    "obv_slope",
    "cci14",
    "dist_52w_high","dist_52w_low","near_52w_high",
    "vol_explosion","price_compress","breakout_flag",
    "consec_up","consec_down",
]


def _rsi(s, period):
    delta = s.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochastic(high, low, close, k=14, d=3):
    lo = low.rolling(k).min()
    hi = high.rolling(k).max()
    sk = 100 * (close - lo) / (hi - lo + 1e-9)
    return sk, sk.rolling(d).mean()


def make_features(df: pd.DataFrame, target_pct: float = 0.15) -> pd.DataFrame:
    if df is None or len(df) < 150:
        return pd.DataFrame()

    d = df.copy()
    c, h, l, o, v, tv = d["close"], d["high"], d["low"], d["open"], d["volume"], d["trading_value"]

    for n in [1,2,3,5,10,20]:
        d[f"ret{n}"] = c.pct_change(n)

    for n in [3,5,10,20,60,120]:
        ma = c.rolling(n).mean()
        d[f"ma{n}"]       = ma
        d[f"dist_ma{n}"]  = c / ma - 1

    for n in [5,20]:
        d[f"vol_ma{n}"]  = v.rolling(n).mean()
        d[f"tv_ma{n}"]   = tv.rolling(n).mean()
    d["volume_ratio5"]          = v / d["vol_ma5"]
    d["volume_ratio20"]         = v / d["vol_ma20"]
    d["trading_value_ratio5"]   = tv / d["tv_ma5"]
    d["trading_value_ratio20"]  = tv / d["tv_ma20"]

    body = (c - o).abs()
    rng  = (h - l).clip(lower=1e-9)
    d["range_pct"]    = rng / c
    d["open_gap"]     = o / c.shift(1) - 1
    d["upper_shadow"] = (h - pd.concat([o,c],axis=1).max(axis=1)) / c
    d["lower_shadow"] = (pd.concat([o,c],axis=1).min(axis=1) - l) / c
    d["body_ratio"]   = body / rng

    d["rsi7"]  = _rsi(c, 7)
    d["rsi14"] = _rsi(c, 14)
    d["rsi21"] = _rsi(c, 21)

    ema12  = c.ewm(span=12, adjust=False).mean()
    ema26  = c.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    sig    = macd.ewm(span=9, adjust=False).mean()
    hist   = macd - sig
    d["macd_gap"]        = hist
    d["macd_hist_slope"] = hist.diff(3)

    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    bb_u   = bb_mid + 2*bb_std
    bb_l   = bb_mid - 2*bb_std
    d["bb_pos"]   = (c - bb_l) / (bb_u - bb_l + 1e-9)
    d["bb_width"] = (bb_u - bb_l) / bb_mid

    tr = pd.concat([(h-l), (h-c.shift(1)).abs(), (l-c.shift(1)).abs()], axis=1).max(axis=1)
    d["atr_ratio"] = tr.rolling(14).mean() / c

    d["stoch_k"], d["stoch_d"] = _stochastic(h, l, c)

    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d["obv_slope"] = obv.diff(5) / (v.rolling(5).mean() + 1e-9)

    tp = (h+l+c)/3
    d["cci14"] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std().replace(0,1e-9))

    h52 = h.rolling(252, min_periods=60).max()
    l52 = l.rolling(252, min_periods=60).min()
    d["dist_52w_high"] = c / h52 - 1
    d["dist_52w_low"]  = c / l52 - 1
    d["near_52w_high"] = (d["dist_52w_high"] > -0.05).astype(int)

    d["vol_explosion"]  = (d["volume_ratio20"] > 3.0).astype(int)
    price_r5 = c.rolling(5).max() - c.rolling(5).min()
    d["price_compress"] = (price_r5 < tr.rolling(14).mean() * 1.0).astype(int)
    d["breakout_flag"]  = ((c.shift(1) < d["ma20"].shift(1)) & (c >= d["ma20"])).astype(int)

    up = (c.diff() > 0).astype(int)
    dn = (c.diff() < 0).astype(int)
    d["consec_up"]   = up * (up.groupby((up != up.shift()).cumsum()).cumcount() + 1)
    d["consec_down"] = dn * (dn.groupby((dn != dn.shift()).cumsum()).cumcount() + 1)

    d["next_ret"]    = c.shift(-1) / c - 1
    d["target_15up"] = (d["next_ret"] >= target_pct).astype(int)

    d = d.replace([np.inf, -np.inf], np.nan)
    return d.dropna(subset=FEATURE_COLS + ["target_15up"]).reset_index(drop=True)
