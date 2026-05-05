# utils/search.py  ─ 종목 검색 + 스코어링
import re, difflib
import numpy as np
import pandas as pd
from core.data import naver_quote, normalize_text

CONFIG = {
    "min_price":          1_000,
    "min_trading_value":  2_000_000_000,
    "already_surged_pct": 0.08,
    "target_surge_pct":   0.15,
    "history_days":       400,
    "scan_history_days":  120,
    "ts_splits":          5,
    "min_pos_samples":    3,
}


# ── 종목 검색 ──────────────────────────────────────────────────
def fuzzy_search(master: pd.DataFrame, keyword: str, top_n: int = 10) -> pd.DataFrame:
    q = normalize_text(keyword)
    if not q:
        return pd.DataFrame(columns=["ticker","name","market","similarity"])

    # 코드 완전 일치
    exact_code = master[master["ticker"] == q].copy()
    if len(exact_code):
        exact_code["similarity"] = 1.0
        return exact_code[["ticker","name","market","similarity"]].head(top_n)

    # 이름 완전 일치
    exact_name = master[master["search_key"] == q].copy()
    exact_name["similarity"] = 1.0

    # 포함
    contains = master[
        master["search_key"].str.contains(q, na=False) |
        master["ticker"].str.contains(q, na=False)
    ].copy()
    contains["similarity"] = contains["search_key"].apply(
        lambda x: min(0.95, len(q) / max(len(x), 1)))

    # fuzzy
    fuzzy = master.copy()
    fuzzy["similarity"] = fuzzy["search_key"].apply(
        lambda x: difflib.SequenceMatcher(None, q, x).ratio())
    fuzzy = fuzzy.sort_values("similarity", ascending=False).head(top_n * 5)

    out = pd.concat([exact_name, contains, fuzzy], ignore_index=True)
    out = out.sort_values("similarity", ascending=False).drop_duplicates("ticker")
    return out[["ticker","name","market","similarity"]].head(top_n).reset_index(drop=True)


def resolve_ticker(master: pd.DataFrame, query: str, threshold: float = 0.50):
    """입력 문자열 → (ticker, name, market) or None"""
    q = str(query).strip()

    # 6자리 코드 직접 입력
    if re.fullmatch(r"\d{6}", q):
        row = master[master["ticker"] == q]
        if len(row):
            r = row.iloc[0]
            return r["ticker"], r["name"], r["market"]
        return None

    table = fuzzy_search(master, q, top_n=5)
    if len(table) and float(table.iloc[0]["similarity"]) >= threshold:
        r = table.iloc[0]
        return r["ticker"], r["name"], r["market"]
    return None


# ── 스코어 산출 ────────────────────────────────────────────────
def compute_score(last: pd.Series, model_result: dict,
                  price_now=np.nan, volume_now=np.nan) -> float:
    def s(k, d=0.0):
        v = last.get(k, d)
        return float(v) if not pd.isna(v) else d

    vol20    = s("volume_ratio20"); tv20   = s("trading_value_ratio20")
    vol5     = s("volume_ratio5");  ret3   = s("ret3"); ret5 = s("ret5")
    chg      = s("change_pct");     rsi    = s("rsi14", 50)
    macd_gap = s("macd_gap");       macd_s = s("macd_hist_slope")
    dm5      = s("dist_ma5");       dm20   = s("dist_ma20"); dm60 = s("dist_ma60")
    bb_pos   = s("bb_pos",.5);      bb_w   = s("bb_width")
    stoch_k  = s("stoch_k",50);     cci    = s("cci14")
    up_sh    = s("upper_shadow");   lo_sh  = s("lower_shadow")
    vol_ex   = s("vol_explosion");  pc     = s("price_compress")
    bk       = s("breakout_flag");  n52h   = s("near_52w_high")
    obv      = s("obv_slope");      cdn    = s("consec_down")

    close   = s("close", 1)
    vol_ma20 = float(last.get("vol_ma20", 1) or 1)

    # 실시간 부스팅
    rt_boost = 0.0
    if not np.isnan(price_now) and close > 0:
        ir = (price_now / close - 1) * 100
        rt_boost += np.clip(ir, -5, 12) * 1.5
    if not np.isnan(volume_now) and vol_ma20 > 0:
        ivr = volume_now / vol_ma20
        rt_boost += np.clip(ivr, 0, 5) * 3.5

    sc = 0.0
    sc += model_result["prob_up15"] * 40          # [A] 모델 확률
    sc += np.clip(vol20, 0, 6) * 1.5              # [B] 거래량
    sc += np.clip(tv20,  0, 6) * 1.5
    sc += np.clip(vol5,  0, 4) * 1.0
    sc += vol_ex * 3.0
    sc += np.clip(ret3*100,-5,12) * 0.5           # [C] 모멘텀
    sc += np.clip(ret5*100,-5,15) * 0.4
    sc += np.clip(chg, -3, 10)   * 0.5
    sc += 3.0 if dm5  > 0 else 0                  # [D] 추세
    sc += 2.5 if dm20 > 0 else 0
    sc += 2.5 if dm60 > 0 else 0
    sc += 3.0 if 45<=rsi<=75 else 0               # [E] 기술적
    sc += 2.0 if macd_gap>0 and macd_s>0 else 0
    sc += 2.0 if stoch_k > 50 else 0
    sc += 1.0 if cci > 0 else 0
    sc += np.clip(bb_w*100,0,3)*0.8               # [F] 볼린저
    sc += 2.0 if 0.3<=bb_pos<=0.8 else 0
    sc += pc * 4.0 + bk * 3.0                     # [G] 패턴
    sc += lo_sh * 100 * 0.3
    sc += n52h * 2.0                               # [H] 52주
    sc -= np.clip(up_sh*100,0,10)*1.2             # [I] 페널티
    sc -= np.clip(cdn,0,5)*1.5
    sc += np.clip(obv,-2,3)*1.0                   # [J] OBV
    sc += rt_boost                                  # [K] 실시간
    return float(np.clip(sc, 0, 100))
