# utils/search.py
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
    "min_pos_samples":    3,
}

def fuzzy_search(master, keyword, top_n=10):
    q = normalize_text(keyword)
    if not q:
        return pd.DataFrame(columns=["ticker","name","market","similarity"])
    code_exact = master[master["ticker"]==q].copy()
    if len(code_exact):
        code_exact["similarity"]=1.0
        return code_exact[["ticker","name","market","similarity"]].head(top_n)
    exact = master[master["search_key"]==q].copy(); exact["similarity"]=1.0
    contains = master[master["search_key"].str.contains(q,na=False)|
                      master["ticker"].str.contains(q,na=False)].copy()
    if len(contains):
        contains["similarity"]=contains["search_key"].apply(
            lambda x:min(0.95,len(q)/max(len(x),1)))
    fuzzy = master.copy()
    fuzzy["similarity"]=fuzzy["search_key"].apply(
        lambda x:difflib.SequenceMatcher(None,q,x).ratio())
    fuzzy=fuzzy.sort_values("similarity",ascending=False).head(top_n*5)
    fuzzy["match_type"]="fuzzy"
    out=pd.concat([exact,contains,fuzzy],ignore_index=True)
    out=out.sort_values("similarity",ascending=False).drop_duplicates("ticker")
    return out[["ticker","name","market","similarity"]].head(top_n).reset_index(drop=True)

def resolve_ticker(master, query, threshold=0.50):
    q=str(query).strip()
    if re.fullmatch(r"\d{6}",q):
        row=master[master["ticker"]==q]
        if len(row): r=row.iloc[0]; return r["ticker"],r["name"],r["market"]
        return None
    table=fuzzy_search(master,q,top_n=5)
    if len(table) and float(table.iloc[0]["similarity"])>=threshold:
        r=table.iloc[0]; return r["ticker"],r["name"],r["market"]
    return None

def compute_score(last, model_result, price_now=np.nan, volume_now=np.nan):
    def s(k,d=0.0):
        v=last.get(k,d); return float(v) if not pd.isna(v) else d
    vol20=s("volume_ratio20"); tv20=s("trading_value_ratio20"); vol5=s("volume_ratio5")
    ret3=s("ret3"); ret5=s("ret5"); chg=s("change_pct")
    rsi=s("rsi14",50); macd_gap=s("macd_gap"); macd_s=s("macd_hist_slope")
    dm5=s("dist_ma5"); dm20=s("dist_ma20"); dm60=s("dist_ma60")
    bb_pos=s("bb_pos",.5); stoch_k=s("stoch_k",50); cci=s("cci14")
    up_sh=s("upper_shadow"); lo_sh=s("lower_shadow")
    vol_ex=s("vol_explosion"); pc=s("price_compress"); bk=s("breakout_flag")
    n52h=s("near_52w_high"); obv=s("obv_slope"); cdn=s("consec_down")
    close=s("close",1); vol_ma20=float(last.get("vol_ma20",1) or 1)

    rt_boost=0.0
    if not np.isnan(price_now) and close>0:
        rt_boost+=np.clip((price_now/close-1)*100,-5,12)*1.5
    if not np.isnan(volume_now) and vol_ma20>0:
        rt_boost+=np.clip(volume_now/vol_ma20,0,5)*3.5

    sc=0.0
    sc+=model_result["prob_up15"]*40
    sc+=np.clip(vol20,0,6)*1.5+np.clip(tv20,0,6)*1.5+np.clip(vol5,0,4)+vol_ex*3
    sc+=np.clip(ret3*100,-5,12)*.5+np.clip(ret5*100,-5,15)*.4+np.clip(chg,-3,10)*.5
    sc+=(3 if dm5>0 else 0)+(2.5 if dm20>0 else 0)+(2.5 if dm60>0 else 0)
    sc+=(3 if 45<=rsi<=75 else 0)+(2 if macd_gap>0 and macd_s>0 else 0)
    sc+=(2 if stoch_k>50 else 0)+(1 if cci>0 else 0)
    sc+=pc*4+bk*3+lo_sh*100*.3+n52h*2
    sc-=np.clip(up_sh*100,0,10)*1.2+np.clip(cdn,0,5)*1.5
    sc+=np.clip(obv,-2,3)+rt_boost
    return float(np.clip(sc,0,100))
