# agent.py
import numpy as np
import pandas as pd
from tqdm import tqdm
from core.data    import build_master, fetch_daily, naver_quote, now_str
from core.features import make_features, FEATURE_COLS
from models.predictor import predict_ticker
from utils.search  import fuzzy_search, resolve_ticker, compute_score, CONFIG

class KoreaSurgeAgent:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self._log("MASTER 종목 목록 로딩 중...")
        self.master = build_master()
        self.ticker_name   = dict(zip(self.master["ticker"],self.master["name"]))
        self.ticker_market = dict(zip(self.master["ticker"],self.master["market"]))
        self._log(f"✅ MASTER {len(self.master):,}개 종목 로드 완료")

    def _log(self, msg):
        if self.verbose: print(msg)

    def search(self, keyword, top_n=10):
        return fuzzy_search(self.master, keyword, top_n=top_n)

    def analyze(self, query, use_realtime=True):
        resolved = resolve_ticker(self.master, query)
        if resolved is None: return None
        ticker, name, market = resolved
        raw = fetch_daily(ticker, CONFIG["history_days"])
        if raw is None or len(raw) < 200: return None
        feat = make_features(raw, CONFIG["target_surge_pct"])
        if feat is None or len(feat) < 150: return None
        model_res = predict_ticker(feat)
        last  = feat.iloc[-1]
        close = float(last["close"])
        rt    = naver_quote(ticker) if use_realtime else {"price_now":np.nan,"volume_now":np.nan}
        price_now  = rt["price_now"]
        volume_now = rt["volume_now"]
        score = compute_score(last, model_res, price_now, volume_now)

        pr = model_res["pred_ret_pct"]
        direction = "상승" if pr>=1.0 else "하락" if pr<=-1.0 else "중립"
        intraday = round((price_now/close-1)*100,2) if not np.isnan(price_now) and close>0 else np.nan

        def sv(k,d=None):
            v=last.get(k,d); return round(float(v),4) if v is not None and not pd.isna(v) else d

        return {
            "ticker":ticker,"name":name,"market":market,
            "close":close,"price_now":price_now,"intraday_ret_pct":intraday,
            "change_pct":sv("change_pct",0.0),"trading_value":sv("trading_value",0.0),
            "pred_ret_pct":model_res["pred_ret_pct"],"pred_up_pct":model_res["pred_up_pct"],
            "pred_dn_pct":model_res["pred_dn_pct"],"conf_band":model_res["conf_band"],
            "prob_up15":model_res["prob_up15"],"direction":direction,
            "score":round(score,1),
            "cv_auc":model_res["cv_auc"],"cv_reg":model_res["cv_reg"],
            "base_rate":model_res["base_rate"],"pos_count":model_res["pos_count"],
            "up_hist_avg":model_res["up_hist_avg"],"dn_hist_avg":model_res["dn_hist_avg"],
            "model_info":model_res["model_info"],
            "rsi14":sv("rsi14",50.0),"macd_gap":sv("macd_gap",0.0),
            "bb_pos":sv("bb_pos",0.5),"bb_width":sv("bb_width",0.0),
            "atr_ratio":sv("atr_ratio",0.0),"vol_ratio20":sv("volume_ratio20",1.0),
            "tv_ratio20":sv("trading_value_ratio20",1.0),
            "stoch_k":sv("stoch_k",50.0),"cci14":sv("cci14",0.0),
            "dist_52w_high":sv("dist_52w_high",0.0),
            "vol_explosion":bool(sv("vol_explosion",0)),
            "price_compress":bool(sv("price_compress",0)),
            "breakout_flag":bool(sv("breakout_flag",0)),
            "near_52w_high":bool(sv("near_52w_high",0)),
            "raw":raw,"feat":feat,"analyzed_at":now_str(),
        }

    def scan_top(self, max_stocks=200, top_n=10, use_realtime=True, progress_cb=None):
        candidates = self._build_candidates(max_stocks)
        if candidates is None or len(candidates)==0: return None
        tickers=candidates["ticker"].tolist(); total=len(tickers); scored=[]
        for i,t in enumerate(tqdm(tickers,desc="모델 예측",disable=not self.verbose)):
            if progress_cb: progress_cb(i,total,t)
            try:
                raw=fetch_daily(t,CONFIG["history_days"])
                if raw is None or len(raw)<200: continue
                feat=make_features(raw,CONFIG["target_surge_pct"])
                if feat is None or len(feat)<150: continue
                model_res=predict_ticker(feat)
                last=feat.iloc[-1]
                close=float(last["close"]); tv=float(last["trading_value"])
                if close<CONFIG["min_price"] or tv<CONFIG["min_trading_value"]: continue
                rt=naver_quote(t) if use_realtime else {"price_now":np.nan,"volume_now":np.nan}
                score=compute_score(last,model_res,rt["price_now"],rt["volume_now"])
                scored.append({
                    "ticker":t,"name":self.ticker_name.get(t,t),
                    "market":self.ticker_market.get(t,"-"),
                    "close":close,"price_now":rt["price_now"],
                    "change_pct":float(last.get("change_pct",0) or 0),
                    "trading_value":tv,
                    "vol_ratio20":float(last.get("volume_ratio20",1) or 1),
                    "rsi14":float(last.get("rsi14",50) or 50),
                    "prob_up15":model_res["prob_up15"],
                    "pred_ret_pct":model_res["pred_ret_pct"],
                    "pred_up_pct":model_res["pred_up_pct"],
                    "pred_dn_pct":model_res["pred_dn_pct"],
                    "conf_band":model_res["conf_band"],
                    "cv_auc":model_res["cv_auc"],
                    "vol_explosion":bool(float(last.get("vol_explosion",0) or 0)),
                    "breakout_flag":bool(float(last.get("breakout_flag",0) or 0)),
                    "near_52w_high":bool(float(last.get("near_52w_high",0) or 0)),
                    "score":round(score,1),
                })
            except Exception: continue
        if progress_cb: progress_cb(total,total,"완료")
        if not scored: return None
        df=pd.DataFrame(scored).sort_values("score",ascending=False).head(top_n).reset_index(drop=True)
        df.insert(0,"rank",range(1,len(df)+1))
        return df

    def _build_candidates(self, max_stocks):
        rows=[]
        for t in tqdm(self.master["ticker"].tolist(),desc="1차 거래대금 필터",disable=not self.verbose):
            try:
                raw=fetch_daily(t,CONFIG["scan_history_days"])
                if raw is None or len(raw)<40: continue
                last=raw.iloc[-1]
                close=float(last["close"]); tv=float(last["trading_value"])
                chg=float(last.get("change_pct",0) or 0)
                if (close>=CONFIG["min_price"] and tv>=CONFIG["min_trading_value"]
                        and chg<CONFIG["already_surged_pct"]*100):
                    rows.append({"ticker":t,"trading_value":tv})
            except Exception: continue
        if not rows: return pd.DataFrame()
        return pd.DataFrame(rows).sort_values("trading_value",ascending=False).head(max_stocks).reset_index(drop=True)
