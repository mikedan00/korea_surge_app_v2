# core/data.py
import re, warnings
import numpy as np
import pandas as pd
import requests
import FinanceDataReader as fdr
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

try:
    from pykrx import stock as krx
except Exception:
    krx = None

def ymd(dt):   return dt.strftime("%Y%m%d")
def now_str(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def normalize_text(x):
    x = str(x).lower().strip()
    x = re.sub(r"\s+", "", x)
    x = re.sub(r"[^0-9a-zA-Z가-힣]", "", x)
    return x

def build_master():
    rows = []
    listing = fdr.StockListing("KRX").copy()
    code_col   = "Code"   if "Code"   in listing.columns else listing.columns[0]
    name_col   = "Name"   if "Name"   in listing.columns else listing.columns[1]
    market_col = "Market" if "Market" in listing.columns else None
    for _, r in listing.iterrows():
        code   = str(r[code_col]).zfill(6)
        name   = str(r[name_col])
        market = str(r[market_col]) if market_col else "KRX"
        if market not in ["KOSPI","KOSDAQ"]:
            continue
        rows.append({"ticker":code,"name":name,"market":market})
    master = pd.DataFrame(rows).drop_duplicates("ticker").reset_index(drop=True)
    master["search_key"] = master["name"].apply(normalize_text)
    return master

def fetch_daily(ticker: str, days: int = 400) -> pd.DataFrame:
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=int(days * 2.0))
    if krx is not None:
        try:
            df = krx.get_market_ohlcv_by_date(ymd(start_dt), ymd(end_dt), ticker)
            if df is not None and len(df) > 0:
                df = df.reset_index().rename(columns={
                    "날짜":"date","시가":"open","고가":"high","저가":"low",
                    "종가":"close","거래량":"volume","거래대금":"trading_value","등락률":"change_pct"})
                if "date" not in df.columns:
                    df = df.rename(columns={df.columns[0]:"date"})
                for c in ["open","high","low","close","volume","trading_value","change_pct"]:
                    df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
                if df["trading_value"].isna().all():
                    df["trading_value"] = df["close"] * df["volume"]
                df["ticker"] = ticker
                return df.tail(days).reset_index(drop=True)
        except Exception:
            pass
    try:
        df = fdr.DataReader(ticker, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df is None or len(df) == 0:
            return pd.DataFrame()
        df = df.reset_index().rename(columns={
            "Date":"date","Open":"open","High":"high","Low":"low",
            "Close":"close","Volume":"volume","Change":"change_ratio"})
        df["change_pct"]    = df.get("change_ratio", df["close"].pct_change()) * 100
        df["trading_value"] = df["close"] * df["volume"]
        df["ticker"]        = ticker
        for c in ["open","high","low","close","volume","trading_value","change_pct"]:
            df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        return df.tail(days).reset_index(drop=True)
    except Exception:
        return pd.DataFrame()

def naver_quote(ticker: str) -> dict:
    url = f"https://finance.naver.com/item/main.naver?code={ticker}"
    try:
        html  = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=5).text
        soup  = BeautifulSoup(html, "lxml")
        p     = soup.select_one("p.no_today span.blind")
        price = int(p.text.replace(",","")) if p else np.nan
        text  = soup.get_text(" ", strip=True)
        vol   = np.nan
        m = re.search(r"거래량\s*([0-9,]+)", text)
        if m: vol = int(m.group(1).replace(",",""))
        return {"price_now":price,"volume_now":vol}
    except Exception:
        return {"price_now":np.nan,"volume_now":np.nan}
