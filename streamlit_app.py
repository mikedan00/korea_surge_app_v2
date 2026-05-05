# streamlit_app.py  ─ Streamlit 배포용 (Gemma-4 HF Router 통합)
# 실행: streamlit run streamlit_app.py
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agent import KoreaSurgeAgent
from models.llm_analyst import (
    analyze_stock_with_gemma,
    analyze_top10_with_gemma,
    chat_with_gemma,
    test_connection,
    HF_MODEL,
)

# ── 페이지 설정 ────────────────────────────────────────────────
st.set_page_config(
    page_title="KR 급등 예측기 + Gemma-4",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&family=JetBrains+Mono:wght@400;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }

.main-title {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f59e0b 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: .1rem;
}
.sub-title { font-size: .9rem; color: #94a3b8; margin-bottom: 1.2rem; }

.metric-card {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    border: 1px solid #334155; border-radius: 12px;
    padding: 1rem 1.2rem; text-align: center; transition: border-color .2s;
}
.metric-card:hover { border-color: #7c3aed; }
.metric-label { font-size:.72rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; }
.metric-value { font-size:1.7rem; font-weight:900; margin:.15rem 0;
                font-family:'JetBrains Mono',monospace; }
.metric-sub   { font-size:.78rem; color:#64748b; }
.up-color   { color:#22c55e; }
.down-color { color:#ef4444; }
.neu-color  { color:#f59e0b; }

.badge-up   { background:#16a34a22;color:#22c55e;border:1px solid #22c55e44;
              padding:.2rem .7rem;border-radius:99px;font-size:.82rem;font-weight:700; }
.badge-down { background:#dc262622;color:#ef4444;border:1px solid #ef444444;
              padding:.2rem .7rem;border-radius:99px;font-size:.82rem;font-weight:700; }
.badge-neu  { background:#d9770622;color:#f59e0b;border:1px solid #f59e0b44;
              padding:.2rem .7rem;border-radius:99px;font-size:.82rem;font-weight:700; }

.tag { display:inline-block;padding:.12rem .5rem;border-radius:6px;
       font-size:.76rem;font-weight:600;margin:.1rem; }
.tag-on  { background:#7c3aed33;color:#a78bfa;border:1px solid #7c3aed55; }
.tag-off { background:#1e293b;  color:#475569;border:1px solid #334155; }

.gemma-box {
    background: linear-gradient(135deg,#0f172a,#1a0a2e);
    border: 1px solid #7c3aed55; border-radius: 12px;
    padding: 1.2rem 1.4rem; margin-top: .8rem;
}
.gemma-header { font-size:.8rem; color:#a78bfa; font-weight:700;
                text-transform:uppercase; letter-spacing:.08em; margin-bottom:.5rem; }
.conn-ok  { color:#22c55e; font-weight:700; }
.conn-err { color:#ef4444; font-weight:700; }

.chat-user { background:#1e293b; border-radius:10px; padding:.7rem 1rem;
             margin:.4rem 0; border-left:3px solid #60a5fa; }
.chat-ai   { background:#1a0a2e; border-radius:10px; padding:.7rem 1rem;
             margin:.4rem 0; border-left:3px solid #a78bfa; }

[data-testid="stSidebar"] { background:#0f172a; }
</style>
""", unsafe_allow_html=True)


# ── 헬퍼 ──────────────────────────────────────────────────────
def dir_badge(d):
    cls  = {"상승":"badge-up","하락":"badge-down","중립":"badge-neu"}.get(d,"badge-neu")
    icon = {"상승":"📈","하락":"📉","중립":"➡️"}.get(d,"")
    return f'<span class="{cls}">{icon} {d}</span>'

def tag(label, active):
    c = "tag-on" if active else "tag-off"
    return f'<span class="tag {c}">{label}</span>'

def mc(label, value, sub="", color_class=""):
    return f"""<div class="metric-card">
  <div class="metric-label">{label}</div>
  <div class="metric-value {color_class}">{value}</div>
  <div class="metric-sub">{sub}</div>
</div>"""

@st.cache_resource(show_spinner="종목 목록 로딩 중…")
def get_agent():
    return KoreaSurgeAgent(verbose=False)

def hf_token() -> str:
    return st.session_state.get("hf_token", "")

def has_token() -> bool:
    return bool(hf_token())


# ── 차트 ──────────────────────────────────────────────────────
def plot_candle(raw, result):
    df = raw.tail(90).copy()
    df["date"] = pd.to_datetime(df["date"])
    feat   = result["feat"].tail(90)
    fdates = pd.to_datetime(raw.tail(len(feat))["date"].values)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[.6,.2,.2], vertical_spacing=.03,
                        subplot_titles=["주가(캔들)","거래량","RSI(14)"])

    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"],
        low=df["low"], close=df["close"], name="OHLC",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444"
    ), row=1, col=1)

    for ma, col in [("ma5","#f59e0b"),("ma20","#60a5fa"),("ma60","#a78bfa")]:
        if ma in feat.columns:
            fig.add_trace(go.Scatter(x=fdates, y=feat[ma], name=ma.upper(),
                line=dict(color=col, width=1.2), opacity=.8), row=1, col=1)

    c2 = feat["close"]
    bm = c2.rolling(20).mean(); bs = c2.rolling(20).std()
    fig.add_trace(go.Scatter(x=fdates, y=bm+2*bs,
        line=dict(color="#475569",width=.8,dash="dash"), name="BB상단", opacity=.5), row=1,col=1)
    fig.add_trace(go.Scatter(x=fdates, y=bm-2*bs,
        line=dict(color="#475569",width=.8,dash="dash"), name="BB하단",
        fill="tonexty", fillcolor="rgba(71,85,105,0.06)", opacity=.5), row=1,col=1)

    colors = ["#22c55e" if c>=o else "#ef4444"
              for c,o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"],
        marker_color=colors, name="거래량", opacity=.7), row=2, col=1)

    if "rsi14" in feat.columns:
        fig.add_trace(go.Scatter(x=fdates, y=feat["rsi14"],
            line=dict(color="#f59e0b",width=1.5), name="RSI14"), row=3, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#ef4444", opacity=.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", opacity=.5, row=3, col=1)

    fig.update_layout(
        height=520, paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8",size=11), xaxis_rangeslider_visible=False,
        legend=dict(orientation="h",y=1.02,bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=10,r=10,t=40,b=10)
    )
    for i in range(1,4):
        fig.update_xaxes(gridcolor="#1e293b", row=i, col=1)
        fig.update_yaxes(gridcolor="#1e293b", row=i, col=1)
    return fig


def plot_gauge(pred_ret, conf_band):
    color = "#22c55e" if pred_ret >= 0 else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=pred_ret,
        delta={"reference":0,"valueformat":".2f",
               "increasing":{"color":"#22c55e"},"decreasing":{"color":"#ef4444"}},
        number={"suffix":"%","font":{"size":30,"color":color}},
        gauge={
            "axis":{"range":[-15,15],"tickcolor":"#475569","tickfont":{"color":"#94a3b8","size":10}},
            "bar":{"color":color,"thickness":.25},"bgcolor":"#1e293b","bordercolor":"#334155",
            "steps":[{"range":[-15,-5],"color":"#2d1b1b"},{"range":[-5,0],"color":"#1f1818"},
                     {"range":[0,5],"color":"#172117"},{"range":[5,15],"color":"#0f2a17"}],
            "threshold":{"line":{"color":"#f59e0b","width":2},"thickness":.7,"value":pred_ret},
        },
        title={"text":"다음날 예측 수익률","font":{"color":"#94a3b8","size":13}},
    ))
    fig.add_annotation(
        text=f"신뢰구간: {pred_ret-conf_band:+.2f}% ~ {pred_ret+conf_band:+.2f}%",
        xref="paper", yref="paper", x=.5, y=-.1,
        showarrow=False, font=dict(color="#64748b",size=11)
    )
    fig.update_layout(height=270, paper_bgcolor="#0f172a",
        font=dict(color="#94a3b8"), margin=dict(l=20,r=20,t=60,b=55))
    return fig


def plot_scan_bar(df):
    colors = ["#22c55e" if v>=0 else "#ef4444" for v in df["pred_ret_pct"]]
    fig = go.Figure(go.Bar(
        x=df["name"], y=df["pred_ret_pct"], marker_color=colors,
        text=[f"{v:+.1f}%" for v in df["pred_ret_pct"]], textposition="outside"
    ))
    fig.update_layout(
        title="TOP10 예측 수익률 비교", height=300,
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155"),
        margin=dict(l=10,r=10,t=50,b=10)
    )
    return fig


# ══════════════════════════════════════════════════════════════
#  사이드바
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    # ── HF Token 입력 ────────────────────────────────────────
    st.markdown("### 🤖 Gemma-4 설정")
    st.caption(f"`{HF_MODEL}`")

    token_input = st.text_input(
        "HuggingFace API Token",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="HuggingFace → Settings → Access Tokens 에서 발급 (무료)"
    )
    if token_input:
        st.session_state["hf_token"] = token_input

    if has_token():
        if st.button("🔌 연결 테스트", use_container_width=True):
            with st.spinner("Gemma-4 연결 확인 중…"):
                res = test_connection(hf_token())
            if res["success"]:
                st.success("✅ HF Router 연결 성공!")
                st.info(f"💬 {res['reply']}")
                st.session_state["gemma_ok"] = True
            else:
                st.error(f"❌ 연결 실패\n{res['error']}")
                st.session_state["gemma_ok"] = False
    else:
        st.caption("토큰 입력 → 연결 테스트")

    gemma_ok = st.session_state.get("gemma_ok", False)
    status_html = ('<span class="conn-ok">● Gemma-4 연결됨</span>'
                   if gemma_ok else
                   '<span style="color:#475569">○ 미연결</span>')
    st.markdown(status_html, unsafe_allow_html=True)
    st.markdown("---")

    # ── 모드 선택 ─────────────────────────────────────────────
    mode = st.radio("모드 선택", [
        "🔍 단일 종목 분석",
        "🚀 TOP10 급등 스캔",
        "💬 AI 종목 챗봇",
    ])
    st.markdown("---")

    if "단일" in mode:
        ticker_query = st.text_input("종목명 또는 코드",
                                      placeholder="삼성전자 / 005930 / 하이닉스")
        use_rt  = st.toggle("실시간 시세 반영", value=True)
        use_llm = st.toggle("Gemma-4 AI 분석", value=gemma_ok, disabled=not gemma_ok)
        run_btn = st.button("📊 분석 시작", use_container_width=True, type="primary")

    elif "TOP10" in mode:
        scan_n  = st.slider("스캔 종목 수", 50, 500, 150, 50)
        top_n   = st.slider("TOP N", 5, 20, 10)
        use_rt  = st.toggle("실시간 시세 반영", value=True)
        use_llm = st.toggle("Gemma-4 종합 분석", value=gemma_ok, disabled=not gemma_ok)
        run_btn = st.button("🚀 스캔 시작", use_container_width=True, type="primary")

    else:
        run_btn = False; use_llm = True
        use_rt  = True;  scan_n = 150; top_n = 10
        ticker_query = ""

    st.markdown("""
**⚠️ 투자 주의**
본 예측은 참고용이며 투자 손실에 대해 책임지지 않습니다.
""")


# ══════════════════════════════════════════════════════════════
#  헤더
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">🚀 KR NextDay Surge Predictor + Gemma-4</div>',
            unsafe_allow_html=True)
g_badge = (f'<span style="color:#a78bfa;font-size:.85rem">● {HF_MODEL} 연결됨</span>'
           if gemma_ok else
           '<span style="color:#475569;font-size:.85rem">○ Gemma-4 미연결</span>')
st.markdown(f'<div class="sub-title">KOSPI+KOSDAQ │ 앙상블 ML │ 다음날 수익률 예측 &nbsp; {g_badge}</div>',
            unsafe_allow_html=True)

agent = get_agent()


# ══════════════════════════════════════════════════════════════
#  모드 1 ─ 단일 종목 분석
# ══════════════════════════════════════════════════════════════
if "단일" in mode:
    if run_btn and ticker_query:
        with st.spinner(f"'{ticker_query}' ML 분석 중… (30~60초)"):
            r = agent.analyze(ticker_query, use_realtime=use_rt)

        if r is None:
            st.error("❌ 종목을 찾을 수 없거나 데이터가 부족합니다.")
        else:
            st.session_state["last_result"] = r
            st.markdown(f"### {r['name']}  `{r['ticker']}`  {r['market']}")
            st.markdown("---")

            # ── 핵심 예측 카드 5개 ───────────────────────────
            pr = r["pred_ret_pct"]; cb = r["conf_band"]
            col_c = "up-color" if pr >= 0 else "down-color"

            c1,c2,c3,c4,c5 = st.columns(5)
            c1.markdown(mc("예측 수익률",   f"{pr:+.2f}%",
                           f"± {cb:.2f}%", col_c), unsafe_allow_html=True)
            c2.markdown(mc("예측 상승률",   f"+{r['pred_up_pct']:.2f}%",
                           f"과거평균 {r['up_hist_avg'] or '-'}%",
                           "up-color"), unsafe_allow_html=True)
            c3.markdown(mc("예측 하락률",   f"-{r['pred_dn_pct']:.2f}%",
                           f"과거평균 -{r['dn_hist_avg'] or '-'}%",
                           "down-color"), unsafe_allow_html=True)
            c4.markdown(mc("15% 급등 확률", f"{r['prob_up15']*100:.1f}%",
                           f"과거 {r['pos_count']}회",
                           "neu-color"), unsafe_allow_html=True)
            c5.markdown(mc("급등 점수",     f"{r['score']:.1f}",
                           "/ 100", col_c), unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # ── 방향 배지 + 패턴 태그 ───────────────────────
            st.markdown(
                dir_badge(r["direction"]) + "&nbsp;&nbsp;" +
                tag("🔥 거래량폭발", r["vol_explosion"]) +
                tag("📌 눌림목압축", r["price_compress"]) +
                tag("⚡ 이평돌파",   r["breakout_flag"]) +
                tag("⭐ 52주신고가", r["near_52w_high"]),
                unsafe_allow_html=True
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # ── 게이지 + 캔들 차트 ──────────────────────────
            ga, ca = st.columns([1,2])
            with ga:
                st.plotly_chart(plot_gauge(pr, cb), use_container_width=True)
            with ca:
                st.plotly_chart(plot_candle(r["raw"], r), use_container_width=True)

            # ── 기술적 지표 ──────────────────────────────────
            st.markdown("#### 🔬 기술적 지표")
            ia, ib, ic = st.columns(3)
            with ia:
                st.metric("RSI(14)",          f"{r['rsi14']:.1f}")
                st.metric("BB 위치",           f"{r['bb_pos']:.2f}  (0=하단 / 1=상단)")
                st.metric("ATR 비율",          f"{r['atr_ratio']*100:.2f}%")
            with ib:
                st.metric("거래량 비율(20일)", f"{r['vol_ratio20']:.2f}x")
                st.metric("CCI(14)",           f"{r['cci14']:.1f}")
                st.metric("스토캐스틱K",        f"{r['stoch_k']:.1f}")
            with ic:
                st.metric("52주 신고가 거리",  f"{r['dist_52w_high']*100:.1f}%")
                st.metric("MACD Gap",          f"{r['macd_gap']:.4f}")
                cv = r["cv_auc"]
                st.metric("CV AUC (분류)",     f"{cv:.3f}" if cv else "N/A")

            # ── Gemma-4 AI 분석 리포트 (스트리밍) ───────────
            if use_llm and gemma_ok:
                st.markdown("---")
                st.markdown(f"""
<div class="gemma-box">
  <div class="gemma-header">🤖 Gemma-4 AI 분석 리포트 &nbsp;·&nbsp; {HF_MODEL}</div>
""", unsafe_allow_html=True)
                ph = st.empty()
                full = ""
                try:
                    gen = analyze_stock_with_gemma(r, hf_token(), stream=True)
                    for chunk in gen:
                        full += chunk
                        ph.markdown(full + "▌")
                    ph.markdown(full)
                except Exception as e:
                    ph.error(f"Gemma-4 오류: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

            elif use_llm and not gemma_ok:
                st.info("💡 Gemma-4 AI 분석을 사용하려면 사이드바에서 HF 토큰을 입력하세요.")

            # ── 모델 메타 ────────────────────────────────────
            with st.expander("🤖 모델 정보"):
                mm1,mm2,mm3 = st.columns(3)
                mm1.metric("CV AUC (분류)",  r["cv_auc"] or "N/A")
                mm2.metric("CV Score (회귀)", r["cv_reg"] or "N/A")
                mm3.metric("과거 15%↑ 발생률", f"{r['base_rate']*100:.1f}%")
                st.caption(f"모델: {r['model_info']}  |  분석시각: {r['analyzed_at']}")

    elif run_btn:
        st.warning("종목명 또는 코드를 입력해 주세요.")
    else:
        st.info("👈 사이드바에서 종목을 입력하고 **분석 시작**을 누르세요.")
        st.markdown("""
| 입력 예시 | 설명 |
|---|---|
| `삼성전자` | 종목명 한글 |
| `005930` | 종목 코드 6자리 |
| `하이닉스` | 부분 이름 퍼지 검색 |
| `카카오` | 부분 이름 |
""")


# ══════════════════════════════════════════════════════════════
#  모드 2 ─ TOP10 급등 스캔
# ══════════════════════════════════════════════════════════════
elif "TOP10" in mode:
    if run_btn:
        pb = st.progress(0, text="스캔 준비 중…")

        def pcb(i, total, name):
            if total > 0:
                pb.progress(min(int(i/total*100),100),
                            text=f"분석 중: {name}  ({i}/{total})")

        df = agent.scan_top(max_stocks=scan_n, top_n=top_n,
                             use_realtime=use_rt, progress_cb=pcb)
        pb.empty()

        if df is None or len(df) == 0:
            st.error("❌ 결과가 없습니다.")
        else:
            st.session_state["last_scan"] = df
            st.success(f"✅ {scan_n}개 종목 스캔 완료 → TOP{top_n} 추출")

            st.plotly_chart(plot_scan_bar(df), use_container_width=True)

            # 상세 테이블
            st.markdown("#### 📋 상세 결과")
            disp = df[["rank","name","market","close",
                        "pred_ret_pct","pred_up_pct","pred_dn_pct","conf_band",
                        "prob_up15","cv_auc","score","vol_ratio20","rsi14",
                        "vol_explosion","breakout_flag","near_52w_high"]].copy()
            disp.columns = ["순위","종목명","시장","종가",
                             "예측수익률%","상승%","하락%","신뢰±",
                             "15%확률","CV_AUC","점수","거래량비율","RSI14",
                             "거래량폭발","이평돌파","52주신고가"]

            def highlight(s):
                try:
                    v = float(s)
                    return "color:#22c55e;font-weight:700" if v>=0 else "color:#ef4444;font-weight:700"
                except:
                    return ""

            styled = disp.style\
                .applymap(highlight, subset=["예측수익률%","상승%"])\
                .format({
                    "종가":       "{:,.0f}",
                    "예측수익률%":"{:+.2f}",
                    "상승%":      "+{:.2f}",
                    "하락%":      "-{:.2f}",
                    "신뢰±":      "±{:.2f}",
                    "15%확률":    "{:.1%}",
                    "점수":       "{:.1f}",
                    "거래량비율": "{:.2f}x",
                    "RSI14":     "{:.0f}",
                    "CV_AUC":    lambda x: f"{x:.3f}" if pd.notna(x) else "N/A",
                }, na_rep="N/A")
            st.dataframe(styled, use_container_width=True, height=400)

            # ── Gemma-4 종합 시장 판단 ───────────────────────
            if use_llm and gemma_ok:
                st.markdown("---")
                st.markdown(f"""
<div class="gemma-box">
  <div class="gemma-header">🤖 Gemma-4 종합 시장 판단 &nbsp;·&nbsp; {HF_MODEL}</div>
""", unsafe_allow_html=True)
                ph = st.empty(); full = ""
                try:
                    gen = analyze_top10_with_gemma(df, hf_token(), stream=True)
                    for chunk in gen:
                        full += chunk
                        ph.markdown(full + "▌")
                    ph.markdown(full)
                except Exception as e:
                    ph.error(f"Gemma-4 오류: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

            # 개별 종목 상세
            st.markdown("---")
            st.markdown("#### 🔎 개별 종목 상세 분석")
            sel = st.selectbox("종목 선택", df["name"].tolist())
            if st.button("상세 분석", type="secondary"):
                with st.spinner(f"'{sel}' 분석 중…"):
                    r2 = agent.analyze(sel, use_realtime=use_rt)
                if r2:
                    st.session_state["last_result"] = r2
                    pr2 = r2["pred_ret_pct"]; cb2 = r2["conf_band"]
                    d1,d2,d3 = st.columns(3)
                    d1.metric("예측 수익률",   f"{pr2:+.2f}%", f"±{cb2:.2f}%")
                    d2.metric("15% 급등 확률", f"{r2['prob_up15']*100:.1f}%")
                    d3.metric("급등 점수",     f"{r2['score']:.1f}/100")
                    st.plotly_chart(plot_candle(r2["raw"], r2), use_container_width=True)

                    if use_llm and gemma_ok:
                        st.markdown(f"""
<div class="gemma-box">
  <div class="gemma-header">🤖 Gemma-4 개별 분석</div>
""", unsafe_allow_html=True)
                        ph3 = st.empty(); full3 = ""
                        try:
                            gen3 = analyze_stock_with_gemma(r2, hf_token(), stream=True)
                            for chunk in gen3:
                                full3 += chunk
                                ph3.markdown(full3 + "▌")
                            ph3.markdown(full3)
                        except Exception as e:
                            ph3.error(f"오류: {e}")
                        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info(f"👈 설정 후 **스캔 시작**을 눌러주세요.")
        st.markdown("""
> **소요 시간 안내**
> - 50종목 → 약 5~10분
> - 150종목 → 약 15~25분
> - 300종목 → 약 30~50분
""")


# ══════════════════════════════════════════════════════════════
#  모드 3 ─ AI 종목 챗봇
# ══════════════════════════════════════════════════════════════
else:
    st.markdown("### 💬 Gemma-4 주식 AI 챗봇")
    st.caption(f"모델: `{HF_MODEL}` │ 분석한 종목 컨텍스트 자동 연동")

    if not gemma_ok:
        st.warning("⚠️ 사이드바에서 HuggingFace API 토큰을 입력하고 연결 테스트를 완료해 주세요.")
        st.stop()

    # 세션 초기화
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # 컨텍스트 표시
    ctx = st.session_state.get("last_result", None)
    if ctx:
        st.info(
            f"📌 현재 컨텍스트: **{ctx['name']}** "
            f"(예측: {ctx['pred_ret_pct']:+.2f}% │ 점수: {ctx['score']:.1f}점) "
            f"— 단일 분석 탭에서 다른 종목 분석 시 자동 변경"
        )
    else:
        st.caption("💡 '단일 종목 분석' 탭에서 종목을 분석하면 컨텍스트가 자동으로 연동됩니다.")

    # 채팅 히스토리 출력
    for msg in st.session_state["chat_history"]:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">👤 &nbsp;{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 &nbsp;{msg["content"]}</div>',
                        unsafe_allow_html=True)

    # 빠른 질문 버튼
    st.markdown("**빠른 질문:**")
    q_cols = st.columns(4)
    quick_qs = [
        "이 종목 내일 매수 타이밍인가요?",
        "RSI 70이면 어떻게 해야 하나요?",
        "볼린저밴드 상단 돌파 의미는?",
        "15% 급등 가능성 높은 조건은?",
    ]
    selected_q = None
    for qc, qq in zip(q_cols, quick_qs):
        with qc:
            if st.button(qq, use_container_width=True):
                selected_q = qq

    # 입력창
    st.markdown("<br>", unsafe_allow_html=True)
    col_i, col_s, col_c = st.columns([6,1,1])
    with col_i:
        user_q = st.text_input("질문 입력",
            placeholder="예: 이 종목 사도 될까요? / 거래량 폭발이 왜 중요한가요?",
            label_visibility="collapsed", key="chat_input")
    with col_s:
        send = st.button("전송 ▶", type="primary", use_container_width=True)
    with col_c:
        if st.button("초기화 🗑", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()

    question = selected_q or (user_q.strip() if send and user_q.strip() else None)

    if question:
        st.session_state["chat_history"].append({"role":"user","content":question})
        st.markdown(f'<div class="chat-user">👤 &nbsp;{question}</div>',
                    unsafe_allow_html=True)

        ai_ph = st.empty()
        full_reply = ""
        try:
            hist = st.session_state["chat_history"][:-1][-6:]
            gen  = chat_with_gemma(
                question  = question,
                hf_token  = hf_token(),
                history   = hist,
                context   = ctx,
                stream    = True,
            )
            for chunk in gen:
                full_reply += chunk
                ai_ph.markdown(
                    f'<div class="chat-ai">🤖 &nbsp;{full_reply}▌</div>',
                    unsafe_allow_html=True
                )
            ai_ph.markdown(
                f'<div class="chat-ai">🤖 &nbsp;{full_reply}</div>',
                unsafe_allow_html=True
            )
        except Exception as e:
            full_reply = f"오류 발생: {e}"
            ai_ph.error(full_reply)

        st.session_state["chat_history"].append(
            {"role":"assistant","content":full_reply}
        )
