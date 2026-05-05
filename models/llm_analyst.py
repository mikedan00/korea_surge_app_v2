# models/llm_analyst.py  ─ HF Router + Gemma-4-26B 분석 모듈
# ============================================================
#  HuggingFace Inference API (Serverless Router) 경유
#  모델: google/gemma-4-26B-A4B-it
#  역할:
#    1. 단일 종목 → 자연어 분석 리포트 생성
#    2. TOP10 결과 → 종합 시장 판단 + 우선순위 코멘트
#    3. 사용자 자유 질문 → AI 답변 (종목 챗봇)
# ============================================================

import os
import textwrap
from typing import Generator

HF_MODEL = "google/gemma-4-26B-A4B-it"
HF_PROVIDER = "auto"          # HF Router 자동 선택


def _get_client(hf_token: str):
    from huggingface_hub import InferenceClient
    return InferenceClient(
        provider=HF_PROVIDER,
        api_key=hf_token,
    )


# ── 단일 종목 분석 프롬프트 ────────────────────────────────────
def _build_stock_prompt(result: dict) -> str:
    r = result
    pr  = r.get("pred_ret_pct", 0)
    cb  = r.get("conf_band", 0)
    p15 = r.get("prob_up15", 0)
    sc  = r.get("score", 0)

    patterns = []
    if r.get("vol_explosion"):  patterns.append("거래량 폭발")
    if r.get("price_compress"): patterns.append("눌림목 압축")
    if r.get("breakout_flag"):  patterns.append("이동평균 돌파")
    if r.get("near_52w_high"):  patterns.append("52주 신고가 근접")
    pattern_str = ", ".join(patterns) if patterns else "특이 패턴 없음"

    return textwrap.dedent(f"""
    당신은 한국 주식 시장 전문 AI 애널리스트입니다.
    아래 머신러닝 분석 데이터를 바탕으로 **한국어**로 전문적이고 간결한 투자 분석 리포트를 작성하세요.

    ## 분석 데이터
    - 종목명: {r.get('name','N/A')} ({r.get('ticker','')}/{r.get('market','')})
    - 전일 종가: {r.get('close',0):,.0f}원
    - 현재가: {r.get('price_now','N/A')}원
    - 전일 등락률: {r.get('change_pct',0):+.2f}%
    - 거래대금: {r.get('trading_value',0)/1e8:.1f}억원

    ## ML 예측 결과
    - 다음 거래일 예측 수익률: {pr:+.2f}% (신뢰구간 ±{cb:.2f}%)
    - 예측 상승률: +{r.get('pred_up_pct',0):.2f}%
    - 예측 하락률: -{r.get('pred_dn_pct',0):.2f}%
    - 15% 급등 확률: {p15*100:.1f}%
    - 종합 급등 점수: {sc:.1f}/100
    - 방향 판단: {r.get('direction','중립')}
    - 모델 CV AUC: {r.get('cv_auc','N/A')}

    ## 기술적 지표
    - RSI(14): {r.get('rsi14',50):.1f}
    - 볼린저밴드 위치: {r.get('bb_pos',0.5):.2f} (0=하단, 1=상단)
    - 거래량 비율(20일): {r.get('vol_ratio20',1):.2f}배
    - CCI(14): {r.get('cci14',0):.1f}
    - 스토캐스틱K: {r.get('stoch_k',50):.1f}
    - 52주 신고가 대비: {r.get('dist_52w_high',0)*100:.1f}%
    - 감지된 패턴: {pattern_str}

    ## 리포트 작성 지침
    다음 구조로 **300~400자** 분량의 리포트를 작성하세요:

    **[종목 요약]** 한 줄 핵심 요약
    **[기술적 분석]** 주요 지표 해석 (RSI, BB, 거래량 등)
    **[패턴 분석]** 감지된 패턴의 의미와 시사점
    **[리스크 요인]** 주의해야 할 위험 요소
    **[종합 의견]** 매수/관망/주의 의견 및 근거

    ※ 면책: 본 분석은 참고용이며 투자 손실에 책임지지 않습니다.
    """).strip()


# ── TOP10 스캔 결과 종합 프롬프트 ─────────────────────────────
def _build_top10_prompt(df) -> str:
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f"  {int(r['rank'])}위. {r['name']}({r.get('market','-')}) | "
            f"예측수익률:{float(r.get('pred_ret_pct',0)):+.2f}% | "
            f"15%급등확률:{float(r.get('prob_up15',0))*100:.1f}% | "
            f"점수:{float(r.get('score',0)):.1f}"
        )
    stock_list = "\n".join(rows)

    return textwrap.dedent(f"""
    당신은 한국 주식 시장 전문 AI 애널리스트입니다.
    머신러닝이 선별한 다음 거래일 급등 예상 TOP10 종목을 분석하고 **한국어**로 종합 리포트를 작성하세요.

    ## TOP10 종목 목록
    {stock_list}

    ## 리포트 작성 지침
    다음 구조로 **400~500자** 분량의 종합 리포트를 작성하세요:

    **[오늘의 시장 신호]** 전체적인 시장 분위기 판단
    **[주목 TOP3]** 상위 3개 종목 개별 코멘트
    **[공통 패턴]** TOP10에서 공통적으로 보이는 특징
    **[섹터/테마]** 두드러지는 섹터나 테마 분석
    **[투자 전략]** 접근 전략 제안 (분산/집중/관망 등)
    **[리스크 경보]** 주의해야 할 시장 리스크

    ※ 면책: 본 분석은 참고용이며 투자 손실에 책임지지 않습니다.
    """).strip()


# ── 자유 질문 프롬프트 (챗봇) ─────────────────────────────────
def _build_chat_prompt(question: str, context: dict | None = None) -> str:
    ctx = ""
    if context:
        ctx = f"""
## 현재 분석 중인 종목 컨텍스트
- 종목: {context.get('name','N/A')} ({context.get('ticker','')})
- 예측 수익률: {context.get('pred_ret_pct',0):+.2f}%
- 15% 급등 확률: {context.get('prob_up15',0)*100:.1f}%
- 점수: {context.get('score',0):.1f}/100
"""

    return textwrap.dedent(f"""
    당신은 한국 주식 시장 전문 AI 애널리스트입니다.
    사용자의 질문에 **한국어**로 전문적이고 친절하게 답변하세요.
    {ctx}
    ## 사용자 질문
    {question}

    전문적이고 간결하게 (200~300자 이내) 답변하세요.
    투자 손실에 대한 책임 면책을 마지막에 한 줄로 표기하세요.
    """).strip()


# ── 메인 API 함수들 ────────────────────────────────────────────

def analyze_stock_with_gemma(
    result: dict,
    hf_token: str,
    stream: bool = True,
) -> str | Generator:
    """
    단일 종목 ML 결과 → Gemma-4 자연어 분석 리포트
    stream=True: Generator 반환 (Streamlit st.write_stream 용)
    stream=False: 완성 문자열 반환
    """
    client = _get_client(hf_token)
    prompt = _build_stock_prompt(result)

    response = client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0.4,
        top_p=0.9,
        stream=stream,
    )

    if stream:
        def _gen():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()
    else:
        return response.choices[0].message.content


def analyze_top10_with_gemma(
    df,
    hf_token: str,
    stream: bool = True,
) -> str | Generator:
    """TOP10 스캔 결과 → Gemma-4 종합 시장 판단"""
    client = _get_client(hf_token)
    prompt = _build_top10_prompt(df)

    response = client.chat_completion(
        model=HF_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=700,
        temperature=0.4,
        top_p=0.9,
        stream=stream,
    )

    if stream:
        def _gen():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()
    else:
        return response.choices[0].message.content


def chat_with_gemma(
    question: str,
    hf_token: str,
    history: list[dict] | None = None,
    context: dict | None = None,
    stream: bool = True,
) -> str | Generator:
    """
    자유 질문 챗봇 (멀티턴 히스토리 지원)
    history: [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}]
    context: 현재 분석 중인 종목 결과 dict (선택)
    """
    client = _get_client(hf_token)

    system_msg = {
        "role": "user",
        "content": "당신은 한국 주식 시장 전문 AI 애널리스트입니다. 항상 한국어로 답변하세요."
    }

    messages = [system_msg]
    if history:
        messages.extend(history)

    user_prompt = _build_chat_prompt(question, context)
    messages.append({"role": "user", "content": user_prompt})

    response = client.chat_completion(
        model=HF_MODEL,
        messages=messages,
        max_tokens=500,
        temperature=0.5,
        top_p=0.9,
        stream=stream,
    )

    if stream:
        def _gen():
            for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()
    else:
        return response.choices[0].message.content


def test_connection(hf_token: str) -> dict:
    """HF Router 연결 테스트"""
    try:
        client = _get_client(hf_token)
        resp = client.chat_completion(
            model=HF_MODEL,
            messages=[{"role": "user", "content": "안녕하세요. 한 문장으로 자기소개 해주세요."}],
            max_tokens=80,
            stream=False,
        )
        return {
            "success": True,
            "model":   HF_MODEL,
            "reply":   resp.choices[0].message.content,
        }
    except Exception as e:
        return {"success": False, "model": HF_MODEL, "error": str(e)}
