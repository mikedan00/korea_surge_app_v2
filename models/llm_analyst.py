# models/llm_analyst.py  ─ HF Router + Gemma-4-26B 분석 모듈
# ============================================================
#  환경변수(Streamlit Secrets 자동 주입) 에서 설정 읽기:
#    HF_TOKEN        : HuggingFace API 토큰
#    HF_MODEL_ID     : 모델명  (기본: google/gemma-4-26B-A4B-it)
#    HF_MAX_TOKENS   : 최대 토큰 (기본: 1200)
#    HF_TEMPERATURE  : 온도     (기본: 0.2)
#    LLM_ENGINE      : "hf_api" (예약)
# ============================================================

import os
import textwrap
from typing import Generator

# ── Secrets/환경변수에서 기본값 로드 ──────────────────────────
HF_MODEL       = os.environ.get("HF_MODEL_ID",    "google/gemma-4-26B-A4B-it")
_DEFAULT_MAXT  = int(os.environ.get("HF_MAX_TOKENS",  "1200"))
_DEFAULT_TEMP  = float(os.environ.get("HF_TEMPERATURE", "0.2"))
HF_PROVIDER    = "auto"   # HF Router 자동 선택


# ── InferenceClient 생성 ──────────────────────────────────────
def _get_client(hf_token: str):
    from huggingface_hub import InferenceClient
    return InferenceClient(provider=HF_PROVIDER, api_key=hf_token)


# ── 프롬프트 빌더: 단일 종목 ──────────────────────────────────
def _build_stock_prompt(result: dict) -> str:
    r   = result
    pr  = r.get("pred_ret_pct", 0)
    cb  = r.get("conf_band", 0)
    p15 = r.get("prob_up15", 0)
    sc  = r.get("score", 0)

    patterns = []
    if r.get("vol_explosion"):  patterns.append("거래량 폭발")
    if r.get("price_compress"): patterns.append("눌림목 압축")
    if r.get("breakout_flag"):  patterns.append("이동평균 돌파")
    if r.get("near_52w_high"):  patterns.append("52주 신고가 근접")
    pat_str = ", ".join(patterns) if patterns else "특이 패턴 없음"

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
    - 감지된 패턴: {pat_str}

    ## 리포트 구조 (이 형식을 반드시 따르세요)
    **[종목 요약]** 한 줄 핵심 요약
    **[기술적 분석]** RSI·볼린저밴드·거래량 해석
    **[패턴 분석]** 감지된 패턴의 의미와 시사점
    **[리스크 요인]** 주의해야 할 위험 요소
    **[종합 의견]** 매수/관망/주의 의견 및 근거 (300~400자)

    ※ 마지막 줄: "본 분석은 참고용이며 투자 손실에 책임지지 않습니다."
    """).strip()


# ── 프롬프트 빌더: TOP10 종합 ──────────────────────────────────
def _build_top10_prompt(df) -> str:
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

    ## 리포트 구조 (이 형식을 반드시 따르세요)
    **[오늘의 시장 신호]** 전체적인 시장 분위기 판단
    **[주목 TOP3]** 상위 3개 종목 개별 코멘트
    **[공통 패턴]** TOP10에서 공통적으로 보이는 특징
    **[섹터/테마]** 두드러지는 섹터나 테마 분석
    **[투자 전략]** 접근 전략 제안 (분산/집중/관망 등)
    **[리스크 경보]** 주의해야 할 시장 리스크 (400~500자)

    ※ 마지막 줄: "본 분석은 참고용이며 투자 손실에 책임지지 않습니다."
    """).strip()


# ── 프롬프트 빌더: 챗봇 ───────────────────────────────────────
def _build_chat_prompt(question: str, context: dict | None = None) -> str:
    ctx_str = ""
    if context:
        ctx_str = f"""
## 현재 분석 중인 종목 컨텍스트
- 종목: {context.get('name','N/A')} ({context.get('ticker','')}/{context.get('market','')})
- 전일 종가: {context.get('close',0):,.0f}원
- 예측 수익률: {context.get('pred_ret_pct',0):+.2f}% (±{context.get('conf_band',0):.2f}%)
- 예측 상승률: +{context.get('pred_up_pct',0):.2f}%
- 예측 하락률: -{context.get('pred_dn_pct',0):.2f}%
- 15% 급등 확률: {context.get('prob_up15',0)*100:.1f}%
- 급등 점수: {context.get('score',0):.1f}/100
- RSI(14): {context.get('rsi14',50):.1f}
- 방향 판단: {context.get('direction','중립')}
"""
    return textwrap.dedent(f"""
    당신은 한국 주식 시장 전문 AI 애널리스트입니다.
    사용자의 질문에 **한국어**로 전문적이고 친절하게 답변하세요.
    {ctx_str}
    ## 사용자 질문
    {question}

    200~300자 이내로 핵심만 답변하고,
    마지막 줄에 "본 분석은 참고용이며 투자 손실에 책임지지 않습니다." 를 붙이세요.
    """).strip()


# ══════════════════════════════════════════════════════════════
#  공개 API 함수
# ══════════════════════════════════════════════════════════════

def analyze_stock_with_gemma(
    result: dict,
    hf_token: str,
    stream: bool = True,
    max_tokens: int  = _DEFAULT_MAXT,
    temperature: float = _DEFAULT_TEMP,
) -> str | Generator:
    """단일 종목 ML 결과 → Gemma-4 자연어 분석 리포트"""
    client = _get_client(hf_token)
    prompt = _build_stock_prompt(result)

    resp = client.chat_completion(
        model       = HF_MODEL,
        messages    = [{"role":"user","content":prompt}],
        max_tokens  = max_tokens,
        temperature = temperature,
        top_p       = 0.9,
        stream      = stream,
    )

    if stream:
        def _gen():
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()
    return resp.choices[0].message.content


def analyze_top10_with_gemma(
    df,
    hf_token: str,
    stream: bool = True,
    max_tokens: int  = _DEFAULT_MAXT,
    temperature: float = _DEFAULT_TEMP,
) -> str | Generator:
    """TOP10 스캔 결과 → Gemma-4 종합 시장 판단"""
    client = _get_client(hf_token)
    prompt = _build_top10_prompt(df)

    resp = client.chat_completion(
        model       = HF_MODEL,
        messages    = [{"role":"user","content":prompt}],
        max_tokens  = max_tokens,
        temperature = temperature,
        top_p       = 0.9,
        stream      = stream,
    )

    if stream:
        def _gen():
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()
    return resp.choices[0].message.content


def chat_with_gemma(
    question: str,
    hf_token: str,
    history: list | None = None,
    context: dict | None = None,
    stream: bool = True,
    max_tokens: int  = _DEFAULT_MAXT,
    temperature: float = _DEFAULT_TEMP,
) -> str | Generator:
    """멀티턴 챗봇 (히스토리 + 종목 컨텍스트 지원)"""
    client = _get_client(hf_token)

    messages = [{"role":"user",
                 "content":"당신은 한국 주식 시장 전문 AI 애널리스트입니다. 항상 한국어로 답변하세요."}]

    if history:
        messages.extend(history)

    messages.append({"role":"user", "content":_build_chat_prompt(question, context)})

    resp = client.chat_completion(
        model       = HF_MODEL,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
        top_p       = 0.9,
        stream      = stream,
    )

    if stream:
        def _gen():
            for chunk in resp:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        return _gen()
    return resp.choices[0].message.content


def test_connection(hf_token: str) -> dict:
    """HF Router 연결 및 모델 동작 테스트"""
    if not hf_token:
        return {"success": False, "model": HF_MODEL,
                "error": "HF_TOKEN이 비어 있습니다.", "reply": ""}
    try:
        client = _get_client(hf_token)
        resp = client.chat_completion(
            model       = HF_MODEL,
            messages    = [{"role":"user",
                            "content":"안녕하세요. 한 문장으로 자기소개 해주세요."}],
            max_tokens  = 100,
            temperature = 0.3,
            stream      = False,
        )
        return {
            "success": True,
            "model":   HF_MODEL,
            "reply":   resp.choices[0].message.content,
            "error":   "",
        }
    except Exception as e:
        return {"success": False, "model": HF_MODEL,
                "error": str(e), "reply": ""}
