# models/llm_analyst.py  ─ HF Router + Gemma-4-26B
# ============================================================
#  수정 핵심:
#  - stream=False 로 변경 (HF Router 스트림 끊김 오류 근본 해결)
#  - Streamlit write_stream 대신 st.markdown 으로 점진적 표시
#  - 재시도 로직 (최대 3회)
#  - 타임아웃 처리
# ============================================================

import os, time, textwrap

HF_MODEL      = os.environ.get("HF_MODEL_ID",    "google/gemma-4-26B-A4B-it")
_DEFAULT_MAXT = int(os.environ.get("HF_MAX_TOKENS",  "1200"))
_DEFAULT_TEMP = float(os.environ.get("HF_TEMPERATURE", "0.2"))
HF_PROVIDER   = "auto"


def _get_client(hf_token: str):
    from huggingface_hub import InferenceClient
    return InferenceClient(provider=HF_PROVIDER, api_key=hf_token)


# ── 핵심: stream=False 로 안전하게 호출 + 재시도 ─────────────
def _call_hf(
    hf_token: str,
    messages: list,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    HF Router 호출 (stream=False).
    연결 오류 시 최대 max_retries 회 재시도.
    """
    client = _get_client(hf_token)
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat_completion(
                model       = HF_MODEL,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = temperature,
                top_p       = 0.9,
                stream      = False,   # ← 스트림 끊김 오류 방지
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
            raise ValueError("빈 응답 수신")

        except Exception as e:
            last_err = e
            err_msg  = str(e)
            # 스트림 관련 오류 or 일시적 오류 → 재시도
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
                continue
            break

    raise RuntimeError(f"Gemma-4 호출 실패 ({max_retries}회 시도): {last_err}")


# ── 프롬프트: 단일 종목 ───────────────────────────────────────
def _prompt_stock(r: dict) -> str:
    pr  = r.get("pred_ret_pct", 0)
    cb  = r.get("conf_band", 0)
    p15 = r.get("prob_up15", 0)
    sc  = r.get("score", 0)

    pats = []
    if r.get("vol_explosion"):  pats.append("거래량 폭발")
    if r.get("price_compress"): pats.append("눌림목 압축")
    if r.get("breakout_flag"):  pats.append("이동평균 돌파")
    if r.get("near_52w_high"):  pats.append("52주 신고가 근접")
    pat_str = ", ".join(pats) if pats else "특이 패턴 없음"

    return textwrap.dedent(f"""
    당신은 한국 주식 시장 전문 AI 애널리스트입니다.
    아래 데이터를 바탕으로 한국어로 투자 분석 리포트를 작성하세요.

    ## 종목 정보
    - 종목: {r.get('name','N/A')} ({r.get('ticker','')}/{r.get('market','')})
    - 전일 종가: {r.get('close',0):,.0f}원  |  등락률: {r.get('change_pct',0):+.2f}%
    - 거래대금: {r.get('trading_value',0)/1e8:.1f}억원

    ## ML 예측
    - 예측 수익률: {pr:+.2f}% (신뢰구간 ±{cb:.2f}%)
    - 상승 예측: +{r.get('pred_up_pct',0):.2f}%  |  하락 예측: -{r.get('pred_dn_pct',0):.2f}%
    - 15% 급등 확률: {p15*100:.1f}%  |  급등 점수: {sc:.1f}/100
    - 방향 판단: {r.get('direction','중립')}  |  CV AUC: {r.get('cv_auc','N/A')}

    ## 기술적 지표
    - RSI(14): {r.get('rsi14',50):.1f}  |  BB위치: {r.get('bb_pos',0.5):.2f}
    - 거래량비율(20일): {r.get('vol_ratio20',1):.2f}배  |  CCI: {r.get('cci14',0):.1f}
    - 52주 신고가 대비: {r.get('dist_52w_high',0)*100:.1f}%
    - 감지 패턴: {pat_str}

    ## 리포트 형식 (반드시 준수)
    **[종목 요약]** 한 줄 핵심
    **[기술적 분석]** RSI·BB·거래량 해석
    **[패턴 분석]** 감지 패턴 시사점
    **[리스크 요인]** 주의 위험 요소
    **[종합 의견]** 매수/관망/주의 + 근거

    ※ "본 분석은 참고용이며 투자 손실에 책임지지 않습니다."
    """).strip()


# ── 프롬프트: TOP10 종합 ──────────────────────────────────────
def _prompt_top10(df) -> str:
    rows = "\n".join(
        f"  {int(r['rank'])}위. {r['name']}({r.get('market','-')}) | "
        f"예측:{float(r.get('pred_ret_pct',0)):+.2f}% | "
        f"15%확률:{float(r.get('prob_up15',0))*100:.1f}% | "
        f"점수:{float(r.get('score',0)):.1f}"
        for _, r in df.iterrows()
    )
    return textwrap.dedent(f"""
    당신은 한국 주식 시장 전문 AI 애널리스트입니다.
    ML이 선별한 급등 예상 TOP10을 분석하고 한국어로 종합 리포트를 작성하세요.

    ## TOP10 종목
    {rows}

    ## 리포트 형식 (반드시 준수)
    **[시장 신호]** 전체 분위기 판단
    **[주목 TOP3]** 상위 3종목 개별 코멘트
    **[공통 패턴]** TOP10 공통 특징
    **[섹터/테마]** 두드러지는 섹터·테마
    **[투자 전략]** 분산/집중/관망 제안
    **[리스크 경보]** 주의 시장 리스크

    ※ "본 분석은 참고용이며 투자 손실에 책임지지 않습니다."
    """).strip()


# ── 프롬프트: 챗봇 ───────────────────────────────────────────
def _prompt_chat(question: str, context: dict | None) -> str:
    ctx = ""
    if context:
        ctx = f"""
## 현재 분석 종목
- {context.get('name','N/A')} ({context.get('ticker','')}/{context.get('market','')})
- 종가: {context.get('close',0):,.0f}원
- 예측 수익률: {context.get('pred_ret_pct',0):+.2f}% (±{context.get('conf_band',0):.2f}%)
- 상승: +{context.get('pred_up_pct',0):.2f}%  하락: -{context.get('pred_dn_pct',0):.2f}%
- 15% 급등 확률: {context.get('prob_up15',0)*100:.1f}%  점수: {context.get('score',0):.1f}/100
- RSI: {context.get('rsi14',50):.1f}  방향: {context.get('direction','중립')}
"""
    return textwrap.dedent(f"""
    당신은 한국 주식 시장 전문 AI 애널리스트입니다.
    질문에 한국어로 전문적이고 간결하게 답변하세요.
    {ctx}
    ## 질문
    {question}

    200~300자 이내로 핵심만 답하고,
    마지막 줄: "본 분석은 참고용이며 투자 손실에 책임지지 않습니다."
    """).strip()


# ══════════════════════════════════════════════════════════════
#  공개 API
# ══════════════════════════════════════════════════════════════

def analyze_stock_with_gemma(
    result: dict,
    hf_token: str,
    max_tokens: int   = _DEFAULT_MAXT,
    temperature: float = _DEFAULT_TEMP,
) -> str:
    """단일 종목 분석 리포트 (str 반환)"""
    return _call_hf(
        hf_token    = hf_token,
        messages    = [{"role":"user","content":_prompt_stock(result)}],
        max_tokens  = max_tokens,
        temperature = temperature,
    )


def analyze_top10_with_gemma(
    df,
    hf_token: str,
    max_tokens: int   = _DEFAULT_MAXT,
    temperature: float = _DEFAULT_TEMP,
) -> str:
    """TOP10 종합 시장 판단 (str 반환)"""
    return _call_hf(
        hf_token    = hf_token,
        messages    = [{"role":"user","content":_prompt_top10(df)}],
        max_tokens  = max_tokens,
        temperature = temperature,
    )


def chat_with_gemma(
    question: str,
    hf_token: str,
    history: list | None  = None,
    context: dict | None  = None,
    max_tokens: int       = _DEFAULT_MAXT,
    temperature: float    = _DEFAULT_TEMP,
) -> str:
    """멀티턴 챗봇 (str 반환)"""
    messages = [
        {"role":"user",
         "content":"당신은 한국 주식 시장 전문 AI 애널리스트입니다. 항상 한국어로 답변하세요."},
    ]
    if history:
        messages.extend(history[-6:])   # 최근 6턴만 유지
    messages.append({"role":"user","content":_prompt_chat(question, context)})

    return _call_hf(
        hf_token    = hf_token,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
    )


def test_connection(hf_token: str) -> dict:
    """HF Router 연결 테스트"""
    if not hf_token:
        return {"success":False,"model":HF_MODEL,
                "error":"HF_TOKEN이 비어 있습니다.","reply":""}
    try:
        reply = _call_hf(
            hf_token    = hf_token,
            messages    = [{"role":"user",
                            "content":"안녕하세요. 한 문장으로 자기소개 해주세요."}],
            max_tokens  = 120,
            temperature = 0.3,
            max_retries = 2,
        )
        return {"success":True,"model":HF_MODEL,"reply":reply,"error":""}
    except Exception as e:
        return {"success":False,"model":HF_MODEL,"error":str(e),"reply":""}
