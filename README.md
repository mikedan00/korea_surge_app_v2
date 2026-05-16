# 🚀 Korea NextDay Surge Predictor v2.0 + Gemma-4

KOSPI + KOSDAQ 종목의 **다음 거래일 상승/하락률**과 **15% 급등 확률**을 앙상블 ML로 예측하고,  
`google/gemma-4-26B-A4B-it` (HF Router) 로 자연어 분석 리포트를 자동 생성합니다.

---

## 📁 프로젝트 구조

```
korea_surge_app/
├── streamlit_app.py          # ① Streamlit 배포용 앱 (메인)
├── run_local.py              # ② VS Code 터미널 실행
├── agent.py                  # 공통 Agent
├── core/
│   ├── data.py               # 데이터 취득 (pykrx / FDR)
│   └── features.py           # 피처 엔지니어링 (34개)
├── models/
│   ├── predictor.py          # 앙상블 분류 + 회귀 모델
│   └── llm_analyst.py        # Gemma-4 HF Router 연동
├── utils/
│   └── search.py             # 종목 검색 + 스코어 산출
├── .streamlit/
│   ├── config.toml           # Streamlit 테마 (다크모드)
│   └── secrets.toml          # ⚠️ 로컬 전용 (Git 제외)
├── .gitignore
└── requirements.txt
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

---

## 🔑 Gemma-4 토큰 설정 (핵심)

### ① Streamlit Cloud 배포 시 → Secrets 등록

Streamlit Cloud 대시보드에서:  
`앱 선택` → `⋮ 메뉴` → `Settings` → `Secrets` → 아래 내용 붙여넣기

```toml
LLM_ENGINE      = "hf_api"
HF_TOKEN        = "hf_여기에_본인_토큰_입력"
HF_MODEL_ID     = "google/gemma-4-26B-A4B-it"
HF_MAX_TOKENS   = "1200"
HF_TEMPERATURE  = "0.2"
```

저장하면 앱이 자동 재시작되고 **Gemma-4가 즉시 연결**됩니다.  
사이드바에서 토큰을 입력할 필요 없습니다.

### ② 로컬 Streamlit 실행 시 → secrets.toml 편집

```bash
# .streamlit/secrets.toml 파일을 열어 토큰 입력
nano .streamlit/secrets.toml
```

```toml
LLM_ENGINE      = "hf_api"
HF_TOKEN        = "hf_여기에_본인_토큰_입력"   # ← 실제 토큰으로 교체
HF_MODEL_ID     = "google/gemma-4-26B-A4B-it"
HF_MAX_TOKENS   = "1200"
HF_TEMPERATURE  = "0.2"
```

> ⚠️ `secrets.toml`은 `.gitignore`에 포함되어 있어 Git에 업로드되지 않습니다.

### ③ VS Code 터미널 실행 시 → 환경변수 or 실행 시 입력

```bash
# 방법 A: 환경변수로 미리 설정
export HF_TOKEN=hf_여기에_본인_토큰_입력
python run_local.py

# 방법 B: 실행 후 프롬프트에서 입력 (토큰 감춰짐)
python run_local.py
# → HF Token (hf_xxx...): 입력창에 붙여넣기
```

---

## 🌐 Streamlit 실행

```bash
# 로컬
streamlit run streamlit_app.py
# → http://localhost:8501 자동 오픈
```

---

## ☁️ Streamlit Cloud 배포 절차

```
1. GitHub 저장소 생성 후 이 폴더 push
   (secrets.toml은 .gitignore로 자동 제외됨)

2. https://streamlit.io/cloud 접속 → New App

3. Repository / Branch / Main file 선택
   Main file: streamlit_app.py

4. Advanced settings → Secrets 탭에 토큰 붙여넣기

5. Deploy! → 자동 빌드 후 Gemma-4 연결 완료
```

---

## 📊 예측 항목

| 항목 | 설명 |
|------|------|
| **예측 수익률 %** | 다음 거래일 예상 등락률 (회귀 앙상블) |
| **예측 상승률 %** | 상승 시나리오 |
| **예측 하락률 %** | 하락 시나리오 |
| **신뢰구간 ±%** | 앙상블 모델 간 편차 |
| **15% 급등 확률** | 분류 앙상블 + 캘리브레이션 |
| **급등 점수** | 종합 0~100 점수 |
| **Gemma-4 리포트** | 자연어 분석 (스트리밍) |

---

## 🤖 Gemma-4 설정 파라미터

| 파라미터 | Secrets 키 | 기본값 | 설명 |
|---|---|---|---|
| 모델명 | `HF_MODEL_ID` | `google/gemma-4-26B-A4B-it` | HF 모델 ID |
| 최대 토큰 | `HF_MAX_TOKENS` | `1200` | 응답 최대 길이 |
| 온도 | `HF_TEMPERATURE` | `0.2` | 낮을수록 일관성↑ |
| 엔진 | `LLM_ENGINE` | `hf_api` | 예약 (현재 hf_api 고정) |

---

## ⚠️ 면책

본 소프트웨어는 교육/연구 목적입니다.  
투자 결과에 대한 책임은 전적으로 사용자에게 있습니다.
