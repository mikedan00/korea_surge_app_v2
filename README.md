# 🚀 Korea NextDay Surge Predictor v2.0

KOSPI + KOSDAQ 종목의 **다음 거래일 상승/하락률**과 **15% 급등 확률**을 앙상블 ML로 예측합니다.

---

## 📁 프로젝트 구조

```
korea_surge_app/
├── streamlit_app.py     # ① Streamlit 배포용 앱
├── run_local.py         # ② VS Code 로컬 터미널 실행
├── agent.py             # 공통 Agent (양쪽에서 import)
├── core/
│   ├── data.py          # 데이터 취득 (pykrx / FDR)
│   └── features.py      # 피처 엔지니어링 (34개 지표)
├── models/
│   └── predictor.py     # 앙상블 분류 + 회귀 모델
├── utils/
│   └── search.py        # 종목 검색 + 스코어 산출
├── requirements.txt
└── README.md
```

---

## ⚙️ 설치

```bash
pip install -r requirements.txt
```

---

## 🖥️ VS Code 로컬 실행

```bash
cd korea_surge_app
python run_local.py
```

터미널 메뉴가 표시됩니다:
```
1. 종목 검색 (이름/코드)
2. 단일 종목 예측 분석       ← 원하는 종목 입력
3. TOP10 급등 후보 스캔
q. 종료
```

**예시 입력:**
- `삼성전자` → 종목명 한글
- `005930`   → 종목 코드 6자리
- `하이닉스` → 부분 이름도 검색 가능

---

## 🌐 Streamlit 로컬 실행

```bash
cd korea_surge_app
streamlit run streamlit_app.py
```

브라우저 자동 실행 → http://localhost:8501

---

## ☁️ Streamlit Cloud 배포

1. GitHub 저장소에 이 폴더를 올립니다
2. https://streamlit.io/cloud 접속 후 로그인
3. **New App** → 저장소 선택 → `streamlit_app.py` 지정
4. Deploy!

> `requirements.txt`가 같은 폴더에 있으면 자동으로 의존성 설치됩니다.

---

## 📊 예측 항목

| 항목 | 설명 |
|------|------|
| **예측 수익률 %** | 다음 거래일 예상 등락률 (회귀 앙상블) |
| **예측 상승률 %** | 상승 시나리오 예측값 |
| **예측 하락률 %** | 하락 시나리오 예측값 |
| **신뢰구간 ±%** | 앙상블 내 모델 간 예측 편차 |
| **15% 급등 확률** | 다음날 15% 이상 급등할 확률 (분류 앙상블) |
| **급등 점수** | 모든 신호를 종합한 0~100 점수 |
| **방향 판단** | 상승 / 하락 / 중립 |

---

## 🤖 모델 구성

```
분류 (15% 급등 확률):
  RandomForest + ExtraTrees + GradientBoosting + XGBoost* + LightGBM*
  → VotingClassifier(soft) → CalibratedClassifierCV(isotonic)
  → TimeSeriesSplit 5폴드 교차검증 (AUC 출력)

회귀 (다음날 수익률):
  RandomForest + ExtraTrees + GradientBoosting + XGBoost* + LightGBM*
  → VotingRegressor → 신뢰구간 (앙상블 표준편차)

* 설치되어 있을 때만 포함
```

---

## 📐 피처 목록 (34개)

- **수익률**: ret1~ret20
- **이평 괴리**: dist_ma3~dist_ma120
- **거래량/거래대금**: ratio5, ratio20
- **캔들 패턴**: 몸통비율, 윗꼬리, 아랫꼬리, 갭
- **RSI** (7/14/21), **MACD**, **볼린저밴드**, **ATR**
- **스토캐스틱** (K/D), **OBV**, **CCI**
- **52주 신고가/신저가 거리**
- **급등 직전 패턴**: 거래량 폭발, 눌림목 압축, 이평 돌파
- **연속 상승/하락 일수**

---

## ⚠️ 면책

본 소프트웨어는 교육/연구 목적으로 제작되었습니다.  
투자 결과에 대한 책임은 전적으로 사용자에게 있으며,  
투자 손실에 대해 개발자는 어떠한 책임도 지지 않습니다.
