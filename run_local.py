#!/usr/bin/env python3
# run_local.py  ─ VS Code 로컬 터미널 실행 (Gemma-4 HF Router 통합)
# 실행: python run_local.py
# ============================================================

import sys, os, getpass
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

from agent import KoreaSurgeAgent
from models.llm_analyst import (
    analyze_stock_with_gemma,
    analyze_top10_with_gemma,
    chat_with_gemma,
    test_connection,
    HF_MODEL,
)

try:
    from rich.console  import Console
    from rich.table    import Table
    from rich.panel    import Panel
    from rich.text     import Text
    from rich.prompt   import Prompt
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
    from rich.live     import Live
    from rich.markdown import Markdown
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("[안내] pip install rich 설치 시 더 예쁜 UI를 볼 수 있습니다.")

console = Console() if HAS_RICH else None


# ── 출력 헬퍼 ─────────────────────────────────────────────────
def cprint(msg, style=""):
    if HAS_RICH: console.print(msg, style=style)
    else: print(msg)

def ask(prompt):
    if HAS_RICH: return Prompt.ask(f"[bold green]{prompt}[/bold green]").strip()
    return input(f"{prompt}: ").strip()

def rule(title=""):
    if HAS_RICH: console.rule(title)
    else: print(f"\n{'─'*55} {title}")


# ── 배너 ──────────────────────────────────────────────────────
def print_banner(gemma_ok: bool, hf_token: str):
    if HAS_RICH:
        status = "[green]● Gemma-4 연결됨[/green]" if gemma_ok else "[dim]○ Gemma-4 미연결[/dim]"
        console.print(Panel(
            f"[bold cyan]🚀 Korea NextDay Surge Predictor v2.0[/bold cyan]\n"
            f"[dim]KOSPI + KOSDAQ │ 앙상블 ML │ 다음날 수익률 예측[/dim]\n"
            f"[dim]모델: {HF_MODEL}[/dim]\n"
            f"{status}",
            border_style="cyan", expand=False
        ))
    else:
        print(f"""
╔══════════════════════════════════════════════════════╗
║  🚀 Korea NextDay Surge Predictor v2.0               ║
║  {'● Gemma-4 연결됨' if gemma_ok else '○ Gemma-4 미연결 (HF 토큰 필요)'}
╚══════════════════════════════════════════════════════╝
""")


def print_menu(gemma_ok: bool):
    if HAS_RICH:
        console.print("""
[bold yellow]──── 메뉴 ────[/bold yellow]
  [1] 종목 검색 (이름/코드)
  [2] 단일 종목 ML + Gemma-4 분석
  [3] TOP10 급등 후보 스캔 + Gemma-4 종합 판단
  [4] 💬 AI 종목 챗봇 (Gemma-4)
  [t] 🔌 Gemma-4 연결 테스트
  [q] 종료
""")
    else:
        print("""
──── 메뉴 ────
  1. 종목 검색
  2. 단일 종목 ML + Gemma-4 분석
  3. TOP10 급등 스캔 + Gemma-4 판단
  4. AI 종목 챗봇
  t. Gemma-4 연결 테스트
  q. 종료
""")


# ── 단일 종목 결과 출력 ───────────────────────────────────────
def print_single(r: dict):
    if r is None:
        cprint("❌ 종목을 찾을 수 없거나 데이터 부족", style="red"); return

    pr   = r["pred_ret_pct"]; cb = r["conf_band"]
    dir_col = {"상승":"green","하락":"red","중립":"yellow"}.get(r["direction"],"white")
    dir_icon = {"상승":"📈","하락":"📉","중립":"➡️"}.get(r["direction"],"")

    if HAS_RICH:
        console.print(Panel(
            f"[bold white]{r['name']}[/bold white]  "
            f"[dim]({r['ticker']} / {r['market']})[/dim]",
            style="blue", expand=False
        ))

        # 예측 결과
        t1 = Table(title="📊 다음 거래일 예측", header_style="bold magenta", show_lines=True)
        t1.add_column("항목",   style="cyan",  width=20)
        t1.add_column("값",     style="white", width=20)
        t1.add_column("비고",   style="dim",   width=35)

        t1.add_row("예측 수익률",
                   f"[{dir_col}]{pr:+.2f}%[/{dir_col}]",
                   f"신뢰구간: {pr-cb:+.2f}% ~ {pr+cb:+.2f}%")
        t1.add_row("예측 상승률",
                   f"[green]+{r['pred_up_pct']:.2f}%[/green]",
                   f"과거 상승 평균: {r['up_hist_avg'] or 'N/A'}%")
        t1.add_row("예측 하락률",
                   f"[red]-{r['pred_dn_pct']:.2f}%[/red]",
                   f"과거 하락 평균: {r['dn_hist_avg'] or 'N/A'}%")
        t1.add_row("15% 급등 확률",
                   f"[yellow]{r['prob_up15']*100:.1f}%[/yellow]",
                   f"과거 발생률 {r['base_rate']*100:.1f}% ({r['pos_count']}회)")
        t1.add_row("방향 판단",
                   f"[{dir_col}]{dir_icon} {r['direction']}[/{dir_col}]", "")
        t1.add_row("급등 점수",
                   f"[bold]{r['score']:.1f} / 100[/bold]", "")
        console.print(t1)

        # 기술적 지표
        t2 = Table(title="🔬 기술적 지표", header_style="bold cyan", show_lines=True)
        t2.add_column("지표",  width=18)
        t2.add_column("값",    width=14)
        t2.add_column("해석",  width=30)

        def rsi_hint(v):
            if v >= 70: return "[red]과매수 주의[/red]"
            if v <= 30: return "[green]과매도 (반등 가능)[/green]"
            return "중립"

        t2.add_row("RSI(14)",          f"{r['rsi14']:.1f}",       rsi_hint(r['rsi14']))
        t2.add_row("BB 위치",           f"{r['bb_pos']:.2f}",      "0=하단 / 1=상단")
        t2.add_row("거래량 비율(20일)", f"{r['vol_ratio20']:.2f}x","[green]폭발[/green]" if r['vol_ratio20']>=3 else "")
        t2.add_row("CCI(14)",           f"{r['cci14']:.1f}",       "[green]+100↑=강세[/green]" if r['cci14']>100 else "[red]-100↓=약세[/red]" if r['cci14']<-100 else "중립")
        t2.add_row("52주 신고가 거리",  f"{r['dist_52w_high']*100:.1f}%","[green]신고가 근접[/green]" if r['near_52w_high'] else "")
        t2.add_row("거래량 폭발",       "[green]✓[/green]" if r['vol_explosion'] else "✗","")
        t2.add_row("눌림목 압축",       "[green]✓[/green]" if r['price_compress'] else "✗","")
        t2.add_row("이평 돌파",         "[green]✓[/green]" if r['breakout_flag'] else "✗","")
        cv = r["cv_auc"]
        t2.add_row("CV AUC",            f"{cv:.3f}" if cv else "N/A","")
        console.print(t2)

    else:
        print(f"\n{'='*55}")
        print(f"  {r['name']} ({r['ticker']} / {r['market']})")
        print(f"{'='*55}")
        print(f"  예측 수익률    : {pr:+.2f}%  (±{cb:.2f}%)")
        print(f"  예측 상승률    : +{r['pred_up_pct']:.2f}%")
        print(f"  예측 하락률    : -{r['pred_dn_pct']:.2f}%")
        print(f"  15% 급등 확률  : {r['prob_up15']*100:.1f}%")
        print(f"  방향 판단      : {dir_icon} {r['direction']}")
        print(f"  급등 점수      : {r['score']:.1f}/100")
        print(f"  RSI14          : {r['rsi14']:.1f}")
        print(f"  거래량 비율    : {r['vol_ratio20']:.2f}x")
        print(f"{'='*55}\n")


# ── TOP10 출력 ────────────────────────────────────────────────
def print_top10(df):
    if df is None or len(df) == 0:
        cprint("❌ 결과 없음", style="red"); return

    if not HAS_RICH:
        import pandas as pd
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 220)
        print(df[["rank","name","market","close","pred_ret_pct",
                   "pred_up_pct","pred_dn_pct","prob_up15","score"]].to_string(index=False))
        return

    t = Table(title="🚀 다음 거래일 15% 급등 예상 TOP10",
              header_style="bold magenta", show_lines=True)
    for col, w in [("순위",5),("종목명",14),("시장",8),("종가",10),
                   ("예측수익률%",12),("상승%",9),("하락%",9),
                   ("15%확률",9),("CV_AUC",8),("급등점수",9),("거래량비율",10)]:
        t.add_column(col, width=w)

    for _, row in df.iterrows():
        pr    = float(row.get("pred_ret_pct", 0) or 0)
        col   = "green" if pr >= 0 else "red"
        cv    = f"{row['cv_auc']:.3f}" if not __import__('pandas').isna(row.get("cv_auc")) else "N/A"
        icons = ("🔥" if row.get("vol_explosion") else "") + \
                ("📌" if row.get("breakout_flag")  else "") + \
                ("⭐" if row.get("near_52w_high")   else "")
        t.add_row(
            str(int(row["rank"])),
            f"{row['name']} {icons}",
            str(row["market"]),
            f"{int(row['close']):,}",
            f"[{col}]{pr:+.2f}%[/{col}]",
            f"[green]+{float(row.get('pred_up_pct',0) or 0):.2f}%[/green]",
            f"[red]-{float(row.get('pred_dn_pct',0) or 0):.2f}%[/red]",
            f"{float(row.get('prob_up15',0) or 0)*100:.1f}%",
            cv,
            f"[bold]{float(row.get('score',0)):.1f}[/bold]",
            f"{float(row.get('vol_ratio20',1) or 1):.1f}x",
        )
    console.print(t)
    cprint("🔥=거래량폭발  📌=이평돌파  ⭐=52주신고가근접", style="dim")


# ── Gemma-4 스트리밍 출력 ─────────────────────────────────────
def stream_gemma(gen, title="🤖 Gemma-4 분석"):
    rule(title)
    full = ""
    try:
        if HAS_RICH:
            with Live(console=console, refresh_per_second=8) as live:
                for chunk in gen:
                    full += chunk
                    live.update(Markdown(full))
        else:
            for chunk in gen:
                full += chunk
                print(chunk, end="", flush=True)
            print()
    except Exception as e:
        cprint(f"\n❌ Gemma-4 오류: {e}", style="red")
    rule()
    return full


# ══════════════════════════════════════════════════════════════
#  메인 루프
# ══════════════════════════════════════════════════════════════
def main():
    # HF 토큰 로드
    hf_token = os.environ.get("HF_TOKEN", "")
    if not hf_token:
        cprint("\n[Gemma-4] HuggingFace API 토큰을 입력하세요.", style="yellow")
        cprint("(엔터 스킵 시 Gemma-4 기능 비활성화)", style="dim")
        hf_token = getpass.getpass("  HF Token (hf_xxx...): ").strip()

    gemma_ok = False
    if hf_token:
        cprint("Gemma-4 연결 테스트 중…", style="dim")
        res = test_connection(hf_token)
        if res["success"]:
            gemma_ok = True
            cprint(f"✅ HF Router 연결 성공: {HF_MODEL}", style="green")
            cprint(f"   응답: {res['reply']}", style="dim")
        else:
            cprint(f"❌ 연결 실패: {res['error']}", style="red")
            cprint("Gemma-4 없이 ML 예측만 동작합니다.", style="yellow")
    else:
        cprint("Gemma-4 토큰 없음 → ML 예측 전용 모드", style="yellow")

    agent = KoreaSurgeAgent(verbose=True)
    chat_history = []

    print_banner(gemma_ok, hf_token)

    while True:
        print_menu(gemma_ok)
        choice = ask("선택").lower()

        # ── 종료 ────────────────────────────────────────────
        if choice in ("q","quit","exit"):
            cprint("종료합니다.", style="yellow"); break

        # ── 연결 테스트 ──────────────────────────────────────
        elif choice == "t":
            if not hf_token:
                hf_token = getpass.getpass("HF Token: ").strip()
            with (console.status("연결 테스트 중…") if HAS_RICH
                  else __import__('contextlib').nullcontext()):
                res = test_connection(hf_token)
            if res["success"]:
                gemma_ok = True
                cprint(f"✅ 연결 성공: {HF_MODEL}", style="green")
                cprint(f"   {res['reply']}", style="dim")
            else:
                cprint(f"❌ 연결 실패: {res['error']}", style="red")

        # ── 종목 검색 ────────────────────────────────────────
        elif choice == "1":
            kw = ask("검색 키워드 (종목명/코드)")
            result = agent.search(kw, top_n=10)
            if HAS_RICH:
                t = Table(title=f"검색 결과: {kw}", header_style="bold cyan")
                for col in result.columns: t.add_column(str(col))
                for _, row in result.iterrows():
                    t.add_row(*[str(v) for v in row])
                console.print(t)
            else:
                print(result.to_string(index=False))

        # ── 단일 종목 분석 ───────────────────────────────────
        elif choice == "2":
            query = ask("종목명 또는 코드")
            cprint(f"'{query}' ML 분석 중… (30~60초)", style="yellow")

            if HAS_RICH:
                with console.status("[bold green]앙상블 모델 예측 중…[/bold green]"):
                    r = agent.analyze(query, use_realtime=True)
            else:
                r = agent.analyze(query, use_realtime=True)

            print_single(r)

            # Gemma-4 분석
            if r and gemma_ok:
                use_g = ask("Gemma-4 AI 분석 실행? (y/n)").lower()
                if use_g == "y":
                    gen = analyze_stock_with_gemma(r, hf_token, stream=True)
                    stream_gemma(gen, f"🤖 Gemma-4 분석: {r['name']}")

        # ── TOP10 스캔 ───────────────────────────────────────
        elif choice == "3":
            try:
                n = int(ask("스캔 종목 수 (추천 100~300)"))
            except ValueError:
                n = 100

            cprint(f"{n}개 종목 스캔 시작…", style="yellow")

            if HAS_RICH:
                with Progress(SpinnerColumn(),
                              TextColumn("[progress.description]{task.description}"),
                              BarColumn(),
                              TextColumn("{task.completed}/{task.total}"),
                              console=console) as prog:
                    task = prog.add_task("스캔 중…", total=n)
                    def cb(i, total, name):
                        prog.update(task, completed=i, description=f"[cyan]{name}[/cyan]")
                    df = agent.scan_top(max_stocks=n, top_n=10,
                                        use_realtime=True, progress_cb=cb)
            else:
                df = agent.scan_top(max_stocks=n, top_n=10, use_realtime=True)

            print_top10(df)

            # Gemma-4 종합 판단
            if df is not None and gemma_ok:
                use_g = ask("Gemma-4 종합 시장 판단 실행? (y/n)").lower()
                if use_g == "y":
                    gen = analyze_top10_with_gemma(df, hf_token, stream=True)
                    stream_gemma(gen, "🤖 Gemma-4 TOP10 종합 판단")

        # ── AI 챗봇 ─────────────────────────────────────────
        elif choice == "4":
            if not gemma_ok:
                cprint("❌ Gemma-4 미연결. 't'로 연결 테스트를 먼저 하세요.", style="red")
                continue

            cprint("\n💬 Gemma-4 주식 AI 챗봇 (종료: 'exit')", style="cyan")
            cprint("빠른 질문 예시:", style="dim")
            cprint("  · 이 종목 내일 사도 될까요?", style="dim")
            cprint("  · RSI 70이면 어떻게 봐야 하나요?", style="dim")
            cprint("  · 볼린저밴드 상단 돌파 의미는?", style="dim")

            while True:
                q = ask("질문")
                if q.lower() in ("exit","quit","back","q"):
                    break
                if not q:
                    continue

                chat_history.append({"role":"user","content":q})
                gen = chat_with_gemma(
                    question = q,
                    hf_token = hf_token,
                    history  = chat_history[-6:],
                    context  = None,
                    stream   = True,
                )
                reply = stream_gemma(gen, "🤖 Gemma-4 답변")
                chat_history.append({"role":"assistant","content":reply})

        else:
            cprint("잘못된 입력입니다. 1~4 또는 t/q", style="red")


if __name__ == "__main__":
    main()
