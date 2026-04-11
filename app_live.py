"""One-button live prediction app.

Run:
    streamlit run app_live.py

Design principles:
  1. ONE button does everything — fetch the race list, run the fact
     pipeline on every race of the day, and display a consolidated
     analysis. No separate "fetch list" + "predict" flow.
  2. Per-race drill-down lives in collapsed expanders so the default
     view is the action summary (loose bets across all races).
  3. No external-API connection UI (no key prompts, no status checks).
  4. Strict and Loose layers are displayed side-by-side but kept
     operationally independent (per docs/trading_constitution.md).

Governed by docs/trading_constitution.md §2-3.
"""

from __future__ import annotations

import time
from datetime import date

import streamlit as st
import pandas as pd

import scraper
import live_pipeline as lp
import prediction_log as plog


st.set_page_config(
    page_title="理論予想 Live — 本日の自動分析",
    page_icon="🎯",
    layout="wide",
)

st.title("🎯 理論予想 Live")
st.caption(
    "JRA + netkeiba + KeibaLab + Hochi + Sanspo + Daily (live) + oikiri "
    "editorial → fact merge → dual-mode scoring → LOOSE betting rule"
)


# ── Sidebar: KPI dashboards ──────────────────────────

with st.sidebar:
    kpis = plog.compute_kpis()

    st.header("📊 Strict Trigger KPI")
    st.caption("fact-mode confident override")
    st.metric("ログ済みレース", kpis["n_races"])
    st.metric("結果付きレース", kpis["n_races_with_result"])
    st.metric(
        "triggers / 50 races",
        f"{kpis['triggers_per_50_races']:.2f}",
        delta="✅ ≥8" if kpis["triggers_per_50_races"] >= 8 else "🟡",
    )
    st.metric(
        "trigger 勝率",
        f"{kpis['trigger_win_rate']*100:.1f}%"
        if kpis["triggers_with_result"] else "—",
    )
    st.metric(
        "trigger ROI",
        f"{kpis['trigger_roi']*100:+.1f}%"
        if kpis["triggers_with_result"] else "—",
    )

    st.divider()

    st.header("🧪 Loose Bet KPI")
    st.caption("experimental betting rule")
    st.caption("cons≥1 · comp≥0.60 · odds≤15 · no strong-neg")
    st.metric("loose bet 総数", kpis["loose_bets_total"])
    st.metric("loose bet レース", kpis["loose_bet_races_total"])
    if kpis["loose_bets_with_result"]:
        st.metric(
            "loose 勝率",
            f"{kpis['loose_bet_win_rate']*100:.1f}%",
            delta=f"{(kpis['loose_bet_win_rate']-kpis['odds_favorite_win_rate'])*100:+.1f}pp vs odds-fav",
        )
        st.metric(
            "loose ROI",
            f"{kpis['loose_bet_roi']*100:+.1f}%",
            delta=f"{kpis['loose_vs_odds_roi_delta']*100:+.2f}pp vs odds-fav",
        )
        st.caption(
            f"cost {kpis['loose_bet_cost_yen']:.0f}¥ · "
            f"payout {kpis['loose_bet_payout_yen']:.0f}¥ · "
            f"pnl {kpis['loose_bet_pnl_yen']:+.0f}¥"
        )
    else:
        st.caption("結果付き loose bet がまだありません。")

    st.divider()
    st.caption(
        f"Baseline — odds-fav {kpis.get('odds_favorite_roi', 0)*100:+.1f}% · "
        f"model-top {kpis.get('model_top_roi', 0)*100:+.1f}%"
    )
    st.caption(f"Rule: `{kpis.get('loose_rule_version', 'v1')}`")


# ── Main: one button does everything ─────────────────

col_date, col_mode = st.columns([2, 1])
with col_date:
    race_date = st.date_input("レース日", value=date.today())
is_live = race_date == date.today()
with col_mode:
    st.markdown(
        f"### {'🟢 LIVE' if is_live else '🟡 BACKFILL'}",
        help="LIVE = today. Newspaper scrapers enabled. "
             "BACKFILL = past date — newspaper scrapers skipped automatically.",
    )

st.markdown("")

# THE button
run = st.button(
    "🚀 本日のレースを自動分析",
    type="primary",
    use_container_width=True,
    help="レース一覧の取得から全レースの理論予想まで自動で実行します。",
)

# Keep analysis results in session state so the page stays responsive
# while the user drills into individual races.
if "auto_analysis" not in st.session_state:
    st.session_state.auto_analysis = None


def _run_auto_analysis(race_date_iso: str) -> dict:
    """Fetch race list and run predict_live on every race. Returns a
    batch dict with per-race results and aggregated metrics."""
    started_at = time.time()
    # 1. Fetch the day's race list
    races = scraper.fetch_race_list(race_date) or []
    if not races:
        return {
            "race_date": race_date_iso,
            "is_live": is_live,
            "races_found": 0,
            "results": [],
            "elapsed_s": round(time.time() - started_at, 1),
            "error": "この日付のレースが見つかりませんでした。",
        }

    # 2. Run predict_live on every race sequentially
    results = []
    prog_bar = st.progress(0.0, text=f"0/{len(races)} 予想中...")
    status_line = st.empty()
    for i, race in enumerate(races):
        status_line.info(f"⏳ [{i+1}/{len(races)}] {race.get('race_name','?')} を分析中...")
        try:
            r = lp.predict_live(
                race_id=race.get("race_id", ""),
                venue=race.get("venue", ""),
                race_name=race.get("race_name", ""),
                race_date=race_date_iso,
                progress_cb=None,  # silence per-step spam; the bar is enough
                auto_log=True,
            )
            results.append(r)
        except Exception as e:
            results.append({
                "race_id": race.get("race_id", ""),
                "race_name": race.get("race_name", ""),
                "error": str(e),
            })
        prog_bar.progress((i + 1) / len(races),
                          text=f"{i+1}/{len(races)} 予想中...")
    status_line.empty()
    prog_bar.empty()

    return {
        "race_date": race_date_iso,
        "is_live": is_live,
        "races_found": len(races),
        "results": results,
        "elapsed_s": round(time.time() - started_at, 1),
        "error": None,
    }


if run:
    with st.spinner(f"{race_date} の全レースを自動分析中..."):
        st.session_state.auto_analysis = _run_auto_analysis(race_date.isoformat())


# ── Render auto-analysis output ──────────────────────

batch = st.session_state.auto_analysis

if batch and batch.get("error"):
    st.warning(batch["error"])
elif batch and batch["races_found"] == 0:
    st.info("レースが見つかりませんでした。別の日付を試してください。")
elif batch and batch["results"]:
    results = batch["results"]
    ok_results = [r for r in results if not r.get("error")]

    # Aggregate counts
    total_loose = sum(len(r.get("loose_bets", [])) for r in ok_results)
    total_strict = sum(len(r.get("triggers", [])) for r in ok_results)
    n_races_with_loose = sum(1 for r in ok_results if r.get("loose_bets"))

    # ── Batch summary banner ──
    st.divider()
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("分析レース数", f"{len(ok_results)} / {batch['races_found']}")
    b2.metric("🧪 LOOSE bet 候補", total_loose)
    b3.metric("レース (loose あり)", n_races_with_loose)
    b4.metric("🔥 Strict trigger", total_strict)
    b5.metric("所要時間", f"{batch['elapsed_s']:.0f}s")

    # ── Odds-status health warning ──
    # If a race had no odds on the shutuba page and injection failed,
    # all horses end up with score_runner's uniform 1/N base and the
    # ranking collapses to gate-number order. Surface this clearly.
    odds_issues = [
        r for r in ok_results
        if str(r.get("odds_status", "ok")).startswith("all-zero")
    ]
    odds_recovered = [
        r for r in ok_results
        if str(r.get("odds_status", "ok")).startswith("injected")
    ]
    if odds_issues:
        st.error(
            f"⚠️ {len(odds_issues)} レースでオッズ情報が取得できませんでした。"
            f"該当レースのランキングは **信頼できません** "
            f"(全馬ほぼ同一スコア → 枠順で表示される可能性)。"
            f"該当: {', '.join(r.get('race_name','?') for r in odds_issues[:5])}"
        )
    elif odds_recovered:
        st.info(
            f"ℹ️ {len(odds_recovered)} レースで shutuba ページにオッズが無かったため、"
            f"結果ページ / ライブオッズページから自動補完しました。"
        )

    # ── Consolidated LOOSE bets (the actionable table) ──
    st.markdown("### 🧪 本日の Loose Bet 候補（全レース統合）")
    st.caption(
        "constitution §3: cons≥1 AND comp≥0.60 AND odds≤15 AND no strong-neg. "
        "100-yen flat per bet."
    )

    loose_rows = []
    for r in ok_results:
        for lb in r.get("loose_bets", []):
            loose_rows.append({
                "レース": r.get("race_name", "")[:14],
                "G": r.get("grade", ""),
                "馬名": lb["name"],
                "オッズ": f"{lb.get('odds', 0):.1f}",
                "consensus": lb["consensus_count"],
                "composite": f"{lb['composite_condition']:.2f}",
                "sources": lb["source_count"],
                "cond": f"{lb.get('condition_score', 0.5):.2f}",
                "fat": f"{lb.get('fatigue_score', 0.0):.2f}",
                "str": f"{lb.get('stress_score', 0.0):.2f}",
                "reason": lb.get("loose_trigger_reason", ""),
            })
    if loose_rows:
        st.dataframe(pd.DataFrame(loose_rows),
                     use_container_width=True, hide_index=True)
        st.success(
            f"💰 **{total_loose}** horses across **{n_races_with_loose}** races "
            f"pass the loose rule. Expected cost: **{total_loose * 100:,}¥** "
            f"(100¥ flat per bet)."
        )
    else:
        st.info(
            "本日は loose bet 候補がありませんでした。"
            "constitution §1.3 — 勝てるレースのみ選択する。"
        )

    # ── Strict triggers (audit only) ──
    if total_strict:
        st.markdown("### 🔥 Strict Trigger 馬（監査用・投資には使用しない）")
        strict_rows = []
        for r in ok_results:
            for t in r.get("triggers", []):
                strict_rows.append({
                    "レース": r.get("race_name", "")[:14],
                    "G": r.get("grade", ""),
                    "馬名": t["name"],
                    "オッズ": f"{t.get('odds', 0):.1f}",
                    "consensus": t["consensus_count"],
                    "composite": f"{t['composite_condition']:.2f}",
                    "reason": t.get("reason", ""),
                })
        if strict_rows:
            st.dataframe(pd.DataFrame(strict_rows),
                         use_container_width=True, hide_index=True)

    # ── Per-race drill-down ──
    st.markdown("### 📋 レース別詳細（クリックで展開）")
    for r in results:
        if r.get("error"):
            with st.expander(f"❌ {r.get('race_name','?')} — エラー"):
                st.error(r["error"])
            continue

        n_loose = len(r.get("loose_bets", []))
        n_strict = len(r.get("triggers", []))
        badge_parts = []
        if n_loose: badge_parts.append(f"🧪 {n_loose}")
        if n_strict: badge_parts.append(f"🔥 {n_strict}")
        badge = " · ".join(badge_parts) if badge_parts else "–"

        with st.expander(
            f"{r.get('grade','?')} {r.get('race_name','?')} "
            f"({r.get('venue','?')}) — {badge}"
        ):
            # Top metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("P1 (winner ∈ Top3)", f"{r.get('p1',0)*100:.1f}%")
            m2.metric("P2 (1-2 ∈ Top3)", f"{r.get('p2',0)*100:.1f}%")
            m3.metric("取消・除外", len(r.get("scratched", [])))
            odds_st = r.get("odds_status", "ok")
            m4.metric(
                "odds 状態",
                "✅" if odds_st == "ok" else ("🔵" if "injected" in odds_st else "⚠"),
                help=odds_st,
            )

            # Collection-source badges
            src_badges = []
            for rec in r.get("collection_log", []):
                icon = {
                    "ok": "✅", "partial": "🟡",
                    "failed": "❌", "skipped": "⏭",
                }.get(rec.get("status", "?"), "❓")
                n_f = r["source_counts"].get(rec["source"], 0)
                src_badges.append(f"{icon} {rec['source']}({n_f})")
            if src_badges:
                st.caption("ソース: " + " / ".join(src_badges))

            # Loose bets for this race
            if r.get("loose_bets"):
                st.markdown("**🧪 Loose bets**")
                ldf = pd.DataFrame([
                    {
                        "馬名": lb["name"],
                        "オッズ": f"{lb.get('odds', 0):.1f}",
                        "consensus": lb["consensus_count"],
                        "composite": f"{lb['composite_condition']:.2f}",
                        "sources": lb["source_count"],
                        "reason": lb.get("loose_trigger_reason", ""),
                    }
                    for lb in r["loose_bets"]
                ])
                st.dataframe(ldf, use_container_width=True, hide_index=True)

            # Strict triggers for this race
            if r.get("triggers"):
                st.markdown("**🔥 Strict triggers (audit)**")
                tdf = pd.DataFrame([
                    {
                        "馬名": t["name"],
                        "オッズ": f"{t.get('odds', 0):.1f}",
                        "consensus": t["consensus_count"],
                        "composite": f"{t['composite_condition']:.2f}",
                        "reason": t.get("reason", ""),
                    }
                    for t in r["triggers"]
                ])
                st.dataframe(tdf, use_container_width=True, hide_index=True)

            # Selected top-3
            if r.get("selected_top3"):
                st.markdown("**🥇 選抜 Top-3**")
                sel = pd.DataFrame([
                    {
                        "順位": i + 1,
                        "馬名": h["name"],
                        "オッズ": f"{h['odds']:.1f}" if h["odds"] else "-",
                        "勝率": f"{h['win_prob']*100:.1f}%",
                    }
                    for i, h in enumerate(r["selected_top3"])
                ])
                st.dataframe(sel, use_container_width=True, hide_index=True)

            # Full ranking (first 8)
            if r.get("ranked"):
                st.markdown("**📊 全頭ランキング (上位8)**")
                fdf = pd.DataFrame([
                    {
                        "順位": i + 1,
                        "馬名": h["name"],
                        "オッズ": f"{h['odds']:.1f}" if h["odds"] else "-",
                        "勝率": f"{h['win_prob']*100:.1f}%",
                        "mode": h.get("mode", "odds"),
                    }
                    for i, h in enumerate(r["ranked"][:8])
                ])
                st.dataframe(fdf, use_container_width=True, hide_index=True)

    # ── Post-race: attach results button ──
    st.divider()
    st.markdown("### 📥 結果を取得して KPI を更新")
    st.caption("レース終了後に押してください。各レースの結果を一括取得します。")
    if st.button("全レースの結果を取得", use_container_width=True):
        updated = 0
        prog = st.progress(0.0)
        status = st.empty()
        for i, r in enumerate(ok_results):
            rid = r.get("race_id", "")
            status.info(f"⏳ [{i+1}/{len(ok_results)}] {r.get('race_name','?')} の結果を取得中...")
            try:
                res = scraper.fetch_result_netkeiba(rid)
                if res and res.get("finishing_order"):
                    plog.attach_result(rid, res)
                    updated += 1
            except Exception:
                pass
            prog.progress((i + 1) / len(ok_results))
        status.empty()
        prog.empty()
        if updated:
            st.success(f"{updated} レースの結果を記録しました — サイドバー KPI を更新")
            st.rerun()
        else:
            st.warning("結果が取得できるレースがありませんでした (未開催?)")


# ── Footer: history tables ───────────────────────────

st.divider()
st.markdown("### 📜 最近の Loose Bet 履歴 (実験的)")
loose_history = plog.recent_loose_bets_table(limit=20)
if loose_history:
    ldf = pd.DataFrame(loose_history)
    preferred = [
        "race_date", "race_name", "horse", "odds",
        "loose_trigger_reason", "outcome", "payout", "pnl",
    ]
    cols = [c for c in preferred if c in ldf.columns] + \
           [c for c in ldf.columns if c not in preferred]
    st.dataframe(ldf[cols], use_container_width=True, hide_index=True)
else:
    st.caption("まだ loose bet の記録がありません。")

with st.expander("📜 最近の Strict Trigger 履歴 (監査用)"):
    history = plog.recent_trigger_table(limit=20)
    if history:
        st.dataframe(pd.DataFrame(history),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("まだ strict trigger の記録がありません。")

st.caption(
    "このアプリは `docs/trading_constitution.md` に従って動作します。"
    "「ルールを守れるかどうかが、勝てるかどうか」"
)
