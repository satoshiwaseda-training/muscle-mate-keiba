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

import datetime as dt
import time
from datetime import date

import streamlit as st
import pandas as pd

import scraper
import live_pipeline as lp
import prediction_log as plog
from tools._autolog_utils import last_weekend


st.set_page_config(
    page_title="理論予想 Live — G1/G2 自動分析",
    page_icon="🎯",
    layout="wide",
)

# Keep this source-of-truth display in sync with scraper.LIVE_GRADE_FILTER.
_filter = scraper.LIVE_GRADE_FILTER
_filter_label = "ALL races" if _filter is None else " + ".join(_filter)

st.title(f"🎯 理論予想 Live — {_filter_label}")
st.caption(
    f"対象: **{_filter_label} のみ** (scraper.LIVE_GRADE_FILTER). "
    f"G3 / 非グレードレースは現在処理対象外。"
)
st.caption(
    "JRA + netkeiba + KeibaLab + Hochi + Sanspo + Daily (live) + oikiri "
    "editorial → fact merge → dual-mode scoring → LOOSE betting rule"
)


# ── Weekly data-health banner (TOP-OF-PAGE, highest priority) ──
# The single most important question is "does yesterday / this weekend's
# data actually exist?". Everything else is meaningless if the answer is
# no, so we surface the volume counts above anything else.

def _weekly_status() -> dict:
    sat, sun = last_weekend(dt.date.today())
    all_preds = plog.list_predictions(only_live=True)
    sat_iso, sun_iso = sat.isoformat(), sun.isoformat()
    scoped = [
        e for e in all_preds
        if sat_iso <= (e.get("race_date") or "") <= sun_iso
    ]
    with_result = [e for e in scoped if (e.get("result") or {}).get("finishing_order")]
    wins, cost, payout = 0, 0.0, 0.0
    for e in with_result:
        ranked = e.get("ranked") or []
        if not ranked:
            continue
        top1 = (ranked[0].get("name") or "").strip()
        res = e.get("result") or {}
        fo = res.get("finishing_order") or []
        winner = None
        for h in fo:
            try:
                if int(h.get("rank", 0) or 0) == 1:
                    winner = (h.get("name") or "").strip()
                    break
            except ValueError:
                pass
        if not winner:
            continue
        cost += 100.0
        if top1 == winner:
            wins += 1
            try:
                payout += float(str((res.get("payouts") or {}).get("単勝", 0)).replace(",", ""))
            except Exception:
                pass
    return {
        "sat": sat_iso,
        "sun": sun_iso,
        "n_preds": len(scoped),
        "n_with_result": len(with_result),
        "win_rate": (wins / len(with_result)) if with_result else 0.0,
        "roi": ((payout - cost) / cost) if cost > 0 else 0.0,
    }


_ws = _weekly_status()
w1, w2, w3, w4 = st.columns(4)
w1.metric(
    f"今週の予測数 ({_ws['sat']}～{_ws['sun']})",
    _ws["n_preds"],
    help="最重要: 0 の場合 weekend_autolog が動いていない",
)
w2.metric(
    "今週の結果付与数",
    _ws["n_with_result"],
    help="attach_results.py が結果を紐付けた件数",
)
w3.metric(
    "今週勝率",
    f"{_ws['win_rate']*100:.1f}%" if _ws["n_with_result"] else "—",
)
w4.metric(
    "今週ROI",
    f"{_ws['roi']*100:+.1f}%" if _ws["n_with_result"] else "—",
)
if _ws["n_preds"] == 0:
    st.error(
        f"⚠️ 今週 ({_ws['sat']}～{_ws['sun']}) の予測ログがゼロ件です。"
        f"`python tools/weekend_autolog.py` が動いていない可能性があります。"
        f"cron / Task Scheduler の状態と data/autolog/ を確認してください。"
    )
elif _ws["n_with_result"] == 0 and _ws["n_preds"] > 0:
    st.warning(
        f"📋 今週 {_ws['n_preds']} 件の予測があるが結果が未付与。"
        f"`python tools/attach_results.py` を実行してください。"
    )

st.divider()


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
    f"🚀 本日の {_filter_label} を自動分析",
    type="primary",
    width="stretch",
    help=f"レース一覧の取得から {_filter_label} の理論予想まで自動で実行します。"
         f"G3 / 非グレードレースはスキップされます。",
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
            "error": f"この日付の {_filter_label} レースが見つかりませんでした。",
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
            # Carry race_time from the race-list dict into the result so the
            # UI can show "発走まで Nh" for odds-not-published warnings.
            r["race_time"] = race.get("time", "")
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
    st.info(
        f"{_filter_label} レースが見つかりませんでした。"
        f"G3 や平場レースは `scraper.LIVE_GRADE_FILTER` で意図的に除外しています。"
    )
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
    # We now distinguish THREE states:
    #   1. not-published-yet  — netkeiba has no odds yet (>3h before post).
    #                           This is NORMAL and not a bug — tell the
    #                           user to re-run closer to post time.
    #   2. all-zero-no-source — genuine scraper failure (network / layout
    #                           change / netkeiba block). Ranking is junk.
    #   3. injected            — recovered from live-odds API or result page.
    odds_not_pub = [
        r for r in ok_results
        if str(r.get("odds_status", "ok")).startswith("not-published")
    ]
    odds_failed = [
        r for r in ok_results
        if str(r.get("odds_status", "ok")).startswith("all-zero")
    ]
    odds_recovered = [
        r for r in ok_results
        if str(r.get("odds_status", "ok")).startswith("injected")
    ]

    if odds_not_pub:
        # Compute re-fetch window for each affected race.
        # netkeiba publishes odds roughly 2-3 hours before post. We
        # suggest the window [post - 3h, post - 2h] to give the user
        # a concrete target time instead of "wait and retry".
        now = dt.datetime.now()
        fetched_display = now.strftime("%H:%M")
        rows = []
        for r in odds_not_pub:
            rt = r.get("race_time") or r.get("time") or ""
            api_meta = r.get("odds_api_meta") or {}
            schema = api_meta.get("api_schema_version") or "-"
            http = api_meta.get("api_http_status") or "-"
            stage = r.get("prediction_stage", "early")
            try:
                hh, mm = rt.split(":")
                post = dt.datetime.combine(
                    dt.date.today(), dt.time(int(hh), int(mm))
                )
                delta = post - now
                total_min = int(delta.total_seconds() // 60)
                hrs, mins = divmod(max(0, total_min), 60)
                # Recommended re-fetch: between post-3h and post-2h
                rec_start = (post - dt.timedelta(hours=3)).strftime("%H:%M")
                rec_end   = (post - dt.timedelta(hours=2)).strftime("%H:%M")
                wait_str = f"発走まで {hrs}h{mins:02d}m"
                rec_str  = f"{rec_start}〜{rec_end}"
            except Exception:
                wait_str = "-"
                rec_str  = "-"
            rows.append({
                "レース":      r.get("race_name", "?"),
                "発走":       rt or "-",
                "現状":       wait_str,
                "推奨再取得":  rec_str,
                "stage":     stage,
                "schema":    schema,
                "http":      http,
            })

        st.warning(
            f"🕒 **{len(odds_not_pub)} レースでオッズ未公開** "
            f"(`status: middle` · 取得時刻 {fetched_display}) — "
            f"**スクレイパーのバグではありません**。"
            f"netkeiba が発走 2〜3 時間前にならないとオッズを出さないためです。"
            f"現在の予測は **early version** として記録されていますが、"
            f"ROI 集計からは分離されます (history に保全)。"
        )
        st.dataframe(pd.DataFrame(rows),
                     width="stretch", hide_index=True)
        st.caption(
            f"推奨再取得時刻になったら `🚀 本日の {_filter_label} を自動分析` を再度押してください。"
            f"同じ race_id は上書きされますが、過去版は `history` に保全されます。"
        )
    if odds_failed:
        st.error(
            f"⚠️ {len(odds_failed)} レースでオッズ情報の取得に失敗しました "
            f"(scraper error / netkeiba block)。"
            f"該当レースのランキングは **信頼できません** "
            f"(全馬ほぼ同一スコア → 枠順で表示される可能性)。"
            f"該当: {', '.join(r.get('race_name','?') for r in odds_failed[:5])}"
        )
    if odds_recovered and not odds_not_pub:
        sources = set()
        for r in odds_recovered:
            s = str(r.get("odds_status", ""))
            if "result" in s:
                sources.add("結果ページ")
            elif "live-odds" in s:
                sources.add("ライブオッズAPI")
            elif "sp-shutuba" in s:
                sources.add("SP版出馬表")
            elif "odds-page" in s:
                sources.add("オッズページ")
            elif "yahoo" in s:
                sources.add("Yahoo Sports")
            elif "netkeiba-alt" in s:
                sources.add("netkeiba (代替)")
            else:
                sources.add("フォールバック")
        src_str = " / ".join(sorted(sources)) or "フォールバック"
        st.info(
            f"ℹ️ {len(odds_recovered)} レースで shutuba ページにオッズが無かったため、"
            f"**{src_str}** から自動補完しました。"
        )

    # ── Calibration warnings ──
    # When the calibrated probability layer detects a suspiciously
    # concentrated distribution (e.g. a 1.7-odds horse getting >85%),
    # flag it. Under normal conditions with k=0.8 this should never fire.
    cal_issues = [
        (r, r.get("calibration_warnings", []))
        for r in ok_results
        if r.get("calibration_warnings")
    ]
    if cal_issues:
        with st.expander(
            f"⚠️ キャリブレーション警告 ({len(cal_issues)} レース)",
            expanded=False,
        ):
            for r, warnings in cal_issues:
                st.markdown(f"**{r.get('race_name','?')}**")
                for w in warnings:
                    st.caption(f"• {w}")

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
                     width="stretch", hide_index=True)
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
                         width="stretch", hide_index=True)

    # ── Per-race drill-down ──
    st.markdown("### 📋 レース別詳細（クリックで展開）")
    for r in results:
        if r.get("error"):
            with st.expander(f"❌ {r.get('race_name','?')} — エラー"):
                st.error(r["error"])
            continue

        n_loose = len(r.get("loose_bets", []))
        n_strict = len(r.get("triggers", []))
        stage = r.get("prediction_stage", "final")
        stage_icon = "🟢" if stage == "final" else "🟡"
        badge_parts = [f"{stage_icon} {stage}"]
        if n_loose: badge_parts.append(f"🧪 {n_loose}")
        if n_strict: badge_parts.append(f"🔥 {n_strict}")
        badge = " · ".join(badge_parts)

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

            # ── Audit metadata strip (stage / odds / schema / version) ──
            api_meta = r.get("odds_api_meta") or {}
            meta_row = (
                f"**stage:** `{stage}`  ·  "
                f"**odds_status:** `{odds_st}`  ·  "
                f"**odds_source:** `{api_meta.get('odds_source') or '-'}`  ·  "
                f"**fetched:** `{api_meta.get('api_fetched_at') or '-'}`  ·  "
                f"**schema:** `{api_meta.get('api_schema_version') or '-'}`  ·  "
                f"**http:** `{api_meta.get('api_http_status') or '-'}`  ·  "
                f"**data_source_version:** `{r.get('data_source_version') or '-'}`"
            )
            st.caption(meta_row)
            hist = r.get("history") or []
            if hist:
                st.caption(
                    f"📜 history: {len(hist)} 版保全 · "
                    f"初回 {r.get('first_predicted_at') or '-'} → "
                    f"今 {r.get('prediction_created_at') or '-'}"
                )
            if stage == "early":
                rt = r.get("race_time") or ""
                rec_hint = ""
                if rt and ":" in rt:
                    try:
                        hh, mm = rt.split(":")
                        post = dt.datetime.combine(
                            dt.date.today(), dt.time(int(hh), int(mm))
                        )
                        rec_start = (post - dt.timedelta(hours=3)).strftime("%H:%M")
                        rec_end   = (post - dt.timedelta(hours=2)).strftime("%H:%M")
                        rec_hint = f"推奨再取得: {rec_start}〜{rec_end}"
                    except Exception:
                        pass
                st.info(
                    f"🕒 この予測は **early version** です。"
                    f"オッズ未取得のため ROI 集計から分離されます。"
                    f"{rec_hint}"
                )

            # Core-model reconnection diagnostic
            bridge_d = r.get("bridge_diag", {})
            if bridge_d:
                post = bridge_d.get("post_non_zero", {})
                total = post.get("total_horses", 1) or 1
                channels = [
                    ("jockey_win_rate", "jockey"),
                    ("training_acceleration", "training"),
                    ("training_cardio_index", "cardio"),
                    ("paddock_vascularity", "bio_v"),
                    ("paddock_hindquarter", "bio_h"),
                    ("paddock_gait", "bio_g"),
                    ("horse_weight_delta", "weight"),
                ]
                channel_str = " · ".join(
                    f"{label}={post.get(key, 0)}/{total}"
                    for key, label in channels
                )
                st.caption(f"🔗 core-model channels: {channel_str}")

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
                st.dataframe(ldf, width="stretch", hide_index=True)

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
                st.dataframe(tdf, width="stretch", hide_index=True)

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
                st.dataframe(sel, width="stretch", hide_index=True)

            # Full ranking with calibration breakdown
            if r.get("ranked"):
                st.markdown(
                    "**📊 全頭ランキング (上位8) — 市場 × 構造化モデル**"
                )
                fdf = pd.DataFrame([
                    {
                        "順位": i + 1,
                        "馬名": h["name"],
                        "オッズ": f"{h['odds']:.1f}" if h.get("odds") else "-",
                        "市場勝率": (
                            f"{h.get('base_market_prob', 0)*100:.1f}%"
                            if h.get("base_market_prob") is not None else "—"
                        ),
                        "構造化 edge": (
                            f"{h.get('structured_edge', 0):+.3f}"
                            if h.get("structured_edge") is not None else "—"
                        ),
                        "× multiplier": (
                            f"{h.get('fact_multiplier', 1.0):.2f}"
                            if h.get("fact_multiplier") is not None else "—"
                        ),
                        "最終勝率": f"{h.get('win_prob', 0)*100:.1f}%",
                        "score_runner": f"{h.get('odds_score', 0):.1f}",
                        "mode": h.get("mode", "odds"),
                    }
                    for i, h in enumerate(r["ranked"][:8])
                ])
                st.dataframe(fdf, width="stretch", hide_index=True)
                st.caption(
                    f"構造化 edge = score_runner output − (1/odds)/1.20  "
                    f"|  最終勝率 = 市場勝率 × exp(k × edge), k = "
                    f"{r.get('calibration_k', 1.5):.1f}  |  "
                    f"mode: fact-override = strict trigger fired"
                )

    # ── Post-race: attach results button ──
    st.divider()
    st.markdown("### 📥 結果を取得して KPI を更新")
    st.caption("レース終了後に押してください。各レースの結果を一括取得します。")
    if st.button("全レースの結果を取得", width="stretch"):
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
    st.dataframe(ldf[cols], width="stretch", hide_index=True)
else:
    st.caption("まだ loose bet の記録がありません。")

with st.expander("📜 最近の Strict Trigger 履歴 (監査用)"):
    history = plog.recent_trigger_table(limit=20)
    if history:
        st.dataframe(pd.DataFrame(history),
                     width="stretch", hide_index=True)
    else:
        st.caption("まだ strict trigger の記録がありません。")

st.caption(
    "このアプリは `docs/trading_constitution.md` に従って動作します。"
    "「ルールを守れるかどうかが、勝てるかどうか」"
)
