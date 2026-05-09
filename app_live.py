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

import importlib
import scraper
import live_pipeline as lp
import prediction_log as plog
import probability_engine as pe
import core_model_bridge as bridge
import train
import dual_mode_scoring as dm
import feature_store as fs

# v5.0 (2026-04-19 scratch rewrite): new dedicated modules.
import odds_fetcher
import entries_fetcher
# v5.7 (2026-04-19): grade-specific strategies (G2 diversified).
import grade_strategy
# v5.9 (2026-04-29): Gist persistence (Streamlit Cloud ephemeral fs 対策).
import github_sync

# Force-reload all project modules on every Streamlit rerun so that
# code changes on disk take effect immediately without restarting
# the Streamlit server. Python caches imports in sys.modules and
# Streamlit's file watcher only re-executes the script — it does NOT
# re-import dependency modules. This caused stale HTML-scraping code
# to persist even after the JSON API fix was deployed.
for _mod in [scraper, fs, bridge, train, dm, pe,
             odds_fetcher, entries_fetcher,   # v5.0 scratch modules
             grade_strategy,                    # v5.7
             github_sync,                       # v5.9
             lp, plog]:
    importlib.reload(_mod)

# ── v5.9: Streamlit Cloud で live_predictions.json が無ければ Gist から pull ──
# Streamlit Cloud は filesystem ephemeral なので、リブート後は予測ログが
# 消える。アプリ起動時に Gist に保存された前回ログを自動復元する。
import json as _json
from pathlib import Path as _Path
_LIVE_LOG_FILE = _Path("data/live_predictions.json")
if not _LIVE_LOG_FILE.exists() and github_sync._available():
    try:
        _gist_data = github_sync.pull_file("live_predictions.json")
        if isinstance(_gist_data, dict) and _gist_data:
            _LIVE_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            _LIVE_LOG_FILE.write_text(
                _json.dumps(_gist_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception:
        pass  # Fallback to empty log; no startup blocker

from tools._autolog_utils import last_weekend


st.set_page_config(
    page_title="理論予想 Live — G1/G2 専用分析",
    page_icon="🎯",
    layout="wide",
)

# ── 🏷️ DEPLOY VERSION BANNER (TOP) ─────────────────────────────
# 4/26 週末: v5.8 push 済だがユーザ UI で version 確認できず、
# Streamlit Cloud が古いコードのまま動いていた可能性 (deploy 漏れ事件)。
# 以後、deploy 状態を一目で確認できるようサイドバー先頭に
# DATA_SOURCE_VERSION を **大きく** 常時表示する。
try:
    _running_version = lp.DATA_SOURCE_VERSION
except Exception:
    _running_version = "UNKNOWN"

# Parse "live-v5.9-..." as (major, minor) and check >= (5, 8).
# 旧ロジックは "v58" を探していたが正しくは "v5.8" 形式 (dot あり)。
import re as _re_ver
_m_ver = _re_ver.search(r"v(\d+)\.(\d+)", _running_version or "")
if _m_ver:
    _major = int(_m_ver.group(1))
    _minor = int(_m_ver.group(2))
    _is_v58_plus = _major > 5 or (_major == 5 and _minor >= 8)
else:
    _is_v58_plus = False

_version_color = "#1b5e20" if _is_v58_plus else "#b71c1c"  # green / dark red
_version_status = "✓" if _is_v58_plus else "⚠ OLD"
with st.sidebar:
    st.markdown(
        f"""
        <div style="
            background-color:{_version_color};
            color:white;
            padding:10px 14px;
            border-radius:6px;
            font-weight:bold;
            font-size:0.95rem;
            margin-bottom:8px;
            line-height:1.4;
        ">
            {_version_status} Pipeline<br>
            <span style="font-family:monospace; font-size:0.85rem;">
            {_running_version}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not _is_v58_plus:
        st.warning(
            "古いコードが動いている可能性があります。"
            "Streamlit Cloud → Manage app → Reboot app で再起動してください。"
        )

# Keep this source-of-truth display in sync with scraper.LIVE_GRADE_FILTER.
_filter = scraper.LIVE_GRADE_FILTER
_filter_label = "ALL races" if _filter is None else " + ".join(_filter)

st.title("🎯 理論予想 Live — G1/G2 専用")
st.caption(
    f"対象: **{_filter_label} のみ** (scraper.LIVE_GRADE_FILTER). "
    f"対象外グレード・非グレードレースは現在処理対象外。"
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
    use_container_width=True,
    help=f"レース一覧の取得から {_filter_label} の理論予想まで自動で実行します。"
         f"対象外グレード・非グレードレースはスキップされます。",
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
        f"対象外グレードや平場レースは `scraper.LIVE_GRADE_FILTER` で意図的に除外しています。"
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

    if ok_results:
        archive_bytes = plog.build_prediction_archive_zip(ok_results)
        archive_date = race_date.isoformat()
        st.download_button(
            "💾 本日の予想アーカイブをローカルに保存",
            data=archive_bytes,
            file_name=f"prediction_archive_{archive_date}.zip",
            mime="application/zip",
            use_container_width=True,
            help=(
                "Streamlit Cloud 版はPCのフォルダへ直接書き込めないため、"
                "予想結果をZIPで保存します。ローカル実行時は data/prediction_archive "
                "にも自動保存されます。"
            ),
        )

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
                     use_container_width=True, hide_index=True)
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
        st.info(
            f"ℹ️ {len(odds_recovered)} レースで shutuba ページにオッズが無かったため、"
            f"ライブオッズAPI / 結果ページから自動補完しました。"
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
                "単勝オッズ": (
                    "---" if float(lb.get("odds", 0) or 0) <= 1.0
                    else f"⚠{float(lb.get('odds', 0) or 0):.1f}(要確認)"
                    if float(lb.get("odds", 0) or 0) > 500.0
                    else f"{float(lb.get('odds', 0) or 0):.1f}"
                ),
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
                    "単勝オッズ": (
                        "---" if float(t.get("odds", 0) or 0) <= 1.0
                        else f"⚠{float(t.get('odds', 0) or 0):.1f}(要確認)"
                        if float(t.get("odds", 0) or 0) > 500.0
                        else f"{float(t.get('odds', 0) or 0):.1f}"
                    ),
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

            # ── 🎯 本日のベスト 3 頭予測 (本命 / 対抗 / 単穴) ──
            # 憲法 §1.3「勝てるのは条件付き」を踏まえ、「必ず勝てる」とは
            # 言わない。model の win_prob を本命/対抗/単穴という馴染みの
            # 日本語で示し、買い方の目安を併記する。
            # LOOSE bets (ROI 検証用ルール) とは独立した「見やすい提示」。
            #
            # v5.7: G2 は市場分散戦略を適用 (analyze_g2_misses の結果、
            # G2 勝ち馬の 48% が市場 4-10 番人気のため、現行 win_prob TOP3
            # では構造的に取れない。過去 63 G2 レースの backtest で
            # ROI -48.2% → +20.8% へ改善)
            if r.get("ranked"):
                import grade_strategy as _gs
                ranked = r["ranked"]
                _grade = (r.get("grade", "") or "")
                _apply_diversified = _gs.should_apply_diversified(_grade)

                if _apply_diversified:
                    # Use the specific strategy selected for this grade.
                    _strategy_name = _gs.get_strategy_for_grade(_grade)
                    _mk_map = _gs.build_market_rank_map(ranked)
                    top3 = _gs.pick_diversified_top3(ranked, _mk_map,
                                                      strategy=_strategy_name)
                    labels = [h.get("bucket_mark", "◎") + " " +
                              h.get("bucket_label", "本命") for h in top3]
                    # Strategy description for UI
                    _strategy_desc = {
                        "diversified_1-3_4-7_8+": "G2 市場分散戦略 (1-3/4-7/8+)",
                        "loose_1-4_5-9_10+":      "広域分散戦略 (1-4/5-9/10+)",
                        "tight_1-2_3-5_6+":       "厳選戦略 (1-2/3-5/6+)",
                        "mid_heavy_1-2_3-6_7+":   "中位重視戦略 (1-2/3-6/7+)",
                        "wide_穴_1-3_4-8_9+":     "穴寄り戦略 (1-3/4-8/9+)",
                    }.get(_strategy_name, _strategy_name)
                    st.markdown(
                        f"### 🎯 本日のベスト 3 頭予測（{_strategy_desc}）"
                    )
                    # Per-grade caption
                    if "diversified_1-3" in _strategy_name:
                        st.caption(
                            "G2 は市場 4-10 番人気の勝利が 48% を占めるため、"
                            "本命を市場 1-3、対抗を 4-7、単穴を 8 番以下から"
                            "バランスよく選ぶ設計（63 R backtest: "
                            "ROI -48% → +21%）"
                        )
                    elif "loose_1-4" in _strategy_name:
                        st.caption(
                            "分散を広げつつ中上位を残す設計。"
                            "本命を市場 1-4、対抗を 5-9、単穴を 10 番以下から。"
                            "119 R backtest: ROI -37% → -3%。"
                        )
                else:
                    top3 = ranked[:3]
                    labels = ["◎ 本命", "○ 対抗", "▲ 単穴"]
                    st.markdown(
                        "### 🎯 本日のベスト 3 頭予測（本命 / 対抗 / 単穴）"
                    )

                bet_suggestions = [
                    "単勝 / 複勝",          # 本命: 勝ち狙いと保険
                    "複勝 / ワイド (本命絡み)",  # 対抗: 複勝圏狙い
                    "ワイド / 馬連 (本命絡み)",  # 単穴: 穴押さえ
                ]

                # 期待値計算: win_prob × odds。1.0 を超えれば理論的プラス
                def _ev(h):
                    odds = float(h.get("odds", 0) or 0)
                    wp = float(h.get("win_prob", 0) or 0)
                    if odds <= 1.0 or wp <= 0:
                        return None
                    return wp * odds

                top3_rows = []
                for i, h in enumerate(top3):
                    odds = float(h.get("odds", 0) or 0)
                    wp = float(h.get("win_prob", 0) or 0) * 100
                    ev = _ev(h)
                    odds_str = (
                        "---" if odds <= 1.0 else
                        f"⚠{odds:.1f}(要確認)" if odds > 500.0 else
                        f"{odds:.1f}倍"
                    )
                    ev_str = f"{ev:.2f}" if ev is not None else "—"
                    row = {
                        "印": labels[i],
                        "馬名": h.get("name", "?"),
                        "モデル勝率": f"{wp:.1f}%",
                        "単勝オッズ": odds_str,
                        "単勝期待値 (勝率×オッズ)": ev_str,
                        "推奨買い目": bet_suggestions[i],
                    }
                    # Diversified mode: add market rank column for transparency
                    if _apply_diversified and "market_rank" in h:
                        row["市場人気"] = f"{h['market_rank']}番人気"
                    top3_rows.append(row)
                top3_df = pd.DataFrame(top3_rows)
                st.dataframe(top3_df, use_container_width=True, hide_index=True)

                # 平易な補足 — 本命の根拠 (composite が取れていれば 1 行)
                best = top3[0]
                best_name = best.get("name", "この馬")
                best_wp = float(best.get("win_prob", 0) or 0) * 100
                best_comp = float(best.get("composite_condition", 0.5) or 0.5)
                best_edge = float(best.get("structured_edge", 0) or 0)
                reason_bits = []
                if best_comp >= 0.65:
                    reason_bits.append(f"composite {best_comp:.2f} (強ポジティブ)")
                if best_edge >= 0.05:
                    reason_bits.append(f"構造edge +{best_edge:.2f} (市場超過)")
                if best_edge <= -0.05:
                    reason_bits.append(f"構造edge {best_edge:.2f} (市場割安)")
                if float(best.get("odds", 0) or 0) <= 15.0:
                    reason_bits.append(
                        f"単勝 {float(best.get('odds', 0)):.1f}倍 (LOOSE 許容帯)"
                    )
                reason_text = (" · ".join(reason_bits)
                                if reason_bits else "特筆すべき edge なし")
                st.caption(
                    f"**本命 {best_name}**: モデル勝率 {best_wp:.1f}% · {reason_text}"
                )

                # 期待値警告: 本命の EV < 0.85 なら「本命でも妙味なし」を示唆
                ev_best = _ev(best)
                if ev_best is not None and ev_best < 0.85:
                    st.info(
                        f"⚠️ 本命の単勝期待値が **{ev_best:.2f}** と低め "
                        f"(市場 takeout 0.80 水準)。無理に単勝で勝負せず、"
                        f"複勝 / ワイドで薄く抑えるか、見送りも選択肢です。"
                    )

                # 憲法由来のスタンス
                st.caption(
                    "※ 競馬に「必ず勝てる」はありません（憲法 §1.3）。"
                    "表示は **モデルの確信度** であり、保証ではありません。"
                    "LOOSE 自動ベットは下の別パネルで引き続き運用中。"
                )

                st.divider()

            # Loose bets for this race
            if r.get("loose_bets"):
                st.markdown("**🧪 Loose bets (ROI 検証中の自動ルール)**")
                ldf = pd.DataFrame([
                    {
                        "馬名": lb["name"],
                        "単勝オッズ": (
                    "---" if float(lb.get("odds", 0) or 0) <= 1.0
                    else f"⚠{float(lb.get('odds', 0) or 0):.1f}(要確認)"
                    if float(lb.get("odds", 0) or 0) > 500.0
                    else f"{float(lb.get('odds', 0) or 0):.1f}"
                ),
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
                        "単勝オッズ": (
                        "---" if float(t.get("odds", 0) or 0) <= 1.0
                        else f"⚠{float(t.get('odds', 0) or 0):.1f}(要確認)"
                        if float(t.get("odds", 0) or 0) > 500.0
                        else f"{float(t.get('odds', 0) or 0):.1f}"
                    ),
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
                        "単勝オッズ": (
                            "---" if float(h.get("odds", 0) or 0) <= 1.0
                            else f"⚠{float(h.get('odds', 0) or 0):.1f}(要確認)"
                            if float(h.get("odds", 0) or 0) > 500.0
                            else f"{float(h.get('odds', 0) or 0):.1f}"
                        ),
                        "勝率": f"{h['win_prob']*100:.1f}%",
                    }
                    for i, h in enumerate(r["selected_top3"])
                ])
                st.dataframe(sel, use_container_width=True, hide_index=True)

            # Full ranking with calibration breakdown
            if r.get("ranked"):
                _stage_icon = "🟢" if r.get("prediction_stage") == "final" else "🟡"
                st.markdown(
                    f"**📊 全頭ランキング (上位8) — {_stage_icon} "
                    f"{r.get('prediction_stage', 'unknown')} version**"
                )

                _has_odds = any(
                    float(h.get("odds", 0) or 0) > 1.0
                    for h in r["ranked"][:8]
                )
                if not _has_odds:
                    st.warning(
                        "⚠️ **全馬のオッズが 0 (未公開) です。** "
                        "「単勝オッズ」列は `---` と表示されます。"
                        "この状態では市場勝率は均一 (1/N) であり、"
                        "ランキングは構造化スコアのみで決まっています。"
                        "オッズが公開されたら再実行してください。"
                    )

                # UI-level odds sanity gate (2026-04-19 最終防衛)。
                # 上流 pipeline で何かおかしな値 (>500) が紛れ込んでも、
                # 画面には絶対に "誤オッズ" として表示しない。
                # 現実の JRA 単勝最大は ~500 倍程度なので、
                # それを超える値は ?? + 警告を出す。
                def _fmt_odds(v) -> str:
                    try:
                        vf = float(v or 0)
                    except (TypeError, ValueError):
                        return "---"
                    if vf <= 1.0:
                        return "---"
                    if vf > 500.0:
                        return f"⚠{vf:.1f}(要確認)"
                    return f"{vf:.1f}"

                fdf = pd.DataFrame([
                    {
                        "順位": i + 1,
                        "馬名": h["name"],
                        "単勝オッズ": _fmt_odds(h.get("odds", 0)),
                        "市場勝率 (overround除去)": (
                            f"{h.get('base_market_prob', 0)*100:.1f}%"
                            if h.get("base_market_prob") is not None else "—"
                        ),
                        "構造edge": (
                            f"{h.get('structured_edge', 0):+.3f}"
                            if h.get("structured_edge") is not None else "—"
                        ),
                        "乗数 exp(k*e)": (
                            f"{h.get('fact_multiplier', 1.0):.2f}"
                            if h.get("fact_multiplier") is not None else "—"
                        ),
                        "最終勝率": f"{h.get('win_prob', 0)*100:.1f}%",
                        "内部スコア": f"{h.get('odds_score', 0):.1f}",
                        "mode": h.get("mode", "odds"),
                    }
                    for i, h in enumerate(r["ranked"][:8])
                ])
                st.dataframe(fdf, use_container_width=True, hide_index=True)
                st.caption(
                    f"**単勝オッズ** = netkeiba から取得した実オッズ "
                    f"(--- = 未公開 or 0)  |  "
                    f"**市場勝率** = (1/odds) / Σ(1/odds_all)  |  "
                    f"**内部スコア** = score_runner 出力 [2-95] (オッズではない)  |  "
                    f"**乗数** = exp(k × edge), k={r.get('calibration_k', 4.0):.1f}  |  "
                    f"mode: fact = strict trigger"
                )

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
st.markdown("### 🗂️ 予想アーカイブ")
archive_rows = plog.recent_prediction_archive_table(limit=30)
if archive_rows:
    adf = pd.DataFrame(archive_rows)
    preferred = [
        "race_date", "race_name", "prediction_stage", "loose_bet_count",
        "top1", "top2", "top3", "has_result", "archive_markdown",
    ]
    cols = [c for c in preferred if c in adf.columns] + \
           [c for c in adf.columns if c not in preferred]
    st.dataframe(adf[cols], use_container_width=True, hide_index=True)
    st.caption(f"保存先: `{plog.ARCHIVE_DIR}`")
else:
    st.caption(
        f"まだ保存済みアーカイブがありません。次回の予想から `{plog.ARCHIVE_DIR}` "
        "に日付別フォルダで保存されます。"
    )

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
