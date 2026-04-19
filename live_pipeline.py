"""Live prediction pipeline.

Differences from `fact_pipeline.run`:

  1. Newspaper scrapers (Hochi, Sanspo, Daily) are ENABLED only when
     the race date is today (or within ±1 day). These sources publish
     today's articles, not historical archives, so on any non-today
     race they contribute zero and wasting the call just burns time.

  2. Per-horse dual-mode scoring is applied — when a horse passes the
     strict fact-mode criterion, its score is recomputed with reduced
     odds weight (via `dual_mode_scoring.fact_weighted_score`), so
     the ranking can actually shift away from the market.

  3. Predictions are logged via `prediction_log.store_prediction`
     before the race runs. After results are posted, the same log
     is used to compute trigger_win_rate, trigger_ROI, etc.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Callable, Optional

import scraper
import feature_store as fs
import probability_engine as pe
import fact_collectors as fc
import fact_extractor as fe
import fact_validator as fv
import dual_mode_scoring as dm
import core_model_bridge as bridge
import odds_sources as osrc
from train import score_runner
from data_store import load_weights

import prediction_log


# Monotonic version tag that MUST be bumped whenever any of these change:
#   - score_runner coefficients in train.py
#   - DEFAULT_CALIBRATION_K in probability_engine.py
#   - the composition of features passed to score_runner
#   - the loose-rule definition in dual_mode_scoring.py
# Persisted with every prediction so a later audit can diff predictions
# that were produced under different model states.
DATA_SOURCE_VERSION = "live-v5.6-g3-scope-2026-04-19"
# v5.6 (2026-04-19): scope 拡大 G1/G2 → G1/G2/G3。
#   - scraper.LIVE_GRADE_FILTER = ("G1","G2","G3")
#   - paddock_sources.GRADE_TRIGGERS に G3 / JpnIII を追加
#   - live_pipeline の grade_guess 検出に G3 追加
# ユーザ買い方 (単勝×3 + 馬連×3 = 600円/R) を G3 までの全重賞に適用可能に。
# LOOSE 4 条件の数値は不変更。
#
# v5.4 (2026-04-19): TOP 3 本命/対抗/単穴 UI パネル追加。
# v5.3 (2026-04-19): paddock_sources 導入。G1/G2 限定で
# 東スポ + ラジオ NIKKEI + 日刊スポの 3 追加ソースからパドックテキスト
# を収集し、既存 netkeiba パドックに merge。consensus 向上狙い。
# LOOSE 4 条件の数値は不変更。
#
# v5.2 (2026-04-19): horse_facts_enricher 導入。累計獲得賞金 / 馬体重
# トレンド / 直近成績 (会場・複勝圏) / 休養期間 / 馬主・生産者・外厩
# tier を既取得データから fact として抽出 (追加 fetch 無し)。
# source tier `horse_deep` (conf 0.85) として composite に参加。
#
# 旧版履歴:
# v5.0 (2026-04-19 第5波): ユーザ直訴を受け、**データ取得側を完全に
# scratch-rewrite**。推論 (score_runner, dual_mode_scoring,
# probability_engine, trigger_loose_capped, 憲法) は一切変更しない。
#
# - 新規モジュール `entries_fetcher.py`  : 出馬表 (馬/騎手/厩舎) のみ。odds 非対応。
# - 新規モジュール `odds_fetcher.py`     : 単勝オッズの単一エントリ。
#                                          JSON API (action=init) 1 本だけ。
#                                          サニティ [1.0, 500.0]。欠損は None。
# - `_inject_odds_if_missing` / consensus / shutuba cells[9] 等、
#   過去の多層防御ロジックは**predict_live から除外**された。
#   (※ 関数自体は残しているが呼び出さない — v5.0 経路には関与しない)
#
# 旧版の履歴:
# v4.2 (2026-04-19 第4波): consensus 優先順位を shutuba-direct > API に。
# v4.1 (2026-04-19 第3波): cells[9] 廃止、span[id^='odds-'] のみ。
# v4.0 (2026-04 初回): multi-source consensus 導入。


def _log(cb: Optional[Callable[[str], None]], msg: str) -> None:
    if cb:
        cb(msg)


def _is_today_or_recent(race_date: Optional[str], window_days: int = 1) -> bool:
    """True when race_date (YYYY-MM-DD) is within `window_days` of today."""
    if not race_date:
        return True  # unknown date defaults to "assume live"
    try:
        d = datetime.strptime(race_date[:10], "%Y-%m-%d").date()
    except ValueError:
        return True
    today = date.today()
    return abs((today - d).days) <= window_days


def _parse_odds_safe(raw) -> float:
    s = str(raw or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _inject_odds_if_missing(
    entries: list[dict],
    race_id: str,
    progress_cb: Optional[Callable[[str], None]] = None,
    race_date: Optional[str] = None,
    venue: Optional[str] = None,
) -> tuple[str, dict]:
    """Ensure every entry has the freshest available odds.

    Priority (高→低):
      1. netkeiba の JSON odds API が `status=result` を返したら、
         shutuba の値に**関係なく全馬を上書き**する。これが最も
         直近の市場状態を反映する。(odds_source = "live-odds-api")
      2. (1) が使えないとき、shutuba HTML が既に全馬ぶん埋まっていれば
         そのまま使う。(odds_source = "shutuba")
      3. shutuba の欠損が少数 (<50%) なら残りは scratch と見なして触らない。
         (odds_source = "shutuba-partial")
      4. 欠損が過半なら結果ページから fill-in を試みる (past races)。
         (odds_source = "result-page")
      5. 結果ページも駄目で API が `middle` を返すなら未公開。
         (odds_source = "api-not-published", status = "not-published-yet")
      6. いずれも NG なら全ゼロで返す。(odds_source = "none")

    この順序は「shutuba より API、API より何もしない」ではなく
    「実際に使える最新の数字を最優先」にするための設計。
    RC-1 (`docs/odds_pipeline_audit.md` §2) を解消する。

    Returns (status_str, meta_dict). meta_dict は常に同じキーセット。

    Per-entry provenance:
      各エントリには `odds_source` と `odds_fetched_at` を書き込む。
      後段の `prediction_log` / `recent_loose_bets_table` が
      レース単位ではなく馬単位の出典を追跡できるようにするため。
    """
    meta: dict = {
        "odds_source":           None,
        "api_http_status":       None,
        "api_response_url":      None,
        "api_raw_reason":        None,
        "api_parse_error":       None,
        "api_schema_version":    None,
        "api_fetched_at":        None,
        "api_update_count":      None,
        "api_official_time":     None,
        "injected_count":        0,
        # ── 2026-04 multi-source consensus (RC-3 拡張) ──
        "consensus_primary_source":   None,   # "jra-official" | "netkeiba-api" | ...
        "consensus_enabled_sources":  [],
        "consensus_per_source":       {},     # source_name → SourceResult summary
        "consensus_disagreements":    {},     # umaban(str) → disagreement record
        "consensus_has_disagreement": False,
        "consensus_summary":          "",
        # ── 2026-04 pre-consensus sanity (フォルテアンジェロ事件対策) ──
        "entries_sanity_rejected":    {},     # number → bad_value (for audit)
        # ── pipeline version (古い cached result と見分けるため) ──
        "pipeline_version":           "v4.2-shutuba-primary-2026-04-19",
    }

    now_iso = datetime.now().isoformat(timespec="seconds")

    def _tag(e: dict, src: str, fetched_at: Optional[str] = None) -> None:
        """馬単位で出典と取得時刻を残す。"""
        e["odds_source"] = src
        e["odds_fetched_at"] = fetched_at or now_iso

    # ── Step 0: 入り口で sanity を掛ける (入力健全化) ──
    # shutuba HTML のカラムずれや別ソースのゴミが entries[*].odds に
    # 入ったまま到達すると、次の "欠損チェック" で「値あり」扱いされて
    # overlay のチャンスを失い、168.8 / 681.5 / 782.7 のような偽オッズが
    # 生き残ってしまう。ここで `[ODDS_MIN, ODDS_MAX]` 範囲外を "0" に
    # 叩き落とすことで、欠損扱い → overlay 可能状態にする。
    for e in entries:
        raw = e.get("odds", 0)
        v = _parse_odds_safe(raw)
        if v > 0 and not (osrc.ODDS_MIN <= v <= osrc.ODDS_MAX):
            try:
                num = int(str(e.get("number", "")).strip() or 0)
            except ValueError:
                num = 0
            meta["entries_sanity_rejected"][num] = v
            e["odds"] = "0"  # overlay 対象にする
            e["odds_prev_rejected"] = v
            _log(progress_cb,
                 f"⚠ sanity: 馬番#{num} odds={v} は範囲外なので欠損扱い "
                 f"({e.get('name', '?')})")

    # ── Step 1: Multi-source consensus を先に取る ──
    #   優先順位 (2026-04-19 第4波):
    #     shutuba-direct > JRA公式 > netkeiba JSON API > Yahoo 競馬
    #   `entries` を consensus に渡すことで shutuba-direct (= scraper が
    #   既に拾っている値) が最優先 primary として参加する。
    #   ユーザ確認済みの「scraping 結果は正しい」を pipeline 内で権威化し、
    #   JSON API が別 bet type を返す奇病を overlay で無効化する。
    consensus: dict = {}
    try:
        consensus = osrc.fetch_odds_consensus(
            race_id=race_id, race_date=race_date, venue=venue,
            entries=entries,
        ) or {}
    except Exception as e:
        _log(progress_cb, f"オッズ consensus 取得失敗: {e}")
        meta["api_raw_reason"] = f"consensus-exception: {e.__class__.__name__}: {e}"

    # Consensus 詳細を meta にたたみ込む
    per_source_summary = {}
    for name, res in (consensus.get("per_source") or {}).items():
        per_source_summary[name] = {
            "status":       res.get("status"),
            "n_horses":     len(res.get("by_number") or {}),
            "http_status":  res.get("http_status"),
            "schema_guess": res.get("schema_guess"),
            "fetched_at":   res.get("fetched_at"),
            "raw_reason":   res.get("raw_reason"),
        }
    meta["consensus_primary_source"]   = consensus.get("primary_source")
    meta["consensus_enabled_sources"]  = list(consensus.get("enabled_sources") or [])
    meta["consensus_per_source"]       = per_source_summary
    meta["consensus_disagreements"]    = {
        str(um): d for um, d in (consensus.get("disagreements") or {}).items()
    }
    meta["consensus_has_disagreement"] = bool(consensus.get("has_disagreement_any"))
    meta["consensus_summary"]          = osrc.summarize_disagreement(consensus)

    # 従来の "api_*" フィールドは netkeiba-api の取得結果をそのまま反映
    # （下流コードがこれらの名前を参照するため後方互換に残す）。
    nk = (consensus.get("per_source") or {}).get("netkeiba-api") or {}
    meta.update({
        "api_http_status":    nk.get("http_status"),
        "api_response_url":   nk.get("response_url"),
        "api_raw_reason":     nk.get("raw_reason") or meta["api_raw_reason"],
        "api_parse_error":    nk.get("parse_error"),
        "api_schema_version": nk.get("schema_guess"),
        "api_fetched_at":     nk.get("fetched_at"),
    })

    primary = consensus.get("primary_source") or "none"
    primary_by_number = consensus.get("primary_by_number") or {}

    if primary != "none" and primary_by_number:
        overlaid = 0
        for e in entries:
            try:
                um = int(str(e.get("number", "")).strip() or 0)
            except ValueError:
                um = 0
            if um in primary_by_number:
                e["odds"] = str(primary_by_number[um])
                _tag(e, primary, now_iso)
                # 馬ごとに disagreement フラグも書いておく — 下流 LOOSE veto 用
                dgr = osrc.disagreement_for_number(consensus, um)
                e["odds_disagreement_flag"] = bool(dgr.get("flag"))
                e["odds_disagreement_pct"]  = float(dgr.get("max_pct") or 0.0)
                e["odds_by_source"]         = dict(dgr.get("values") or {})
                overlaid += 1

        if overlaid:
            meta["odds_source"] = primary
            meta["injected_count"] = overlaid
            # 残りの scratch/欠番 entry にも最小限の tag を残す
            for e in entries:
                e.setdefault("odds_source",
                             "shutuba" if _parse_odds_safe(e.get("odds")) > 0
                             else "none")
                e.setdefault("odds_fetched_at", now_iso)
                e.setdefault("odds_disagreement_flag", False)
                e.setdefault("odds_disagreement_pct", 0.0)
                e.setdefault("odds_by_source", {})
            if overlaid >= len(entries):
                return f"overlaid-from-{primary} ({overlaid})", meta
            return f"overlaid-from-{primary} ({overlaid}/{len(entries)})", meta

    # ── Step 2: API 使えず、shutuba の状態で分岐 ──
    missing = [e for e in entries if _parse_odds_safe(e.get("odds", 0)) <= 0]
    if not missing:
        meta["odds_source"] = "shutuba"
        for e in entries:
            _tag(e, "shutuba")
        return "ok", meta
    if len(missing) < len(entries) / 2:
        # 欠損が少ない → scratch と見なして放置
        meta["odds_source"] = "shutuba-partial"
        for e in entries:
            if _parse_odds_safe(e.get("odds", 0)) > 0:
                _tag(e, "shutuba")
            else:
                _tag(e, "none")
        return f"partial-kept ({len(missing)})", meta

    # ── Step 3: 結果ページから fill-in (past races) ──
    _log(progress_cb, f"オッズ欠損検知 ({len(missing)}/{len(entries)}) — 結果ページから取得試行")
    try:
        result = scraper.fetch_result_netkeiba(race_id)
        if result:
            odds_map: dict[str, float] = {}
            for h in result.get("finishing_order", []) or []:
                nm = (h.get("name") or "").strip()
                od = _parse_odds_safe(h.get("odds"))
                if nm and od > 0 and nm not in odds_map:
                    odds_map[nm] = od
            if odds_map:
                injected = 0
                for e in entries:
                    nm = (e.get("name") or "").strip()
                    if _parse_odds_safe(e.get("odds", 0)) <= 0 and nm in odds_map:
                        e["odds"] = str(odds_map[nm])
                        _tag(e, "result-page")
                        injected += 1
                    else:
                        _tag(e, "shutuba" if _parse_odds_safe(e.get("odds")) > 0 else "none")
                if injected:
                    meta["odds_source"] = "result-page"
                    meta["injected_count"] = injected
                    return f"injected-from-result ({injected})", meta
    except Exception as e:
        _log(progress_cb, f"結果ページ取得失敗: {e}")

    # ── Step 4: netkeiba API が middle を返した場合の情報提示 ──
    # consensus の per_source から netkeiba-api の status を参照する。
    nk_status = (per_source_summary.get("netkeiba-api") or {}).get("status")
    if nk_status == "not-published":
        _log(
            progress_cb,
            f"netkeiba オッズ API: まだ公開前 "
            f"(status=middle, schema={meta['api_schema_version']})",
        )
        meta["odds_source"] = "api-not-published"
        for e in entries:
            _tag(e, "none")
        return "not-published-yet", meta

    meta["odds_source"] = "none"
    for e in entries:
        _tag(e, "none")
    return f"all-zero-no-source ({len(missing)}/{len(entries)} missing)", meta


# ── Main entry point ──────────────────────────────────

def predict_live(
    race_id: str,
    venue: str = "",
    race_name: str = "",
    race_date: Optional[str] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
    auto_log: bool = True,
) -> dict:
    """Run the full live fact pipeline and apply dual-mode scoring.

    `race_date` is a YYYY-MM-DD string. When it matches today ± 1 day,
    Hochi / Sanspo / Daily scrapers are called; otherwise they are
    skipped. This gating matches the actual coverage of those sources.
    """
    is_live = _is_today_or_recent(race_date)
    collection_log: list[dict] = []

    # ── Step 1: JRA-tier collection ──
    _log(progress_cb, "Tier 1: JRA 公式情報を収集中")
    jra = fc.collect_jra_facts(race_id, venue)
    collection_log.append(jra)

    # ── v5.0 (2026-04-19): scratch-rewrite data acquisition ──
    # ユーザ直訴 (フォルテアンジェロ事件) を受け、データ取得側を完全に
    # 新モジュール `entries_fetcher` + `odds_fetcher` に切り替える。
    # 旧 `scraper.fetch_entries_netkeiba` / 旧 `_inject_odds_if_missing`
    # は**呼ばない**。推論層 (score_runner, dual_mode_score,
    # probability_engine, trigger_loose_capped) は不変更。
    import entries_fetcher as _ef
    import odds_fetcher as _of

    entries = _ef.fetch_horse_entries(race_id, venue) or []

    # 旧 scraper の enrich cache (paddock / training / horse detail) は
    # これまで通り scraper 側で拾う。entries 側のフィールドを派生情報で
    # 埋める必要があるので、scraper.enrich_entries を run する。
    # ただし **scraper.enrich_entries は odds フィールドに触らない**
    # (新 entries_fetcher は odds を持たないので、もし enrich が偶然
    #  odds を書き込んでも 0 化される)。念のため明示的に削除する。
    try:
        entries = scraper.enrich_entries(entries, race_id,
                                         race_name=race_name) or entries
    except Exception as _e:
        _log(progress_cb, f"enrich_entries skipped: {_e}")
    for _e in entries:
        # enrich 側が past-cached odds (旧 cells[9] 由来) を残したまま
        # だった場合の防御。必ず新取得の値だけを採用する。
        _e.pop("odds", None)

    # 単勝オッズは scratch-rewrite の odds_fetcher から取得。
    _log(progress_cb, f"単勝オッズ取得 ({_of.FETCHER_VERSION})")
    win_odds = _of.fetch_win_odds(race_id)

    # entries に join
    odds_by_num = win_odds.by_number or {}
    filled = 0
    for e in entries:
        try:
            num = int(str(e.get("number", "")).strip() or 0)
        except (TypeError, ValueError):
            num = 0
        v = odds_by_num.get(num)
        if v is not None:
            e["odds"] = f"{v:.1f}"          # 文字列で保持 (既存 pipeline 互換)
            e["odds_source"] = "odds_fetcher-v5"
            e["odds_fetched_at"] = win_odds.fetched_at
            filled += 1
        else:
            e["odds"] = "0"
            e["odds_source"] = "missing"
            e["odds_fetched_at"] = win_odds.fetched_at

    # v5.0 simplified meta (旧 odds_meta の互換レイヤ)。
    if filled > 0:
        odds_status = f"v5-odds_fetcher ({filled}/{len(entries)})"
    elif win_odds.status == "not-published":
        odds_status = "not-published-yet"
    else:
        odds_status = f"all-zero-no-source ({len(entries)} missing)"

    odds_meta = {
        "odds_source":                "odds_fetcher-v5" if filled else "missing",
        "pipeline_version":           "v5.0-scratch-rewrite-2026-04-19",
        "fetcher_version":            _of.FETCHER_VERSION,
        "entries_fetcher_version":    _ef.ENTRIES_FETCHER_VERSION,
        "win_odds_status":            win_odds.status,
        "win_odds_raw_reason":        win_odds.raw_reason,
        "win_odds_http_status":       win_odds.http_status,
        "win_odds_official_time":     win_odds.official_time,
        "win_odds_update_count":      win_odds.update_count,
        "win_odds_rejected":          dict(win_odds.rejected or {}),
        "injected_count":             filled,
        # Legacy-compatible fields (下流の app_live などがまだ読んでる):
        "api_http_status":            win_odds.http_status,
        "api_raw_reason":             win_odds.raw_reason,
        "api_fetched_at":             win_odds.fetched_at,
        "api_schema_version":         "v1-jra-odds-2026",
        "api_response_url":           win_odds.response_url,
        "api_official_time":          win_odds.official_time,
        "api_parse_error":            None,
        "api_update_count":           win_odds.update_count,
        "consensus_primary_source":   "odds_fetcher-v5",
        "consensus_enabled_sources":  ["odds_fetcher-v5"],
        "consensus_per_source":       {"odds_fetcher-v5": {
            "status":       win_odds.status,
            "n_horses":     len(odds_by_num),
            "http_status":  win_odds.http_status,
            "schema_guess": "v1-jra-odds-2026",
            "fetched_at":   win_odds.fetched_at,
            "raw_reason":   win_odds.raw_reason,
        }},
        "consensus_disagreements":    {},
        "consensus_has_disagreement": False,
        "consensus_summary":          "single-source-v5",
        "entries_sanity_rejected":    dict(win_odds.rejected or {}),
    }

    # Provenance dump (hunting フォルテアンジェロ事件):
    # ODDS_TRACE_DIR が設定されている場合、このレースで各馬の odds が
    # どの経路から来たかを詳細 JSON で保存する。
    # これは不正値が UI に現れたとき「最後のチャンスで診断」する手段。
    import os as _os
    _trace_dir = _os.environ.get("ODDS_TRACE_DIR")
    if _trace_dir:
        try:
            from pathlib import Path as _P
            _td = _P(_trace_dir)
            _td.mkdir(parents=True, exist_ok=True)
            _ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            _dump = {
                "race_id": race_id,
                "race_date": race_date,
                "venue": venue,
                "odds_status": odds_status,
                "odds_meta": odds_meta,
                "entries": [
                    {
                        "number": e.get("number"),
                        "name": e.get("name"),
                        "odds": e.get("odds"),
                        "odds_source": e.get("odds_source"),
                        "odds_fetched_at": e.get("odds_fetched_at"),
                        "odds_by_source": e.get("odds_by_source"),
                        "odds_disagreement_flag": e.get("odds_disagreement_flag"),
                        "odds_disagreement_pct": e.get("odds_disagreement_pct"),
                        "odds_prev_rejected": e.get("odds_prev_rejected"),
                    }
                    for e in entries
                ],
            }
            import json as _json
            _p = _td / f"pipeline_{race_id}_{_ts}.json"
            _p.write_text(_json.dumps(_dump, ensure_ascii=False, indent=2),
                          encoding="utf-8")
            _log(progress_cb, f"[odds-trace] pipeline dump saved: {_p}")
        except Exception as _e:
            _log(progress_cb, f"[odds-trace] pipeline dump failed: {_e}")
    horse_names = [(e.get("name") or "").strip() for e in entries if e.get("name")]

    # Scratches go out of the name list
    scratched = {f.horse for f in jra["facts"] if f.type == "scratched"}
    running = [n for n in horse_names if n not in scratched]

    # ── Step 2: Tier-3 observational sources ──
    article_facts = []

    _log(progress_cb, "Tier 3: KeibaLab 観察情報")
    kl = fc.collect_keibalab_facts(race_id, running)
    collection_log.append(kl)
    article_facts.extend(kl["facts"])

    if is_live:
        _log(progress_cb, "Tier 3 (LIVE): Sports Hochi")
        h = fc.collect_hochi_facts(race_id, running, race_name)
        collection_log.append(h)
        article_facts.extend(h["facts"])

        _log(progress_cb, "Tier 3 (LIVE): Sankei Sports")
        s = fc.collect_sanspo_facts(race_id, running, race_name)
        collection_log.append(s)
        article_facts.extend(s["facts"])

        _log(progress_cb, "Tier 3 (LIVE): Daily Sports")
        d = fc.collect_daily_facts(race_id, running, race_name)
        collection_log.append(d)
        article_facts.extend(d["facts"])
    else:
        _log(progress_cb, "Newspaper scrapers skipped (race_date != today)")
        for src in ("hochi", "sanspo", "daily"):
            collection_log.append({
                "source": src, "status": "skipped",
                "facts": [], "items_seen": 0,
                "error": "non-live race date — newspaper sources publish only current articles",
            })

    # ── Step 3a (v5.3): multi-source paddock for G1/G2 only ──
    # 東スポ + ラジオ NIKKEI + 日刊スポ から追加のパドック情報を取得
    # (grade フィルタ内蔵 — G1/G2 以外では no-op)。
    # 得られたテキストは cached_race の paddock_comment に merge され、
    # 直後の collect_paddock_observation_facts の入力として使われる。
    cached_race = scraper._cache_load("enrich_race", race_id)
    try:
        import paddock_sources as _psrc
        grade_guess = ""
        # Try to read grade from race context (not yet fetched, but hints exist)
        # v5.6: G3 も追加 (ユーザ買い方は G3 まで)
        if race_name:
            for g_tag in ("G1", "G2", "G3", "JpnI", "JpnII", "JpnIII"):
                if f"({g_tag})" in race_name or f"({g_tag.lower()})" in race_name:
                    grade_guess = g_tag
                    break
        multi_result = _psrc.fetch_paddock_multi_sources(
            race_id=race_id, race_name=race_name,
            horse_names=list(running), grade=grade_guess,
        )
        _meta = multi_result.get("_meta", {}) if isinstance(multi_result, dict) else {}
        if _meta.get("enabled"):
            _log(progress_cb,
                 f"Tier 3 (v5.3): multi-paddock {_meta.get('n_sources_with_hits', 0)} "
                 f"sources, {_meta.get('total_horses_covered', 0)} horses")
            if cached_race:
                # Build a reports dict from cached_race so we can merge into it
                existing_reports = {
                    (h.get("name") or "").strip(): {
                        "text":   h.get("paddock_comment", "") or "",
                        "source": "netkeiba-cached",
                        "scores": h.get("paddock_scores", {}) or {},
                    }
                    for h in cached_race if h.get("name")
                }
                existing_reports = _psrc.merge_paddock_into_reports(
                    existing_reports, multi_result, list(running),
                )
                # Propagate merged text back onto cached_race horses so
                # downstream paddock fact collector sees the expanded text.
                for h in cached_race:
                    name = (h.get("name") or "").strip()
                    rep = existing_reports.get(name) or {}
                    if rep.get("text"):
                        h["paddock_comment"] = rep["text"]
                    if rep.get("scores"):
                        h["paddock_scores"] = rep["scores"]
        collection_log.append({
            "source": "paddock_multi_v5.3",
            "status": "ok" if _meta.get("enabled") else "skipped",
            "facts": [],  # facts are produced by downstream collector
            "items_seen": _meta.get("total_horses_covered", 0),
            "error": _meta.get("reason") if not _meta.get("enabled") else None,
        })
    except Exception as e:
        _log(progress_cb, f"paddock_sources multi skipped: {e}")

    # ── Step 3: cached paddock text + training_eval ──
    if cached_race:
        _log(progress_cb, "Tier 3: キャッシュ済みパドック観察")
        pd = fc.collect_paddock_observation_facts(race_id, running, cached_race)
        collection_log.append(pd)
        article_facts.extend(pd["facts"])

        # netkeiba oikiri 評価 text (training_eval) as a separate source
        _log(progress_cb, "Tier 3: netkeiba 調教評価")
        oikiri_facts = []
        for h in cached_race:
            te = h.get("training_eval") or ""
            name = (h.get("name") or "").strip()
            if te and len(te) >= 3 and name in running:
                oikiri_facts.extend(fe.extract_canonical_facts(
                    te, source="netkeiba_oikiri", horse=name,
                ))
        collection_log.append({
            "source": "netkeiba_oikiri",
            "status": "ok" if oikiri_facts else "skipped",
            "facts": oikiri_facts,
            "items_seen": sum(1 for h in cached_race if h.get("training_eval")),
            "error": None,
        })
        article_facts.extend(oikiri_facts)

    # ── Step 3d: horse_deep facts (v5.2 deep enrichment) ──
    # 既に `enrich_entries` で取得済みの horse detail / recent_races /
    # weight_trend / owner / breeder / ritto から **追加 fetch 無しで**
    # fact を抽出する。憲法 §7.1 の「ファクト抽出辞書の拡張」枠内。
    _log(progress_cb, "Tier 3 (deep): 累計賞金/体重トレンド/直近成績/休養/馬主 由来 fact")
    deep_facts: list = []
    try:
        import horse_facts_enricher as _hfe
        horses_for_deep = cached_race if cached_race else entries
        # race_info は後段 (Step 5) で fetch されるが、enricher に
        # 必要なのは race_date / venue だけなので predict_live 引数から
        # 直接構築する (Step 3 時点では race_info はまだ未定義)。
        _race_ctx_for_deep = {
            "race_date": race_date or "",
            "venue":     venue or "",
            "grade":     "",
        }
        deep_facts = _hfe.compute_deep_horse_facts(
            horses=horses_for_deep or [],
            race_info=_race_ctx_for_deep,
            venue=venue or "",
        )
        _log(progress_cb, f"horse_deep: {len(deep_facts)} facts generated "
                          f"({_hfe.ENRICHER_VERSION})")
    except Exception as e:
        _log(progress_cb, f"horse_deep enrichment skipped: {e}")
    collection_log.append({
        "source": "horse_deep",
        "status": "ok" if deep_facts else "skipped",
        "facts": deep_facts,
        "items_seen": len(cached_race) if cached_race else len(entries),
        "error": None,
    })
    article_facts.extend(deep_facts)

    # ── Step 4a: validate + contradiction-detect (fact_validator) ──
    _log(progress_cb, "ファクト検証 + 矛盾検出")
    drop_report: list = []
    all_raw_facts = list(jra["facts"]) + list(article_facts)
    validated = fv.validate_and_transform(all_raw_facts, drop_report=drop_report)

    # ── Step 4b: merge (consensus bonus + fuzzy clusters) ──
    _log(progress_cb, "ファクトをマージ (consensus bonus + fuzzy clusters)")
    merged = fe.merge_fact_layers(validated)

    by_horse: dict[str, list[fe.Fact]] = {}
    for f in merged:
        if f.horse:
            by_horse.setdefault(f.horse, []).append(f)

    # ── Step 4c: aggregate + state scores ──
    per_horse_agg: dict[str, dict] = {}
    per_horse_states: dict[str, dict] = {}
    for name in running:
        horse_facts = by_horse.get(name, [])
        per_horse_agg[name] = fe.aggregate_horse_score(horse_facts)
        per_horse_states[name] = fv.compute_state_scores(horse_facts)

    # ── Step 5: scoring with dual-mode ──
    _log(progress_cb, "スコア計算 (score_runner + dual-mode)")
    race_info = scraper.fetch_race_info_netkeiba(race_id) or {}
    entries_run = [e for e in entries if (e.get("name") or "").strip() in set(running)]

    # 馬ごとに odds の出典と取得時刻を後段で参照できるようにするため、
    # 名前→entry の lookup を作る（feature_store は source/fetched_at を
    # 通さないので、ここで持っておく）。
    entry_by_name = {
        (e.get("name") or "").strip(): e
        for e in entries_run
    }

    sf = fs.extract_structured_features(
        entries=entries_run,
        race_info=race_info,
        track_condition=race_info.get("track_condition", ""),
        weather=race_info.get("weather", ""),
        temperature=race_info.get("temperature", ""),
        cushion_value=race_info.get("cushion_value", ""),
        venue=venue,
    )
    sf_horses = sf.get("horses") or {}

    # ── CORE MODEL RECONNECTION ──
    # feature_store reads raw shutuba entries, which lack jockey_win_rate,
    # training_*, and paddock_*. Without this step, score_runner would
    # see 5 of 9 structured signal channels as zero and collapse to
    # essentially `(1/odds)/1.20 * 100`.
    #
    # The bridge fills those fields from:
    #   - scraper disk cache for jockey stats (db.netkeiba → cached JSON)
    #   - scraper oikiri page parse_training_critic for training signals
    #   - per-horse fact category aggregation for paddock signals
    bridge_diag = bridge.enrich_sf_horses_for_live(
        sf_horses=sf_horses,
        entries=entries_run,
        race_id=race_id,
        facts_by_horse=by_horse,
    )
    _log(
        progress_cb,
        f"bridge: jockey={bridge_diag['jockey_win_rate']} "
        f"training={bridge_diag['training_critic']} "
        f"paddock={bridge_diag['paddock_from_facts']} enriched",
    )

    # Inject composite (minus negative-state penalty) into the bio
    # pathway as a FALLBACK for horses where the bridge couldn't fill
    # paddock_* from facts (e.g. no per-horse fact coverage at all).
    #
    # Negative state scores (fatigue/stress/pain) subtract from the
    # composite BEFORE it's mapped into score_runner's [-1, 1] range,
    # so a horse with strong concerns is actively down-weighted rather
    # than merely losing positive support.
    STATE_PENALTY = {
        "fatigue_score": 0.30,
        "stress_score":  0.30,
        "pain_risk":     0.40,
    }
    for name, h in sf_horses.items():
        c = per_horse_agg.get(name, {}).get("composite_condition", 0.5)
        states = per_horse_states.get(name, {})
        penalty = sum(
            float(states.get(k, 0.0)) * w
            for k, w in STATE_PENALTY.items()
        )
        adjusted = max(0.02, min(0.98, c - penalty))
        centered = round(2.0 * adjusted - 1.0, 3)
        for key in ("paddock_gait", "paddock_hindquarter", "paddock_vascularity"):
            v = h.get(key)
            if not isinstance(v, (int, float)) or v == 0:
                h[key] = centered

    ctx = {"weights": load_weights()}
    grade_str = race_info.get("grade", "")
    entry_names = list(sf_horses.keys())

    scored = []
    trigger_info: list[dict] = []
    for this_name in entry_names:
        # Build score_runner input rotating this horse into slot 0
        rotated = [{
            "name": this_name, "rank": 1,
            "odds": sf_horses[this_name].get("odds", 0),
            "confidence": 0, "ev_gap": 0, "bet": "",
        }]
        for other in entry_names:
            if other == this_name:
                continue
            rotated.append({
                "name": other, "rank": 2,
                "odds": sf_horses[other].get("odds", 0),
                "confidence": 0, "ev_gap": 0, "bet": "",
            })
        feat = {
            "grade": grade_str, "num_horses": len(entry_names),
            "horse_features": rotated, "structured_features": sf,
        }
        odds_score = float(score_runner(feat, ctx).get("top_confidence", 50.0))

        # Pull fact aggregate for this horse
        agg = per_horse_agg.get(this_name, {})
        consensus_count = agg.get("consensed_fact_count", 0)
        composite = agg.get("composite_condition", 0.5)
        negatives = [f for f in by_horse.get(this_name, []) if f.polarity < 0]
        strong_negative_present = any(
            float(f.confidence) > dm.STRONG_FACT_MAX_NEG_CONF for f in negatives
        )

        decision = dm.dual_mode_score(
            h_sf=sf_horses[this_name],
            odds_score=odds_score,
            consensus_count=consensus_count,
            composite_condition=composite,
            negative_facts=negatives,
        )

        # Independent LOOSE trigger evaluation — does NOT affect the
        # strict dual-mode decision above, only sets a parallel flag.
        loose_odds = sf_horses[this_name].get("odds", 0) or 0
        loose_input = {
            "odds": loose_odds if loose_odds > 0 else None,
            "consensus_count": consensus_count,
            "composite_condition": composite,
            "strong_negative_present": strong_negative_present,
        }
        loose_flag, loose_reason = dm.trigger_loose_capped(loose_input)

        # ── Odds disagreement veto (RC-new 2026-04) ──
        #   憲法 §7.2 は LOOSE の 4 数値条件を fix するが、「入力される
        #   odds がそもそも正確でない」ケースの発火抑制は数値改変では
        #   なく入力健全性の担保。主権ソース (JRA 公式 > netkeiba-api)
        #   と他ソースで 20%+ 差が出ている馬は、オッズ反映の信頼度が
        #   確保できないので LOOSE を保留する。
        entry_for_veto = entry_by_name.get(this_name, {})
        disagreement_flag = bool(entry_for_veto.get("odds_disagreement_flag", False))
        disagreement_pct = float(entry_for_veto.get("odds_disagreement_pct", 0.0) or 0.0)
        if loose_flag and disagreement_flag:
            by_src = entry_for_veto.get("odds_by_source") or {}
            src_str = ", ".join(f"{s}={v:.1f}" for s, v in by_src.items())
            loose_flag = False
            loose_reason = (
                f"held: odds source disagreement "
                f"{disagreement_pct*100:.0f}% ({src_str})"
            )

        # Tiebreaker for exactly-tied scores (rare after bridge reconnection).
        jwr = float(sf_horses[this_name].get("jockey_win_rate", 0) or 0)
        epsilon = jwr * 0.2
        final_score = decision["score"] + epsilon

        # Recover score_runner's STRUCTURED adjustment (non-odds portion).
        # This is the signal the calibrated probability layer consumes
        # as the fact/model edge — NOT the raw score, which would double-
        # count the odds base.
        horse_odds = float(sf_horses[this_name].get("odds", 0) or 0)
        struct_edge = bridge.structured_edge_from_score(
            decision["odds_score"], horse_odds,
        )

        # ── v2: pedigree & camp trace fields ──
        h_sf = sf_horses[this_name]
        pedigree_comp = h_sf.get("pedigree_composite", 0.5)
        camp_comp = h_sf.get("camp_composite", 0.5)
        sire_dist_fit = h_sf.get("sire_distance_fit", 0.5)

        # Enrich scored rows with the signals assign_calibrated_probs needs
        scored.append({
            "name": this_name,
            "odds": horse_odds,
            "score": final_score,
            "odds_score": decision["odds_score"],
            "fact_score": decision["fact_score"],
            "mode": decision["mode"],
            # Calibrated-prob inputs
            "structured_edge": struct_edge,
            "composite_condition": composite,
            "consensus_count": consensus_count,
            # v2: pedigree & camp trace
            "pedigree_composite": round(pedigree_comp, 4),
            "camp_composite": round(camp_comp, 4),
            "sire_distance_fit": round(sire_dist_fit, 4),
            "sire_name": h_sf.get("sire_name", ""),
            "damsire_name": h_sf.get("damsire_name", ""),
            "breeder_name": h_sf.get("breeder_name", ""),
        })

        # Track triggers explicitly
        source_count_for_horse = len({
            src
            for f in by_horse.get(this_name, [])
            for src in (f.source or "").split("+")
            if src
        })

        states = per_horse_states.get(this_name, {})
        # Per-horse odds provenance (RC-3 の解消)。
        # `_inject_odds_if_missing` が各 entry に `odds_source` /
        # `odds_fetched_at` を書き込んでいる。ここでそれを trigger_info
        # に射影しておくと、`loose_bets` に自動で流れる。
        entry_for_horse = entry_by_name.get(this_name, {})
        odds_source_for_horse = entry_for_horse.get("odds_source")
        odds_fetched_for_horse = entry_for_horse.get("odds_fetched_at")
        odds_by_source_for_horse = entry_for_horse.get("odds_by_source") or {}
        odds_disagreement_flag_for_horse = bool(
            entry_for_horse.get("odds_disagreement_flag", False)
        )
        odds_disagreement_pct_for_horse = float(
            entry_for_horse.get("odds_disagreement_pct", 0.0) or 0.0
        )
        trigger_info.append({
            "name": this_name,
            "consensus_count": consensus_count,
            "composite_condition": round(composite, 3),
            # STRICT trigger (unchanged, audit-grade)
            "trigger_flag": decision["mode"] == "fact",
            "reason": decision["reason"],
            # LOOSE trigger (experimental betting rule, parallel to strict)
            "loose_trigger_flag": loose_flag,
            "loose_trigger_reason": loose_reason,
            "betting_candidate_flag": loose_flag,
            "strong_negative_present": strong_negative_present,
            # Shared metadata
            "source_count": source_count_for_horse,
            "odds_score": round(decision["odds_score"], 2),
            "fact_score": round(decision["fact_score"], 2),
            "odds": sf_horses[this_name].get("odds", 0),
            # 馬単位の odds 出典 (RC-3): "jra-official" / "netkeiba-api" /
            # "yahoo-keiba" / "live-odds-api" (legacy alias) / "shutuba" /
            # "result-page" / "none" のいずれか
            "odds_source": odds_source_for_horse,
            "odds_fetched_at": odds_fetched_for_horse,
            # 複数ソース cross-check (2026-04 multi-source consensus)
            "odds_by_source":            odds_by_source_for_horse,
            "odds_disagreement_flag":    odds_disagreement_flag_for_horse,
            "odds_disagreement_pct":     round(odds_disagreement_pct_for_horse, 4),
            "facts_preview": [
                f.type for f in by_horse.get(this_name, [])
                if f.meta.get("in_consensed_category")
            ][:6],
            # State scores from fact_validator.compute_state_scores
            "condition_score": states.get("condition_score", 0.5),
            "fatigue_score":   states.get("fatigue_score", 0.0),
            "stress_score":    states.get("stress_score", 0.0),
            "pain_risk":       states.get("pain_risk", 0.0),
        })

    # Market-anchored calibrated probability layer — replaces softmax(score)
    # for live display. score_runner's output was producing 95%+ concentration
    # on the top horse because a 35-point score gap under T=5 softmax collapses
    # to winner-takes-all. The calibrated formula uses market_prob * exp(k*edge)
    # with bounded k=0.8.
    ranked = pe.assign_calibrated_probs(scored, k=pe.DEFAULT_CALIBRATION_K)
    calibration_issues = pe.calibration_warnings(ranked)
    sel = pe.select_top3(ranked, alpha=pe.DEFAULT_ALPHA, beta=pe.DEFAULT_BETA)

    triggers = [t for t in trigger_info if t["trigger_flag"]]
    # LOOSE bets — independent projection of trigger_info.
    loose_bets = [t for t in trigger_info if t.get("loose_trigger_flag")]
    loose_bet_summary = [
        f"{t['name']}@{t['odds']:.1f} (cons={t['consensus_count']},"
        f" comp={t['composite_condition']:.2f})"
        for t in loose_bets
    ]

    # ── Prediction stage decision ──
    # "final" = we have trustworthy odds (scraped OK, partially kept,
    #           or successfully injected from result / live-odds API).
    # "early" = we do NOT have trustworthy odds (pre-publication or
    #           genuine fetch failure). These predictions exist so the
    #           fact layer + gate/field signals can still be reviewed
    #           BUT must never be mixed with final predictions in ROI
    #           aggregation — see weekly_report.by_stage.
    if odds_status.startswith((
        "ok", "partial-kept", "injected", "overlaid",
        "v5-odds_fetcher",   # v5.0 scratch-rewrite success
    )):
        prediction_stage = "final"
    else:
        prediction_stage = "early"

    created_at = datetime.now().isoformat(timespec="seconds")
    result = {
        "race_id": race_id,
        "race_name": race_name,
        "grade": grade_str,
        "venue": venue,
        "race_date": race_date or date.today().isoformat(),
        "is_live": is_live,
        "ranked": ranked,
        "selected_top3": sel["selected"],
        "p1": sel["p1"],
        "p2": sel["p2"],
        # STRICT trigger block (unchanged)
        "triggers": triggers,
        "per_horse_trigger_info": trigger_info,
        "per_horse_states": per_horse_states,
        # LOOSE trigger block (experimental, parallel to strict)
        "loose_bets": loose_bets,
        "loose_bet_count": len(loose_bets),
        "loose_bet_summary": loose_bet_summary,
        "loose_rule_version": dm.LOOSE_RULE_VERSION,
        # Shared
        "odds_status": odds_status,
        "calibration_warnings": calibration_issues,
        "calibration_k": pe.DEFAULT_CALIBRATION_K,
        "bridge_diag": bridge_diag,
        "validation_dropped": len(drop_report),
        "scratched": sorted(scratched),
        "collection_log": [
            {k: v for k, v in rec.items() if k != "facts"}
            for rec in collection_log
        ],
        "source_counts": {
            rec["source"]: len(rec.get("facts", []))
            for rec in collection_log
        },
        # ── Audit metadata (ADDED 2026-04 for history preservation) ──
        # These fields define what version of the system produced this
        # prediction and what state the odds were in at that moment.
        # store_prediction() uses them to snapshot into the history list.
        "data_source_version":         DATA_SOURCE_VERSION,
        "prediction_stage":            prediction_stage,
        "prediction_created_at":       created_at,
        "odds_status_at_prediction":   odds_status,
        "odds_updated_at":             (
            odds_meta.get("api_fetched_at")
            if prediction_stage == "final" and
               odds_meta.get("odds_source") == "live-odds-api"
            else None
        ),
        "odds_api_meta":               odds_meta,
        # Legacy alias retained for backward compat with KPI consumers
        "created_at": created_at,
    }

    if auto_log:
        prediction_log.store_prediction(result)

    return result
