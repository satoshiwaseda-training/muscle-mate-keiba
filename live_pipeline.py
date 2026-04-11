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
from train import score_runner
from data_store import load_weights

import prediction_log


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
) -> str:
    """If ≥50% of entries have 0 odds, try to inject real odds.

    Tries in order:
      1. netkeiba result page (works for past races)
      2. netkeiba live odds page (works for upcoming races)

    Returns a status string used for UI display:
      'ok'                              — no injection needed
      'partial-kept'                    — few missing, probably scratches
      'injected-from-result ({n})'      — result page filled n horses
      'injected-from-live-odds ({n})'   — odds page filled n horses
      'all-zero-no-source'              — both fallbacks failed
    """
    missing = [e for e in entries if _parse_odds_safe(e.get("odds", 0)) <= 0]
    if not missing:
        return "ok"
    if len(missing) < len(entries) / 2:
        # Probably just scratched horses showing `---` — leave alone
        return f"partial-kept ({len(missing)})"

    # Try result page first (past races)
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
                        injected += 1
                if injected:
                    return f"injected-from-result ({injected})"
    except Exception as e:
        _log(progress_cb, f"結果ページ取得失敗: {e}")

    # Fallback: live odds page (upcoming races)
    _log(progress_cb, "ライブオッズページから取得試行")
    try:
        live_odds = scraper.fetch_odds_netkeiba(race_id) or {}
        if live_odds:
            injected = 0
            for e in entries:
                nm = (e.get("name") or "").strip()
                if _parse_odds_safe(e.get("odds", 0)) <= 0 and nm in live_odds:
                    e["odds"] = str(live_odds[nm])
                    injected += 1
            if injected:
                return f"injected-from-live-odds ({injected})"
    except Exception as e:
        _log(progress_cb, f"オッズページ取得失敗: {e}")

    return f"all-zero-no-source ({len(missing)}/{len(entries)} missing)"


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

    entries = scraper.fetch_entries_netkeiba(race_id, venue) or []
    # Inject odds from result/live-odds page when shutuba shows 0/`---`.
    # Without this, all horses would get score_runner's uniform 1/N base
    # and the ranking would collapse to gate-number order.
    odds_status = _inject_odds_if_missing(entries, race_id, progress_cb)
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

    # ── Step 3: cached paddock text + training_eval ──
    cached_race = scraper._cache_load("enrich_race", race_id)
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
    # Inject composite (minus negative-state penalty) into the bio
    # pathway for horses without real paddock data.
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

        # Tiebreaker: if score_runner returns near-identical scores (happens
        # when odds=0 for all horses), add a tiny per-horse epsilon from
        # jockey_win_rate so the ranking isn't frozen in gate-number order.
        # Scale: max +0.05 points on a 2-95 score range. Cannot flip
        # meaningful rank differences but breaks exact ties.
        jwr = float(sf_horses[this_name].get("jockey_win_rate", 0) or 0)
        epsilon = jwr * 0.2   # 0.20 → 0.04 pt boost for top jockey (25% win rate → 0.05)
        final_score = decision["score"] + epsilon

        scored.append({
            "name": this_name,
            "odds": sf_horses[this_name].get("odds", 0),
            "score": final_score,
            "odds_score": decision["odds_score"],
            "fact_score": decision["fact_score"],
            "mode": decision["mode"],
        })

        # Track triggers explicitly
        source_count_for_horse = len({
            src
            for f in by_horse.get(this_name, [])
            for src in (f.source or "").split("+")
            if src
        })

        states = per_horse_states.get(this_name, {})
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

    ranked = pe.assign_win_probs(scored, temperature=pe.DEFAULT_TEMPERATURE)
    sel = pe.select_top3(ranked, alpha=pe.DEFAULT_ALPHA, beta=pe.DEFAULT_BETA)

    triggers = [t for t in trigger_info if t["trigger_flag"]]
    # LOOSE bets — independent projection of trigger_info.
    loose_bets = [t for t in trigger_info if t.get("loose_trigger_flag")]
    loose_bet_summary = [
        f"{t['name']}@{t['odds']:.1f} (cons={t['consensus_count']},"
        f" comp={t['composite_condition']:.2f})"
        for t in loose_bets
    ]

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
        "created_at": datetime.now().isoformat(),
    }

    if auto_log:
        prediction_log.store_prediction(result)

    return result
