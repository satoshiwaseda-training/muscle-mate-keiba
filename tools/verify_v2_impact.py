"""V2 pedigree/camp feature impact verification.

Compares v1 (stored predictions) vs v2 (re-computed with pedigree)
for the same races to measure whether the new features actually
change rankings.

This script does NOT modify live_predictions.json.

Usage:
  python tools/verify_v2_impact.py
  python tools/verify_v2_impact.py --date 2026-04-12
  python tools/verify_v2_impact.py --all

Output:
  - Per-race ranking comparison (v1 vs v2)
  - Rank change count
  - Market follow rate (v1 vs v2)
  - Non-favorite top1 count / hit rate / ROI
  - Pedigree composite contribution analysis
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._autolog_utils import banner, log, ensure_project_on_path
ensure_project_on_path()

import scraper
import feature_store as fs
import probability_engine as pe
import core_model_bridge as bridge
import dual_mode_scoring as dm
import fact_extractor as fe
import fact_collectors as fc
import fact_validator as fv
import pedigree_features as pf
import prediction_log as plog
from train import score_runner
from data_store import load_weights


def _norm(s) -> str:
    return (s or "").strip()


def _winner_of(result: dict) -> str | None:
    fo = (result or {}).get("finishing_order") or []
    for h in fo:
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return _norm(h.get("name"))
        except ValueError:
            pass
    return None


def _odds_favorite_of(result: dict) -> str | None:
    fo = (result or {}).get("finishing_order") or []
    best = None
    for h in fo:
        try:
            od = float(str(h.get("odds", 0)).replace("---", "0"))
        except Exception:
            od = 0.0
        if od > 1.0 and (best is None or od < best[1]):
            best = (_norm(h.get("name")), od)
    return best[0] if best else None


def recompute_v2(race_id: str, venue: str = "", race_name: str = "",
                 race_date: str = "") -> dict | None:
    """Re-run scoring with v2 code but do NOT persist.

    Returns a dict with ranked list and pedigree trace, or None on failure.
    """
    try:
        entries = scraper.fetch_entries_netkeiba(race_id, venue) or []
        if not entries:
            return None

        # Enrich entries (uses cache, re-fetches horse detail if stale)
        entries = scraper.enrich_entries(entries, race_id, race_name=race_name)

        # JRA facts
        jra = fc.collect_jra_facts(race_id, venue)
        scratched = {f.horse for f in jra["facts"] if f.type == "scratched"}
        horse_names = [_norm(e.get("name")) for e in entries if e.get("name")]
        running = [n for n in horse_names if n not in scratched]

        # Fact collection (minimal — cached)
        article_facts = []
        kl = fc.collect_keibalab_facts(race_id, running)
        article_facts.extend(kl["facts"])

        cached_race = scraper._cache_load("enrich_race", race_id)
        if cached_race:
            pd_facts = fc.collect_paddock_observation_facts(race_id, running, cached_race)
            article_facts.extend(pd_facts["facts"])

        all_raw = list(jra["facts"]) + article_facts
        drop_report = []
        validated = fv.validate_and_transform(all_raw, drop_report=drop_report)
        merged = fe.merge_fact_layers(validated)

        by_horse = {}
        for f in merged:
            if f.horse:
                by_horse.setdefault(f.horse, []).append(f)

        per_horse_agg = {}
        per_horse_states = {}
        for name in running:
            hf = by_horse.get(name, [])
            per_horse_agg[name] = fe.aggregate_horse_score(hf)
            per_horse_states[name] = fv.compute_state_scores(hf)

        # Feature extraction (v2 — includes pedigree)
        race_info = scraper.fetch_race_info_netkeiba(race_id) or {}
        entries_run = [e for e in entries if _norm(e.get("name")) in set(running)]

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

        # Bridge enrichment
        bridge.enrich_sf_horses_for_live(
            sf_horses=sf_horses,
            entries=entries_run,
            race_id=race_id,
            facts_by_horse=by_horse,
        )

        # Composite injection
        STATE_PENALTY = {"fatigue_score": 0.30, "stress_score": 0.30, "pain_risk": 0.40}
        for name, h in sf_horses.items():
            c = per_horse_agg.get(name, {}).get("composite_condition", 0.5)
            states = per_horse_states.get(name, {})
            penalty = sum(float(states.get(k, 0.0)) * w for k, w in STATE_PENALTY.items())
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
        for this_name in entry_names:
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

            agg = per_horse_agg.get(this_name, {})
            consensus_count = agg.get("consensed_fact_count", 0)
            composite = agg.get("composite_condition", 0.5)
            negatives = [f for f in by_horse.get(this_name, []) if f.polarity < 0]

            decision = dm.dual_mode_score(
                h_sf=sf_horses[this_name],
                odds_score=odds_score,
                consensus_count=consensus_count,
                composite_condition=composite,
                negative_facts=negatives,
            )

            jwr = float(sf_horses[this_name].get("jockey_win_rate", 0) or 0)
            epsilon = jwr * 0.2
            final_score = decision["score"] + epsilon

            horse_odds = float(sf_horses[this_name].get("odds", 0) or 0)
            struct_edge = bridge.structured_edge_from_score(decision["odds_score"], horse_odds)

            h_sf = sf_horses[this_name]
            scored.append({
                "name": this_name,
                "odds": horse_odds,
                "score": final_score,
                "odds_score": decision["odds_score"],
                "fact_score": decision["fact_score"],
                "mode": decision["mode"],
                "structured_edge": struct_edge,
                "composite_condition": composite,
                "consensus_count": consensus_count,
                "pedigree_composite": round(h_sf.get("pedigree_composite", 0.5), 4),
                "camp_composite": round(h_sf.get("camp_composite", 0.5), 4),
                "sire_distance_fit": round(h_sf.get("sire_distance_fit", 0.5), 4),
                "sire_name": h_sf.get("sire_name", ""),
                "damsire_name": h_sf.get("damsire_name", ""),
                "breeder_name": h_sf.get("breeder_name", ""),
            })

        ranked = pe.assign_calibrated_probs(scored, k=pe.DEFAULT_CALIBRATION_K)
        return {
            "race_id": race_id,
            "race_name": race_name,
            "venue": venue,
            "ranked": ranked,
            "sf_version": sf.get("version", 0),
        }

    except Exception as e:
        log(f"FAIL recompute_v2 {race_id}: {e}", level="ERROR")
        import traceback
        traceback.print_exc()
        return None


def analyze(v1_preds: list[dict], v2_results: list[dict]) -> dict:
    """Compare v1 stored predictions with v2 recomputed results."""

    report = {
        "total_races": 0,
        "races_with_result": 0,
        "rank_changes": [],       # per-race rank change details
        "total_rank_changes": 0,
        "market_follow_v1": 0,
        "market_follow_v2": 0,
        "non_favorite_v1": 0,
        "non_favorite_v2": 0,
        "non_favorite_v2_wins": 0,
        "non_favorite_v2_cost": 0.0,
        "non_favorite_v2_payout": 0.0,
        "v1_wins": 0,
        "v2_wins": 0,
        "v1_cost": 0.0,
        "v1_payout": 0.0,
        "v2_cost": 0.0,
        "v2_payout": 0.0,
        "pedigree_impact": [],    # per-race pedigree contribution
    }

    for v1, v2 in zip(v1_preds, v2_results):
        if v2 is None:
            continue

        result = v1.get("result") or {}
        winner = _winner_of(result)
        odds_fav = _odds_favorite_of(result)
        pay = result.get("payouts") or {}

        try:
            win_pay = float(str(pay.get("\u5358\u52dd", 0)).replace(",", "").replace("\u5186", "").strip() or 0)
        except Exception:
            win_pay = 0.0

        v1_ranked = v1.get("ranked") or []
        v2_ranked = v2.get("ranked") or []

        if not v1_ranked or not v2_ranked:
            continue

        report["total_races"] += 1

        v1_top1 = _norm(v1_ranked[0].get("name"))
        v2_top1 = _norm(v2_ranked[0].get("name"))

        # Market follow
        if v1_top1 == odds_fav:
            report["market_follow_v1"] += 1
        if v2_top1 == odds_fav:
            report["market_follow_v2"] += 1

        # Non-favorite picks
        if v1_top1 != odds_fav:
            report["non_favorite_v1"] += 1
        if v2_top1 != odds_fav:
            report["non_favorite_v2"] += 1
            report["non_favorite_v2_cost"] += 100.0
            if winner and v2_top1 == winner:
                report["non_favorite_v2_wins"] += 1
                report["non_favorite_v2_payout"] += win_pay

        # Win check
        if winner:
            report["races_with_result"] += 1

            report["v1_cost"] += 100.0
            if v1_top1 == winner:
                report["v1_wins"] += 1
                report["v1_payout"] += win_pay

            report["v2_cost"] += 100.0
            if v2_top1 == winner:
                report["v2_wins"] += 1
                report["v2_payout"] += win_pay

        # Rank change analysis
        v1_order = [_norm(h.get("name")) for h in v1_ranked[:10]]
        v2_order = [_norm(h.get("name")) for h in v2_ranked[:10]]
        changes = []
        for i, name in enumerate(v2_order):
            if name in v1_order:
                v1_pos = v1_order.index(name)
                if v1_pos != i:
                    changes.append({
                        "name": name,
                        "v1_rank": v1_pos + 1,
                        "v2_rank": i + 1,
                        "delta": v1_pos - i,  # positive = moved up
                    })

        top1_changed = v1_top1 != v2_top1

        # Pedigree contribution for v2 top1
        v2_top_data = v2_ranked[0] if v2_ranked else {}
        ped_comp = v2_top_data.get("pedigree_composite", 0.5)
        camp_comp = v2_top_data.get("camp_composite", 0.5)
        sire_dist = v2_top_data.get("sire_distance_fit", 0.5)

        race_detail = {
            "race_id": v1.get("race_id", ""),
            "race_name": v1.get("race_name", ""),
            "v1_top1": v1_top1,
            "v2_top1": v2_top1,
            "top1_changed": top1_changed,
            "odds_fav": odds_fav or "",
            "winner": winner or "",
            "rank_changes_top10": len(changes),
            "changes": changes,
            "v2_pedigree_composite": ped_comp,
            "v2_camp_composite": camp_comp,
            "v2_sire_distance_fit": sire_dist,
            "v2_sire_name": v2_top_data.get("sire_name", ""),
            "v2_damsire_name": v2_top_data.get("damsire_name", ""),
            "v2_breeder_name": v2_top_data.get("breeder_name", ""),
        }
        report["rank_changes"].append(race_detail)
        report["total_rank_changes"] += len(changes)
        report["pedigree_impact"].append(race_detail)

    return report


def print_report(rep: dict) -> None:
    n = rep["total_races"]
    nr = rep["races_with_result"]

    banner("V2 Pedigree/Camp Impact Verification")

    print("== Volume ==")
    print(f"  total races compared: {n}")
    print(f"  races with result:    {nr}")

    if n == 0:
        print("\n  [!] No races to compare. Exiting.")
        return

    print(f"\n== Market Follow Rate ==")
    mf_v1 = rep["market_follow_v1"] / n if n else 0
    mf_v2 = rep["market_follow_v2"] / n if n else 0
    print(f"  v1 (stored):  {rep['market_follow_v1']}/{n} = {mf_v1:.1%}")
    print(f"  v2 (pedigree): {rep['market_follow_v2']}/{n} = {mf_v2:.1%}")
    print(f"  diff:          {mf_v2 - mf_v1:+.1%}")

    print(f"\n== Non-Favorite Top1 (1番人気以外を本命にしたレース) ==")
    print(f"  v1: {rep['non_favorite_v1']}/{n}")
    print(f"  v2: {rep['non_favorite_v2']}/{n}")
    if rep["non_favorite_v2"] > 0:
        nf_hit = rep["non_favorite_v2_wins"] / rep["non_favorite_v2"]
        nf_roi = (rep["non_favorite_v2_payout"] - rep["non_favorite_v2_cost"]) / rep["non_favorite_v2_cost"] if rep["non_favorite_v2_cost"] > 0 else 0
        print(f"  v2 non-fav hit rate: {rep['non_favorite_v2_wins']}/{rep['non_favorite_v2']} = {nf_hit:.1%}")
        print(f"  v2 non-fav ROI:      {nf_roi:+.1%}")
    else:
        print(f"  (v2 always picked favorite)")

    if nr > 0:
        print(f"\n== Overall Win Rate & ROI ==")
        v1_hr = rep["v1_wins"] / nr
        v2_hr = rep["v2_wins"] / nr
        v1_roi = (rep["v1_payout"] - rep["v1_cost"]) / rep["v1_cost"] if rep["v1_cost"] > 0 else 0
        v2_roi = (rep["v2_payout"] - rep["v2_cost"]) / rep["v2_cost"] if rep["v2_cost"] > 0 else 0
        print(f"  v1 win rate: {rep['v1_wins']}/{nr} = {v1_hr:.1%}  ROI={v1_roi:+.1%}")
        print(f"  v2 win rate: {rep['v2_wins']}/{nr} = {v2_hr:.1%}  ROI={v2_roi:+.1%}")

    print(f"\n== Rank Changes (Top 10) ==")
    print(f"  total position changes: {rep['total_rank_changes']}")
    for rd in rep["rank_changes"]:
        changed_mark = "***" if rd["top1_changed"] else ""
        print(f"\n  {rd['race_id']} {rd['race_name']}")
        print(f"    v1 top1: {rd['v1_top1']}")
        print(f"    v2 top1: {rd['v2_top1']} {changed_mark}")
        print(f"    odds fav: {rd['odds_fav']}")
        print(f"    winner:   {rd['winner']}")
        print(f"    rank changes in top10: {rd['rank_changes_top10']}")
        for ch in rd["changes"]:
            arrow = "UP" if ch["delta"] > 0 else "DOWN"
            print(f"      {ch['name']}: {ch['v1_rank']} -> {ch['v2_rank']} ({arrow} {abs(ch['delta'])})")

    print(f"\n== Pedigree Contribution (v2 Top1) ==")
    for rd in rep["pedigree_impact"]:
        ped = rd["v2_pedigree_composite"]
        camp = rd["v2_camp_composite"]
        dist = rd["v2_sire_distance_fit"]
        ped_effect = "ACTIVE" if ped != 0.5 else "neutral"
        camp_effect = "ACTIVE" if camp != 0.5 else "neutral"
        dist_effect = "ACTIVE" if dist != 0.5 else "neutral"
        print(f"  {rd['race_name']}: {rd['v2_top1']}")
        print(f"    sire={rd['v2_sire_name']} damsire={rd['v2_damsire_name']} breeder={rd['v2_breeder_name']}")
        print(f"    pedigree={ped:.3f}({ped_effect}) camp={camp:.3f}({camp_effect}) dist_fit={dist:.3f}({dist_effect})")


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", help="Target race date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="All stored predictions")
    args = parser.parse_args()

    all_preds = plog.list_predictions(only_live=True)

    if args.date:
        preds = [e for e in all_preds if e.get("race_date") == args.date]
    elif args.all:
        preds = [e for e in all_preds if e.get("result")]
    else:
        # Default: all predictions with results
        preds = [e for e in all_preds if e.get("result")]

    if not preds:
        log("No predictions found matching criteria.", level="WARN")
        return 1

    log(f"Found {len(preds)} predictions to verify.")

    # Recompute each race with v2
    v2_results = []
    for i, pred in enumerate(preds):
        rid = pred.get("race_id", "")
        rname = pred.get("race_name", "")
        venue = pred.get("venue", "")
        rdate = pred.get("race_date", "")
        log(f"[{i+1}/{len(preds)}] Recomputing {rid} '{rname}'...")
        v2 = recompute_v2(rid, venue=venue, race_name=rname, race_date=rdate)
        v2_results.append(v2)
        if v2:
            top1 = _norm(v2["ranked"][0]["name"]) if v2["ranked"] else "?"
            ped = v2["ranked"][0].get("pedigree_composite", 0.5) if v2["ranked"] else 0.5
            log(f"  -> v2 top1={top1} pedigree_composite={ped:.3f}")
        else:
            log(f"  -> FAILED", level="ERROR")

    report = analyze(preds, v2_results)
    print_report(report)

    # Save JSON report
    out_path = Path("data/v2_verification_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log(f"Report saved: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
