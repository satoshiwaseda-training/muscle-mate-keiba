"""Isolated pedigree A/B test.

Runs the SAME enriched data through score_runner twice:
  - Control: pedigree_composite = camp_composite = sire_distance_fit = 0.5
  - Treatment: actual pedigree values from entity_tier tables

This eliminates the confound of re-enrichment and isolates
the pure effect of pedigree/camp features on ranking.

Usage:
  python tools/isolate_pedigree_effect.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._autolog_utils import banner, ensure_project_on_path
ensure_project_on_path()

import scraper
import feature_store as fs
import probability_engine as pe
import core_model_bridge as bridge
import dual_mode_scoring as dm
import fact_collectors as fc
import fact_extractor as fe
import fact_validator as fv
from train import score_runner
from data_store import load_weights
import prediction_log as plog


def _norm(s):
    return (s or "").strip()


def score_race(race_id, venue, race_name, use_pedigree=True):
    """Score a race. If use_pedigree=False, neutralize pedigree fields."""
    entries = scraper.fetch_entries_netkeiba(race_id, venue) or []
    entries = scraper.enrich_entries(entries, race_id, race_name=race_name)

    jra = fc.collect_jra_facts(race_id, venue)
    scratched = {f.horse for f in jra["facts"] if f.type == "scratched"}
    running = [_norm(e.get("name")) for e in entries
               if _norm(e.get("name")) not in scratched]

    article_facts = []
    kl = fc.collect_keibalab_facts(race_id, running)
    article_facts.extend(kl["facts"])
    cached = scraper._cache_load("enrich_race", race_id)
    if cached:
        pd_f = fc.collect_paddock_observation_facts(race_id, running, cached)
        article_facts.extend(pd_f["facts"])

    all_facts = list(jra["facts"]) + article_facts
    drop = []
    validated = fv.validate_and_transform(all_facts, drop_report=drop)
    merged = fe.merge_fact_layers(validated)
    by_horse = {}
    for f in merged:
        if f.horse:
            by_horse.setdefault(f.horse, []).append(f)

    per_horse_agg = {n: fe.aggregate_horse_score(by_horse.get(n, []))
                     for n in running}
    per_horse_states = {n: fv.compute_state_scores(by_horse.get(n, []))
                        for n in running}

    race_info = scraper.fetch_race_info_netkeiba(race_id) or {}
    entries_run = [e for e in entries
                   if _norm(e.get("name")) in set(running)]

    sf = fs.extract_structured_features(
        entries=entries_run, race_info=race_info,
        track_condition=race_info.get("track_condition", ""),
        weather=race_info.get("weather", ""),
        temperature=race_info.get("temperature", ""),
        cushion_value=race_info.get("cushion_value", ""),
        venue=venue,
    )
    sf_horses = sf.get("horses") or {}

    bridge.enrich_sf_horses_for_live(sf_horses, entries_run, race_id, by_horse)

    # Composite injection
    PENALTIES = {"fatigue_score": 0.3, "stress_score": 0.3, "pain_risk": 0.4}
    for name, h in sf_horses.items():
        c = per_horse_agg.get(name, {}).get("composite_condition", 0.5)
        states = per_horse_states.get(name, {})
        penalty = sum(float(states.get(k, 0)) * w for k, w in PENALTIES.items())
        adjusted = max(0.02, min(0.98, c - penalty))
        centered = round(2.0 * adjusted - 1.0, 3)
        for key in ("paddock_gait", "paddock_hindquarter", "paddock_vascularity"):
            v = h.get(key)
            if not isinstance(v, (int, float)) or v == 0:
                h[key] = centered

    # KEY: neutralize pedigree if disabled
    if not use_pedigree:
        for name, h in sf_horses.items():
            h["pedigree_composite"] = 0.5
            h["camp_composite"] = 0.5
            h["sire_distance_fit"] = 0.5

    ctx = {"weights": load_weights()}
    grade_str = race_info.get("grade", "")
    entry_names = list(sf_horses.keys())

    scored = []
    for this_name in entry_names:
        rotated = [{"name": this_name, "rank": 1,
                     "odds": sf_horses[this_name].get("odds", 0),
                     "confidence": 0, "ev_gap": 0, "bet": ""}]
        for other in entry_names:
            if other != this_name:
                rotated.append({"name": other, "rank": 2,
                                "odds": sf_horses[other].get("odds", 0),
                                "confidence": 0, "ev_gap": 0, "bet": ""})
        feat = {"grade": grade_str, "num_horses": len(entry_names),
                "horse_features": rotated, "structured_features": sf}
        odds_score = float(score_runner(feat, ctx).get("top_confidence", 50))

        agg = per_horse_agg.get(this_name, {})
        cc = agg.get("consensed_fact_count", 0)
        comp = agg.get("composite_condition", 0.5)
        negs = [f for f in by_horse.get(this_name, []) if f.polarity < 0]
        decision = dm.dual_mode_score(
            sf_horses[this_name], odds_score, cc, comp, negs)

        jwr = float(sf_horses[this_name].get("jockey_win_rate", 0) or 0)
        final_score = decision["score"] + jwr * 0.2
        horse_odds = float(sf_horses[this_name].get("odds", 0) or 0)
        struct_edge = bridge.structured_edge_from_score(
            decision["odds_score"], horse_odds)

        h_sf = sf_horses[this_name]
        scored.append({
            "name": this_name, "odds": horse_odds, "score": final_score,
            "structured_edge": struct_edge, "composite_condition": comp,
            "consensus_count": cc,
            "pedigree_composite": round(h_sf.get("pedigree_composite", 0.5), 4),
            "camp_composite": round(h_sf.get("camp_composite", 0.5), 4),
            "sire_distance_fit": round(h_sf.get("sire_distance_fit", 0.5), 4),
            "sire_name": h_sf.get("sire_name", ""),
            "breeder_name": h_sf.get("breeder_name", ""),
        })

    ranked = pe.assign_calibrated_probs(scored, k=pe.DEFAULT_CALIBRATION_K)
    return ranked


def main():
    races = [
        ("202609020611", "\u962a\u795e", "\u685c\u82b1\u8cde (G1)"),
        ("202606030511", "\u4e2d\u5c71", "NZT (G2)"),
        ("202609020511", "\u962a\u795e", "\u962a\u795e\u7261\u99acS (G2)"),
    ]

    # Load results for hit check
    all_preds = plog.list_predictions(only_live=True)
    results_map = {}
    for p in all_preds:
        rid = p.get("race_id", "")
        result = p.get("result") or {}
        fo = result.get("finishing_order") or []
        for h in fo:
            try:
                if int(h.get("rank", 0) or 0) == 1:
                    results_map[rid] = {
                        "winner": _norm(h.get("name")),
                        "payouts": result.get("payouts", {}),
                    }
            except ValueError:
                pass
        # odds favorite
        best = None
        for h in fo:
            try:
                od = float(str(h.get("odds", 0)).replace("---", "0"))
            except Exception:
                od = 0.0
            if od > 1.0 and (best is None or od < best[1]):
                best = (_norm(h.get("name")), od)
        if best and rid in results_map:
            results_map[rid]["odds_fav"] = best[0]

    banner("ISOLATED PEDIGREE A/B TEST")
    print("  Same enrichment data. Only difference: pedigree features on vs off.")
    print()

    total_changes = 0
    top1_changes = 0
    market_follow_off = 0
    market_follow_on = 0
    non_fav_on_count = 0
    non_fav_on_wins = 0
    non_fav_on_cost = 0.0
    non_fav_on_payout = 0.0
    off_wins = 0
    on_wins = 0
    off_cost = 0.0
    on_cost = 0.0
    off_payout = 0.0
    on_payout = 0.0
    n_races = 0

    for rid, venue, rname in races:
        print(f"--- {rname} ({rid}) ---")

        off_ranked = score_race(rid, venue, rname, use_pedigree=False)
        on_ranked = score_race(rid, venue, rname, use_pedigree=True)

        if not off_ranked or not on_ranked:
            print("  SKIP (scoring failed)\n")
            continue

        n_races += 1
        res = results_map.get(rid, {})
        winner = res.get("winner", "")
        odds_fav = res.get("odds_fav", "")
        payouts = res.get("payouts", {})
        try:
            win_pay = float(str(payouts.get("\u5358\u52dd", 0)).replace(",", "").replace("\u5186", "").strip() or 0)
        except Exception:
            win_pay = 0.0

        off_top1 = _norm(off_ranked[0]["name"])
        on_top1 = _norm(on_ranked[0]["name"])
        changed = off_top1 != on_top1

        if changed:
            top1_changes += 1

        if off_top1 == odds_fav:
            market_follow_off += 1
        if on_top1 == odds_fav:
            market_follow_on += 1

        if on_top1 != odds_fav:
            non_fav_on_count += 1
            non_fav_on_cost += 100.0
            if winner and on_top1 == winner:
                non_fav_on_wins += 1
                non_fav_on_payout += win_pay

        if winner:
            off_cost += 100.0
            on_cost += 100.0
            if off_top1 == winner:
                off_wins += 1
                off_payout += win_pay
            if on_top1 == winner:
                on_wins += 1
                on_payout += win_pay

        print(f"  OFF top1: {off_top1}")
        print(f"  ON  top1: {on_top1} {'*** CHANGED' if changed else ''}")
        print(f"  odds_fav: {odds_fav}")
        print(f"  winner:   {winner}")
        print()

        # Score comparison table
        off_map = {}
        for i, h in enumerate(off_ranked):
            off_map[_norm(h["name"])] = (i + 1, h["score"])

        hdr = (f"  {'name':20} {'off':>4} {'on':>4} {'d':>3} "
               f"{'score_off':>9} {'score_on':>9} {'diff':>7} "
               f"{'ped':>5} {'camp':>5} {'dist':>5} {'breeder':>14}")
        print(hdr)
        for i, h in enumerate(on_ranked[:12]):
            name = _norm(h["name"])
            off_rank, off_score = off_map.get(name, (99, 0))
            on_rank = i + 1
            delta = off_rank - on_rank
            score_diff = h["score"] - off_score
            ped = h.get("pedigree_composite", 0.5)
            camp = h.get("camp_composite", 0.5)
            dist = h.get("sire_distance_fit", 0.5)
            breeder = h.get("breeder_name", "")[:14]

            if abs(delta) > 0:
                total_changes += 1

            print(f"  {name:20} {off_rank:4} {on_rank:4} {delta:+3} "
                  f"{off_score:9.2f} {h['score']:9.2f} {score_diff:+7.2f} "
                  f"{ped:5.3f} {camp:5.3f} {dist:5.3f} {breeder:>14}")
        print()

    # Summary
    banner("SUMMARY")
    print(f"  Total races:           {n_races}")
    print(f"  Top1 changed:          {top1_changes}/{n_races}")
    print(f"  Total rank changes:    {total_changes}")
    print()
    print(f"  Market follow (OFF):   {market_follow_off}/{n_races}")
    print(f"  Market follow (ON):    {market_follow_on}/{n_races}")
    print()
    print(f"  Non-favorite top1 (ON): {non_fav_on_count}/{n_races}")
    if non_fav_on_count > 0:
        nf_hr = non_fav_on_wins / non_fav_on_count
        nf_roi = ((non_fav_on_payout - non_fav_on_cost) / non_fav_on_cost
                  if non_fav_on_cost > 0 else 0)
        print(f"    hit rate: {non_fav_on_wins}/{non_fav_on_count} = {nf_hr:.1%}")
        print(f"    ROI:      {nf_roi:+.1%}")
    print()
    if off_cost > 0:
        off_hr = off_wins / n_races
        on_hr = on_wins / n_races
        off_roi = (off_payout - off_cost) / off_cost
        on_roi = (on_payout - on_cost) / on_cost
        print(f"  WIN RATE (OFF): {off_wins}/{n_races} = {off_hr:.1%}  ROI={off_roi:+.1%}")
        print(f"  WIN RATE (ON):  {on_wins}/{n_races} = {on_hr:.1%}  ROI={on_roi:+.1%}")


if __name__ == "__main__":
    main()
