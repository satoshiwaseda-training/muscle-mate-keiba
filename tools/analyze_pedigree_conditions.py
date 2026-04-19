"""Conditional pedigree effectiveness analysis.

For every horse in every backtest prediction pair (on/off), extract:
  - base_score (pedigree off)
  - on_score  (pedigree on)
  - score_delta = on_score - base_score
  - odds
  - pedigree_composite, camp_composite, sire_distance_fit
  - is_winner (from results.json)

Then slice by:
  1. Odds band
  2. Close races (top1-top2 base_score gap < 3pt)
  3. Pedigree extremes (>0.7 or <0.3)
  4. Camp extremes (>0.8 or <0.4)
  5. Distance-fit mismatch (<0.4)

For each slice, compute:
  - horse count
  - mean pedigree score_delta
  - max absolute score_delta
  - base score gap (top1 - top2) distribution
  - "rank flip potential" = races where |score_delta| could exceed
    the base_score gap between current rank and rank-1
  - hit rate / ROI where applicable

RULE: analyze by SCORE DELTA, not rank. Compare base gap vs pedigree gap.

USAGE:
  python tools/analyze_pedigree_conditions.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "data" / "backtest_predictions"
RESULTS_FILE = PROJECT_ROOT / "data" / "results.json"


def _norm(s):
    return (s or "").strip()


def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        key = k[3:] if k.startswith("bt_") else k
        out[key] = v
    return out


def _winner(result: dict) -> str:
    fo = (result or {}).get("finishing_order") or []
    for h in fo:
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return _norm(h.get("name"))
        except ValueError:
            pass
    return ""


def _win_pay(result: dict) -> float:
    pay = (result or {}).get("payouts") or {}
    raw = pay.get("\u5358\u52dd", 0)
    try:
        return float(str(raw).replace(",", "").replace("\u5186", "").strip() or 0)
    except Exception:
        return 0.0


def _odds_band(odds: float) -> str:
    if odds <= 0:
        return "no-odds"
    if odds < 3.0:
        return "<3"
    if odds < 6.0:
        return "3-6"
    if odds < 15.0:
        return "6-15"
    return ">15"


# ═══════════════════════════════════════════════════════════
# Extract horse-level rows from prediction pairs
# ═══════════════════════════════════════════════════════════

def load_pairs() -> list[dict]:
    """Return list of per-race dicts:

    {
      race_id, race_name, race_date,
      winner,
      horses: [
        { name, odds, base_score, on_score, score_delta,
          pedigree_composite, camp_composite, sire_distance_fit,
          sire_name, damsire_name, breeder_name,
          base_rank, on_rank, is_winner }
      ]
    }
    """
    on_files = {p.stem.replace("_on", ""): p
                for p in PRED_DIR.glob("*_on.json")}
    off_files = {p.stem.replace("_off", ""): p
                 for p in PRED_DIR.glob("*_off.json")}
    common = sorted(set(on_files.keys()) & set(off_files.keys()))

    results_map = _load_results()
    races = []

    for rid in common:
        with open(on_files[rid], "r", encoding="utf-8") as f:
            on = json.load(f)
        with open(off_files[rid], "r", encoding="utf-8") as f:
            off = json.load(f)

        on_ranked = on.get("ranked") or []
        off_ranked = off.get("ranked") or []
        if not on_ranked or not off_ranked:
            continue

        # Build lookup: name -> (score, rank, odds) for both modes
        off_by_name = {}
        for i, h in enumerate(off_ranked):
            off_by_name[_norm(h.get("name"))] = {
                "score": h.get("score", 0.0),
                "rank": i + 1,
                "odds": h.get("odds", 0.0),
            }

        result = results_map.get(rid, {})
        winner = _winner(result)
        pay = _win_pay(result)

        horses = []
        for i, h in enumerate(on_ranked):
            name = _norm(h.get("name"))
            off_info = off_by_name.get(name, {})
            base_score = off_info.get("score", 0.0)
            on_score = h.get("score", 0.0)
            odds = h.get("odds", 0.0) or off_info.get("odds", 0.0)

            horses.append({
                "name":       name,
                "odds":       odds,
                "base_score": base_score,
                "on_score":   on_score,
                "score_delta": on_score - base_score,
                "pedigree_composite": h.get("pedigree_composite", 0.5),
                "camp_composite":     h.get("camp_composite", 0.5),
                "sire_distance_fit":  h.get("sire_distance_fit", 0.5),
                "sire_name":          h.get("sire_name", ""),
                "damsire_name":       h.get("damsire_name", ""),
                "breeder_name":       h.get("breeder_name", ""),
                "base_rank":  off_info.get("rank", 99),
                "on_rank":    i + 1,
                "is_winner":  name == winner,
            })

        # Base-score-sorted order for top1/top2 gap
        base_sorted = sorted(horses, key=lambda r: -r["base_score"])

        races.append({
            "race_id":   rid,
            "race_name": on.get("race_name", ""),
            "race_date": on.get("race_date", ""),
            "grade":     on.get("grade", ""),
            "winner":    winner,
            "win_pay":   pay,
            "has_result": bool(winner),
            "horses":    horses,
            "base_sorted": base_sorted,
            "base_top1_top2_gap": (
                base_sorted[0]["base_score"] - base_sorted[1]["base_score"]
                if len(base_sorted) >= 2 else 0.0
            ),
        })

    return races


# ═══════════════════════════════════════════════════════════
# Slice analysis
# ═══════════════════════════════════════════════════════════

def _safe_mean(xs): return mean(xs) if xs else 0.0
def _safe_median(xs): return median(xs) if xs else 0.0

def _hit_roi(horses: list[dict]) -> tuple[int, int, float]:
    """For a filter group (treated as betting each filtered horse):
    returns (n_bets, n_wins, roi)."""
    if not horses:
        return (0, 0, 0.0)
    # Only count horses whose race had a result
    # For this analysis, we bet 100 on each filtered horse
    cost = 0.0
    payout = 0.0
    wins = 0
    # Note: win payout is per-RACE for the winning horse; if our filter
    # picks the winner, the payout is the win_pay stored on the race.
    # We need race-level win_pay here, but the horses list is flat.
    # For simplicity, only require is_winner==True to get payout.
    # We approximate win_pay = on_odds * 100 for winners (assumes efficient market)
    # This is an approximation — actual payout might differ ±10%.
    for h in horses:
        cost += 100
        if h.get("is_winner"):
            wins += 1
            # Use on_odds as win payout estimate
            payout += 100 * (h.get("odds", 0.0) or 0.0)
    roi = ((payout - cost) / cost) if cost > 0 else 0.0
    return (len(horses), wins, roi)


def analyze(races: list[dict]) -> dict:
    """Run all 5 conditional analyses."""

    # Flatten all horse rows across races
    all_horses = []
    for r in races:
        for h in r["horses"]:
            h2 = dict(h)
            h2["race_id"] = r["race_id"]
            h2["race_grade"] = r["grade"]
            h2["base_top1_top2_gap"] = r["base_top1_top2_gap"]
            h2["has_result"] = r["has_result"]
            all_horses.append(h2)

    total_horses = len(all_horses)
    total_races = len(races)
    races_with_result = sum(1 for r in races if r["has_result"])

    report = {
        "total_races":       total_races,
        "races_with_result": races_with_result,
        "total_horses":      total_horses,
    }

    # ──────────────────────────────────────────────────────
    # Axis 1: Odds band
    # ──────────────────────────────────────────────────────
    by_odds: dict[str, list] = defaultdict(list)
    for h in all_horses:
        by_odds[_odds_band(h["odds"])].append(h)

    odds_report = {}
    for band in ["<3", "3-6", "6-15", ">15", "no-odds"]:
        horses = by_odds.get(band, [])
        if not horses:
            continue
        deltas = [h["score_delta"] for h in horses]
        base_scores = [h["base_score"] for h in horses]
        n, wins, roi = _hit_roi([h for h in horses if h.get("has_result")])
        # Flip potential: how often |score_delta| >= gap to horse above
        # (per race, within same race). Compute per-race locally.
        flip_potential = 0
        for r in races:
            base_ranked = sorted(
                [h for h in r["horses"] if _odds_band(h["odds"]) == band],
                key=lambda x: -x["base_score"])
            for i, h in enumerate(base_ranked):
                if i == 0:
                    continue
                gap_to_above = base_ranked[i-1]["base_score"] - h["base_score"]
                if gap_to_above <= 0:
                    continue
                if abs(h["score_delta"] - base_ranked[i-1]["score_delta"]) >= gap_to_above:
                    flip_potential += 1
        odds_report[band] = {
            "n_horses":         len(horses),
            "mean_score_delta": round(_safe_mean(deltas), 3),
            "max_abs_delta":    round(max(abs(d) for d in deltas), 3) if deltas else 0.0,
            "mean_base_score":  round(_safe_mean(base_scores), 2),
            "flip_potential":   flip_potential,
            "n_winners":        sum(1 for h in horses if h.get("is_winner")),
            "hit_rate":         round(wins / n, 4) if n else 0.0,
            "roi_approx":       round(roi, 4),
        }

    report["odds_band"] = odds_report

    # ──────────────────────────────────────────────────────
    # Axis 2: Close races (base top1-top2 gap < 3pt)
    # ──────────────────────────────────────────────────────
    close_races = [r for r in races if 0 < r["base_top1_top2_gap"] < 3.0]
    close_report = {
        "n_close_races": len(close_races),
        "pct_of_total":  round(len(close_races) / total_races, 4) if total_races else 0.0,
        "flip_happened": 0,
        "flip_examples": [],
    }
    for r in close_races:
        # Did pedigree actually flip top1 vs top2?
        bs = sorted(r["horses"], key=lambda x: -x["base_score"])
        os = sorted(r["horses"], key=lambda x: -x["on_score"])
        if bs[0]["name"] != os[0]["name"]:
            close_report["flip_happened"] += 1
            # What was the effective pedigree advantage?
            pedigree_gap = (
                os[0]["score_delta"] - bs[0]["score_delta"]
            )
            close_report["flip_examples"].append({
                "race_id":    r["race_id"],
                "race_name":  r["race_name"],
                "base_top1":  bs[0]["name"],
                "on_top1":    os[0]["name"],
                "base_gap":   round(r["base_top1_top2_gap"], 3),
                "pedigree_adv": round(pedigree_gap, 3),
                "winner":     r["winner"],
            })

    # For close races that DIDN'T flip, measure how close we came
    near_flip = []
    for r in close_races:
        bs = sorted(r["horses"], key=lambda x: -x["base_score"])
        if len(bs) < 2: continue
        top1, top2 = bs[0], bs[1]
        # pedigree advantage of top2 over top1
        ped_advantage = top2["score_delta"] - top1["score_delta"]
        if ped_advantage > 0 and ped_advantage < r["base_top1_top2_gap"]:
            near_flip.append({
                "race_id": r["race_id"],
                "base_gap": round(r["base_top1_top2_gap"], 3),
                "ped_advantage": round(ped_advantage, 3),
                "shortfall": round(r["base_top1_top2_gap"] - ped_advantage, 3),
            })
    close_report["near_flip_count"] = len(near_flip)
    if near_flip:
        close_report["median_shortfall"] = round(
            _safe_median([n["shortfall"] for n in near_flip]), 3)
    report["close_races"] = close_report

    # ──────────────────────────────────────────────────────
    # Axis 3: Pedigree extremes
    # ──────────────────────────────────────────────────────
    elite = [h for h in all_horses
             if h["pedigree_composite"] > 0.7 and h.get("has_result")]
    poor = [h for h in all_horses
            if h["pedigree_composite"] < 0.3 and h.get("has_result")]
    neutral = [h for h in all_horses
               if 0.45 <= h["pedigree_composite"] <= 0.55 and h.get("has_result")]

    report["pedigree_extremes"] = {
        "elite_n":     len(elite),
        "elite_wins":  sum(1 for h in elite if h["is_winner"]),
        "elite_hit_rate":  round(sum(1 for h in elite if h["is_winner"]) / len(elite), 4) if elite else 0,
        "elite_roi_approx": round(_hit_roi(elite)[2], 4),
        "elite_mean_odds": round(_safe_mean([h["odds"] for h in elite if h["odds"] > 0]), 2),
        "elite_mean_delta": round(_safe_mean([h["score_delta"] for h in elite]), 3),

        "poor_n":      len(poor),
        "poor_wins":   sum(1 for h in poor if h["is_winner"]),
        "poor_hit_rate":   round(sum(1 for h in poor if h["is_winner"]) / len(poor), 4) if poor else 0,
        "poor_roi_approx": round(_hit_roi(poor)[2], 4),
        "poor_mean_odds":  round(_safe_mean([h["odds"] for h in poor if h["odds"] > 0]), 2),
        "poor_mean_delta": round(_safe_mean([h["score_delta"] for h in poor]), 3),

        "neutral_n":   len(neutral),
        "neutral_wins": sum(1 for h in neutral if h["is_winner"]),
        "neutral_hit_rate": round(sum(1 for h in neutral if h["is_winner"]) / len(neutral), 4) if neutral else 0,
    }

    # ──────────────────────────────────────────────────────
    # Axis 4: Camp strength
    # ──────────────────────────────────────────────────────
    strong_camp = [h for h in all_horses
                   if h["camp_composite"] > 0.8 and h.get("has_result")]
    weak_camp = [h for h in all_horses
                 if h["camp_composite"] < 0.4 and h.get("has_result")]

    report["camp_strength"] = {
        "strong_n":      len(strong_camp),
        "strong_wins":   sum(1 for h in strong_camp if h["is_winner"]),
        "strong_hit_rate": round(sum(1 for h in strong_camp if h["is_winner"]) / len(strong_camp), 4) if strong_camp else 0,
        "strong_roi_approx": round(_hit_roi(strong_camp)[2], 4),
        "strong_mean_odds":  round(_safe_mean([h["odds"] for h in strong_camp if h["odds"] > 0]), 2),
        "strong_mean_delta": round(_safe_mean([h["score_delta"] for h in strong_camp]), 3),

        "weak_n":        len(weak_camp),
        "weak_wins":     sum(1 for h in weak_camp if h["is_winner"]),
        "weak_hit_rate": round(sum(1 for h in weak_camp if h["is_winner"]) / len(weak_camp), 4) if weak_camp else 0,
        "weak_roi_approx": round(_hit_roi(weak_camp)[2], 4),
        "weak_mean_odds":  round(_safe_mean([h["odds"] for h in weak_camp if h["odds"] > 0]), 2),
        "weak_mean_delta": round(_safe_mean([h["score_delta"] for h in weak_camp]), 3),
    }

    # ──────────────────────────────────────────────────────
    # Axis 5: Distance fit mismatch
    # ──────────────────────────────────────────────────────
    mismatch = [h for h in all_horses
                if h["sire_distance_fit"] < 0.4 and h.get("has_result")]
    match = [h for h in all_horses
             if h["sire_distance_fit"] > 0.7 and h.get("has_result")]

    report["distance_fit"] = {
        "mismatch_n":     len(mismatch),
        "mismatch_wins":  sum(1 for h in mismatch if h["is_winner"]),
        "mismatch_hit_rate": round(sum(1 for h in mismatch if h["is_winner"]) / len(mismatch), 4) if mismatch else 0,
        "mismatch_roi_approx": round(_hit_roi(mismatch)[2], 4),
        "mismatch_mean_odds":  round(_safe_mean([h["odds"] for h in mismatch if h["odds"] > 0]), 2),
        "mismatch_mean_delta": round(_safe_mean([h["score_delta"] for h in mismatch]), 3),

        "match_n":        len(match),
        "match_wins":     sum(1 for h in match if h["is_winner"]),
        "match_hit_rate": round(sum(1 for h in match if h["is_winner"]) / len(match), 4) if match else 0,
        "match_roi_approx": round(_hit_roi(match)[2], 4),
        "match_mean_odds":  round(_safe_mean([h["odds"] for h in match if h["odds"] > 0]), 2),
        "match_mean_delta": round(_safe_mean([h["score_delta"] for h in match]), 3),
    }

    # ──────────────────────────────────────────────────────
    # Global: base_score gap distribution
    # ──────────────────────────────────────────────────────
    gaps = [r["base_top1_top2_gap"] for r in races if r["base_top1_top2_gap"] > 0]
    report["base_gap_distribution"] = {
        "n_races":  len(gaps),
        "median":   round(_safe_median(gaps), 3) if gaps else 0.0,
        "mean":     round(_safe_mean(gaps), 3) if gaps else 0.0,
        "min":      round(min(gaps), 3) if gaps else 0.0,
        "max":      round(max(gaps), 3) if gaps else 0.0,
        "under_1pt": sum(1 for g in gaps if g < 1.0),
        "under_3pt": sum(1 for g in gaps if g < 3.0),
        "under_5pt": sum(1 for g in gaps if g < 5.0),
    }

    # Global score_delta distribution
    all_deltas = [h["score_delta"] for h in all_horses]
    abs_deltas = [abs(d) for d in all_deltas]
    report["score_delta_distribution"] = {
        "n_horses":       len(all_deltas),
        "mean":           round(_safe_mean(all_deltas), 3),
        "median":         round(_safe_median(all_deltas), 3),
        "min":            round(min(all_deltas), 3) if all_deltas else 0,
        "max":            round(max(all_deltas), 3) if all_deltas else 0,
        "mean_abs":       round(_safe_mean(abs_deltas), 3),
        "max_abs":        round(max(abs_deltas), 3) if abs_deltas else 0,
    }

    return report


def print_report(rep: dict):
    print("=" * 75)
    print("  CONDITIONAL PEDIGREE EFFECTIVENESS ANALYSIS")
    print("=" * 75)
    print(f"  Total races: {rep['total_races']}  "
          f"(with result: {rep['races_with_result']})")
    print(f"  Total horses: {rep['total_horses']}")
    print()

    # Base gap
    bg = rep["base_gap_distribution"]
    print("-- Base score top1-top2 gap distribution --")
    print(f"  median={bg['median']:.2f}  mean={bg['mean']:.2f}  "
          f"min={bg['min']:.2f}  max={bg['max']:.2f}")
    print(f"  races under 1pt: {bg['under_1pt']}")
    print(f"  races under 3pt: {bg['under_3pt']}")
    print(f"  races under 5pt: {bg['under_5pt']}")
    print()

    # Score delta
    sd = rep["score_delta_distribution"]
    print("-- Pedigree score_delta distribution (per horse) --")
    print(f"  mean={sd['mean']:+.3f}  median={sd['median']:+.3f}")
    print(f"  range=[{sd['min']:+.2f}, {sd['max']:+.2f}]")
    print(f"  mean |delta|={sd['mean_abs']:.3f}  max |delta|={sd['max_abs']:.3f}")
    print()

    # Axis 1: Odds bands
    print("-- [Axis 1] Odds band --")
    print(f"  {'band':>8} {'n':>4} {'mean_delta':>11} {'max|Δ|':>8} "
          f"{'mean_base':>10} {'flip_pot':>10} {'winners':>8} {'hit_rate':>9} {'roi_approx':>11}")
    for band in ["<3", "3-6", "6-15", ">15", "no-odds"]:
        d = rep["odds_band"].get(band)
        if not d:
            continue
        print(f"  {band:>8} {d['n_horses']:>4} "
              f"{d['mean_score_delta']:>+11.3f} "
              f"{d['max_abs_delta']:>8.3f} "
              f"{d['mean_base_score']:>10.2f} "
              f"{d['flip_potential']:>10} "
              f"{d['n_winners']:>8} "
              f"{d['hit_rate']*100:>8.1f}% "
              f"{d['roi_approx']*100:>+10.1f}%")
    print()

    # Axis 2: Close races
    cr = rep["close_races"]
    print("-- [Axis 2] Close races (base top1-top2 gap < 3pt) --")
    print(f"  close races: {cr['n_close_races']} ({cr['pct_of_total']*100:.1f}% of total)")
    print(f"  top1 actually flipped by pedigree: {cr['flip_happened']}")
    print(f"  near-flip (top2 would flip with more): {cr['near_flip_count']}")
    if cr.get("median_shortfall") is not None:
        print(f"  median shortfall to flip: {cr['median_shortfall']:.2f}pt")
    if cr["flip_examples"]:
        print(f"  flip examples:")
        for ex in cr["flip_examples"][:5]:
            print(f"    {ex['race_name']}: "
                  f"{ex['base_top1']} -> {ex['on_top1']} "
                  f"(base_gap={ex['base_gap']:.2f} ped_adv={ex['pedigree_adv']:+.2f}) "
                  f"winner={ex['winner']}")
    print()

    # Axis 3: Pedigree extremes
    pe = rep["pedigree_extremes"]
    print("-- [Axis 3] Pedigree extremes --")
    print(f"  elite (>0.7):   n={pe['elite_n']:>3} wins={pe['elite_wins']:>3} "
          f"hit={pe['elite_hit_rate']*100:>5.1f}%  "
          f"ROI~={pe['elite_roi_approx']*100:>+6.1f}%  "
          f"mean_odds={pe['elite_mean_odds']:>5.1f}  "
          f"mean_Δ={pe['elite_mean_delta']:+.2f}")
    print(f"  neutral (~0.5): n={pe['neutral_n']:>3} wins={pe['neutral_wins']:>3} "
          f"hit={pe['neutral_hit_rate']*100:>5.1f}%")
    print(f"  poor (<0.3):    n={pe['poor_n']:>3} wins={pe['poor_wins']:>3} "
          f"hit={pe['poor_hit_rate']*100:>5.1f}%  "
          f"ROI~={pe['poor_roi_approx']*100:>+6.1f}%  "
          f"mean_odds={pe['poor_mean_odds']:>5.1f}  "
          f"mean_Δ={pe['poor_mean_delta']:+.2f}")
    print()

    # Axis 4: Camp
    cs = rep["camp_strength"]
    print("-- [Axis 4] Camp strength --")
    print(f"  strong (>0.8): n={cs['strong_n']:>3} wins={cs['strong_wins']:>3} "
          f"hit={cs['strong_hit_rate']*100:>5.1f}%  "
          f"ROI~={cs['strong_roi_approx']*100:>+6.1f}%  "
          f"mean_odds={cs['strong_mean_odds']:>5.1f}  "
          f"mean_Δ={cs['strong_mean_delta']:+.2f}")
    print(f"  weak (<0.4):   n={cs['weak_n']:>3} wins={cs['weak_wins']:>3} "
          f"hit={cs['weak_hit_rate']*100:>5.1f}%  "
          f"ROI~={cs['weak_roi_approx']*100:>+6.1f}%  "
          f"mean_odds={cs['weak_mean_odds']:>5.1f}  "
          f"mean_Δ={cs['weak_mean_delta']:+.2f}")
    print()

    # Axis 5: Distance fit
    df = rep["distance_fit"]
    print("-- [Axis 5] Sire distance fit --")
    print(f"  mismatch (<0.4): n={df['mismatch_n']:>3} wins={df['mismatch_wins']:>3} "
          f"hit={df['mismatch_hit_rate']*100:>5.1f}%  "
          f"ROI~={df['mismatch_roi_approx']*100:>+6.1f}%  "
          f"mean_odds={df['mismatch_mean_odds']:>5.1f}  "
          f"mean_Δ={df['mismatch_mean_delta']:+.2f}")
    print(f"  match (>0.7):    n={df['match_n']:>3} wins={df['match_wins']:>3} "
          f"hit={df['match_hit_rate']*100:>5.1f}%  "
          f"ROI~={df['match_roi_approx']*100:>+6.1f}%  "
          f"mean_odds={df['match_mean_odds']:>5.1f}  "
          f"mean_Δ={df['match_mean_delta']:+.2f}")


def main() -> int:
    races = load_pairs()
    if not races:
        print("[error] No prediction pairs found. Run snapshot_predict.py first.")
        return 1

    rep = analyze(races)
    print_report(rep)

    out = PROJECT_ROOT / "data" / "pedigree_conditional_analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
