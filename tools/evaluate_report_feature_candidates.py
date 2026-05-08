"""Evaluate feature ideas from the profitability report before implementation.

This script does not change production prediction logic.  It takes the
existing backtest predictions, applies small feature-based re-rankers,
then evaluates the user's 600-yen ticket:

  - 100 yen win on the selected 3 horses
  - 100 yen quinella box over the selected 3 horses

The point is to reject attractive-sounding ideas unless they improve ROI
on the historical snapshots.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Callable

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import grade_strategy as gs


BACKTEST_DIR = ROOT / "data" / "backtest_predictions"
RESULTS_FILE = ROOT / "data" / "results.json"
ENRICHED_FILE = ROOT / "data" / "enriched_backtest_results_v3.json"
TICKET = 100.0
COST_PER_RACE = 600.0


def _load_results() -> dict:
    raw = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {(k[3:] if k.startswith("bt_") else k): v for k, v in raw.items()}


def _load_structured_features() -> dict:
    if not ENRICHED_FILE.exists():
        return {}
    raw = json.loads(ENRICHED_FILE.read_text(encoding="utf-8"))
    return {row.get("race_id", ""): row.get("structured_features") for row in raw}


def _parse_float(raw) -> float:
    try:
        return float(str(raw).replace(",", "").replace("--", "0").replace("---", "0"))
    except (TypeError, ValueError):
        return 0.0


def _rank_map(result: dict) -> dict[str, int]:
    out = {}
    for horse in result.get("finishing_order") or []:
        name = (horse.get("name") or "").strip()
        try:
            rank = int(horse.get("rank", 0) or 0)
        except (TypeError, ValueError):
            continue
        if name and rank > 0:
            out[name] = rank
    return out


def _payout(result: dict, key: str) -> float:
    return _parse_float((result.get("payouts") or {}).get(key))


def _grade_bucket(grade: str) -> str:
    g = (grade or "").upper()
    if "G1" in g or g == "GI":
        return "G1"
    if "G2" in g or g == "GII":
        return "G2"
    if "G3" in g or g == "GIII":
        return "G3"
    return "OTHER"


def _features(pred: dict, name: str) -> dict:
    return ((pred.get("structured_features") or {}).get("horses") or {}).get(name, {})


def _race_features(pred: dict) -> dict:
    return (pred.get("structured_features") or {}).get("race") or {}


def _field_values(pred: dict, key: str) -> list[float]:
    vals = []
    horses = (pred.get("structured_features") or {}).get("horses") or {}
    for feat in horses.values():
        value = _parse_float(feat.get(key))
        if value:
            vals.append(value)
    return vals


def _base_selected(pred: dict, ranked: list[dict]) -> list[dict]:
    strategy = gs.get_strategy_for_grade(pred.get("grade", ""))
    if strategy == "win_prob":
        return ranked[:3]
    market = gs.build_market_rank_map(ranked)
    return gs.pick_diversified_top3(ranked, market, strategy=strategy)[:3]


def _score_with_feature_bonus(
    pred: dict,
    bonus_fn: Callable[[dict, dict, dict], float],
    strength: float = 1.0,
    feature_only: bool = False,
) -> list[dict]:
    race = _race_features(pred)
    reranked = []
    for horse in pred.get("ranked") or []:
        name = (horse.get("name") or "").strip()
        feat = _features(pred, name)
        base = float(horse.get("win_prob", 0) or 0)
        bonus = bonus_fn(horse, feat, race)
        if feature_only:
            score = bonus + base * 0.001
        else:
            score = base * (1.0 + bonus * strength)
        reranked.append({**horse, "_feature_score": score})
    reranked.sort(key=lambda h: h.get("_feature_score", 0), reverse=True)
    return reranked


def _evaluate_one(pred: dict, result: dict, selected: list[dict]) -> dict | None:
    if len(selected) < 3:
        return None
    rank = _rank_map(result)
    if not rank:
        return None
    winner = next((name for name, r in rank.items() if r == 1), "")
    second = next((name for name, r in rank.items() if r == 2), "")
    if not winner:
        return None

    top3 = [(h.get("name") or "").strip() for h in selected[:3]]
    top3_set = set(top3)
    tansho = _payout(result, "単勝") if winner in top3_set else 0.0
    umaren = 0.0
    if second:
        combos = {frozenset(c) for c in combinations(top3, 2)}
        if frozenset([winner, second]) in combos:
            umaren = _payout(result, "馬連")

    payout = tansho + umaren
    return {
        "race_id": pred.get("race_id", ""),
        "race_name": pred.get("race_name", ""),
        "grade": _grade_bucket(pred.get("grade", "")),
        "top3": top3,
        "winner": winner,
        "second": second,
        "tansho": tansho,
        "umaren": umaren,
        "payout": payout,
        "pnl": payout - COST_PER_RACE,
    }


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    cost = len(records) * COST_PER_RACE
    payout = sum(r["payout"] for r in records)
    pnl = payout - cost
    sorted_recs = sorted(records, key=lambda r: r["pnl"], reverse=True)
    ex_big2 = sorted_recs[2:] if len(sorted_recs) > 2 else []
    ex_cost = len(ex_big2) * COST_PER_RACE
    ex_payout = sum(r["payout"] for r in ex_big2)
    return {
        "n": len(records),
        "cost": cost,
        "payout": payout,
        "pnl": pnl,
        "roi": pnl / cost if cost else 0.0,
        "tansho_hit": sum(1 for r in records if r["tansho"] > 0) / len(records),
        "umaren_hit": sum(1 for r in records if r["umaren"] > 0) / len(records),
        "ex_big2_roi": ((ex_payout - ex_cost) / ex_cost) if ex_cost else 0.0,
    }


def candidate_bonus_functions(pred: dict) -> dict[str, Callable[[dict, dict, dict], float]]:
    carried_vals = _field_values(pred, "carried_weight")
    weight_deltas = [abs(v) for v in _field_values(pred, "horse_weight_delta")]
    jockey_vals = _field_values(pred, "jockey_win_rate")
    trainer_vals = _field_values(pred, "trainer_win_rate")
    cardio_vals = _field_values(pred, "training_cardio_index")
    camp_vals = _field_values(pred, "camp_composite")
    ped_vals = _field_values(pred, "pedigree_composite")

    carried_med = median(carried_vals) if carried_vals else 0.0
    delta_med = median(weight_deltas) if weight_deltas else 0.0
    jockey_med = median(jockey_vals) if jockey_vals else 0.0
    trainer_med = median(trainer_vals) if trainer_vals else 0.0
    cardio_med = median(cardio_vals) if cardio_vals else 0.0
    camp_med = median(camp_vals) if camp_vals else 0.0
    ped_med = median(ped_vals) if ped_vals else 0.0

    def relative_weight(horse, feat, race):
        carried = _parse_float(feat.get("carried_weight"))
        distance = _parse_float(race.get("distance"))
        if not carried or not carried_med:
            return 0.0
        # Give a small bonus to horses below field median, amplified at distance.
        scale = 1.5 if distance >= 2000 else 1.0
        return 0.06 * scale if carried <= carried_med - 1.0 else 0.0

    def stable_body_delta(horse, feat, race):
        delta = abs(_parse_float(feat.get("horse_weight_delta")))
        if not delta:
            return 0.0
        return 0.07 if delta <= min(6.0, delta_med or 6.0) else -0.04

    def jockey_trainer_combo(horse, feat, race):
        bonus = 0.0
        if _parse_float(feat.get("jockey_win_rate")) >= jockey_med and jockey_med > 0:
            bonus += 0.04
        if _parse_float(feat.get("trainer_win_rate")) >= trainer_med and trainer_med > 0:
            bonus += 0.03
        return bonus

    def training_condition(horse, feat, race):
        cardio = _parse_float(feat.get("training_cardio_index"))
        accel = _parse_float(feat.get("training_acceleration"))
        bonus = 0.0
        if cardio >= cardio_med and cardio_med > 0:
            bonus += 0.05
        if accel > 0:
            bonus += 0.03
        return bonus

    def pedigree_camp(horse, feat, race):
        bonus = 0.0
        if _parse_float(feat.get("pedigree_composite")) >= ped_med and ped_med > 0:
            bonus += 0.04
        if _parse_float(feat.get("camp_composite")) >= camp_med and camp_med > 0:
            bonus += 0.04
        return bonus

    def combined_conservative(horse, feat, race):
        return min(
            0.12,
            0.5 * relative_weight(horse, feat, race)
            + 0.6 * stable_body_delta(horse, feat, race)
            + 0.7 * jockey_trainer_combo(horse, feat, race)
            + 0.7 * training_condition(horse, feat, race)
            + 0.5 * pedigree_camp(horse, feat, race),
        )

    return {
        "relative_weight": relative_weight,
        "stable_body_delta": stable_body_delta,
        "jockey_trainer_combo": jockey_trainer_combo,
        "training_condition": training_condition,
        "pedigree_camp": pedigree_camp,
        "combined_conservative": combined_conservative,
    }


def run_candidate(candidate: str, preds: list[dict], results: dict) -> list[dict]:
    records = []
    for pred in preds:
        result = results.get(pred.get("race_id", ""))
        if not result:
            continue
        if candidate == "baseline":
            ranked = pred.get("ranked") or []
        else:
            feature_only = candidate.startswith("feature_only_")
            strong = candidate.startswith("strong_")
            key = candidate.replace("feature_only_", "").replace("strong_", "")
            bonus = candidate_bonus_functions(pred)[key]
            ranked = _score_with_feature_bonus(
                pred,
                bonus,
                strength=5.0 if strong else 1.0,
                feature_only=feature_only,
            )
        selected = _base_selected(pred, ranked)
        rec = _evaluate_one(pred, result, selected)
        if rec:
            records.append(rec)
    return records


def main() -> int:
    results = _load_results()
    structured_by_race = _load_structured_features()
    preds = []
    for path in sorted(BACKTEST_DIR.glob("*_on.json")):
        try:
            pred = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _grade_bucket(pred.get("grade", "")) in {"G1", "G2"}:
            if not pred.get("structured_features"):
                sf = structured_by_race.get(pred.get("race_id", ""))
                if sf:
                    pred["structured_features"] = sf
            preds.append(pred)

    candidates = [
        "baseline",
        "relative_weight",
        "stable_body_delta",
        "jockey_trainer_combo",
        "training_condition",
        "pedigree_camp",
        "combined_conservative",
        "strong_combined_conservative",
        "feature_only_relative_weight",
        "feature_only_stable_body_delta",
        "feature_only_jockey_trainer_combo",
        "feature_only_training_condition",
        "feature_only_pedigree_camp",
        "feature_only_combined_conservative",
    ]
    def print_scope(label: str, scope_preds: list[dict]) -> None:
        print("=" * 88)
        print(label)
        print("=" * 88)
        print(f"{'candidate':<32s} {'n':>4s} {'ROI%':>9s} {'exBig2%':>9s} {'単勝%':>8s} {'馬連%':>8s} {'PnL':>9s}")
        print("-" * 88)
        baseline_roi = None
        for candidate in candidates:
            recs = run_candidate(candidate, scope_preds, results)
            agg = aggregate(recs)
            if baseline_roi is None:
                baseline_roi = agg["roi"]
            marker = ""
            if candidate != "baseline":
                marker = f" Δ{(agg['roi'] - baseline_roi)*100:+.1f}pp"
            print(
                f"{candidate:<32s}"
                f" {agg['n']:>4d}"
                f" {agg['roi']*100:>+8.1f}%"
                f" {agg['ex_big2_roi']*100:>+8.1f}%"
                f" {agg['tansho_hit']*100:>7.1f}%"
                f" {agg['umaren_hit']*100:>7.1f}%"
                f" {agg['pnl']:>+9.0f}"
                f"{marker}"
            )
        print()

    print("REPORT FEATURE CANDIDATE EVALUATION")
    print("ticket = win x3 + quinella box x3 = 600 yen/race")
    print()
    print_scope("G1/G2", preds)
    print_scope("G1 only", [p for p in preds if _grade_bucket(p.get("grade", "")) == "G1"])
    print_scope("G2 only", [p for p in preds if _grade_bucket(p.get("grade", "")) == "G2"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
