"""Walk-forward evaluator with strict validation gates.

READ-ONLY: This file must NOT be modified by candidate generation.
It computes baseline metrics, evaluates candidates, and enforces
all adoption conditions (performance, safety, consistency, anti-bias, significance).
"""

import math
import copy
from typing import Callable

from prepare import (
    load_paired_data, walk_forward_splits, get_yearly_splits,
    extract_features, get_context, validate_no_leakage,
)


# --- Metric Functions ---

def compute_brier_score(predicted_probs: list[float], actuals: list[int]) -> float:
    """Brier score: mean squared error of probability predictions.

    predicted_probs: list of predicted probability for top pick winning
    actuals: list of 1 (top pick won) or 0 (didn't win)

    Lower is better. Range [0, 1].
    """
    if not predicted_probs:
        return 1.0
    n = len(predicted_probs)
    return sum((p - a) ** 2 for p, a in zip(predicted_probs, actuals)) / n


def compute_roi(bets: list[dict]) -> float:
    """Compute ROI from bet results.

    Each bet: {"stake": float, "payout": float}
    ROI = (total_payout - total_stake) / total_stake

    Returns 0.0 if no bets.
    """
    total_stake = sum(b["stake"] for b in bets)
    total_payout = sum(b["payout"] for b in bets)
    if total_stake == 0:
        return 0.0
    return (total_payout - total_stake) / total_stake


def compute_max_drawdown(cumulative_returns: list[float]) -> float:
    """Maximum drawdown from peak.

    cumulative_returns: list of cumulative return values.
    Returns max drawdown as a positive fraction (0.15 = 15% drawdown).
    """
    if not cumulative_returns:
        return 0.0
    peak = cumulative_returns[0]
    max_dd = 0.0
    for val in cumulative_returns:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return max_dd


def _confidence_to_prob(confidence: float) -> float:
    """Convert confidence (0-100) to probability (0-1)."""
    return max(0.01, min(0.99, confidence / 100.0))


def _evaluate_single_race(
    entry: dict,
    score_fn: Callable,
    context: dict,
    mask_odds: bool = False,
) -> dict:
    """Evaluate a single race using score_fn.

    Returns:
    {
        "race_id": str,
        "predicted_prob": float,
        "actual_win": int (1/0),
        "bet": {"stake": float, "payout": float},
        "cumulative_pnl": float (set later),
    }
    """
    features = extract_features(entry["prediction"])
    result = entry["result"]

    # Leakage check
    if not validate_no_leakage(features, result):
        raise ValueError(f"Data leakage detected in race {entry['race_id']}")

    # Mask odds if running anti-bias test
    if mask_odds:
        for hf in features.get("horse_features", []):
            hf["odds"] = 0.0
            hf["ev_gap"] = 0.0

    # Run scoring function
    scores = score_fn(features, context)

    # Extract top pick info
    horses = entry["prediction"].get("horses", [])
    if not horses:
        return {
            "race_id": entry["race_id"],
            "predicted_prob": 0.0,
            "actual_win": 0,
            "bet": {"stake": 0, "payout": 0},
        }

    top_horse = horses[0]
    top_name = top_horse.get("name", "")
    confidence = scores.get("top_confidence", top_horse.get("confidence", 50))
    predicted_prob = _confidence_to_prob(confidence)

    # Check actual result
    finishing = {h["name"]: h["rank"] for h in result.get("finishing_order", [])}
    actual_win = 1 if finishing.get(top_name) == 1 else 0

    # Compute bet result (unit stake model: bet 100 on every recommended race)
    stake = 100
    payout = 0
    if actual_win and top_horse.get("odds"):
        try:
            odds_val = float(str(top_horse["odds"]).replace("倍", "").strip())
            payout = stake * odds_val
        except (ValueError, TypeError):
            pass

    return {
        "race_id": entry["race_id"],
        "year": entry["year"],
        "predicted_prob": predicted_prob,
        "actual_win": actual_win,
        "bet": {"stake": stake, "payout": payout},
    }


def evaluate_walk_forward(
    score_fn: Callable,
    mask_odds: bool = False,
) -> dict:
    """Run full walk-forward evaluation.

    Returns:
    {
        "roi": float,
        "brier": float,
        "max_drawdown": float,
        "num_races": int,
        "fold_results": list[dict],
        "yearly_results": dict[int, dict],
        "race_results": list[dict],
    }
    """
    data = load_paired_data()
    context = get_context()

    if not data:
        return _empty_result()

    # Evaluate all races
    race_results = []
    for entry in data:
        try:
            r = _evaluate_single_race(entry, score_fn, context, mask_odds=mask_odds)
            race_results.append(r)
        except ValueError as e:
            # Leakage detected - fail entire evaluation
            return {"error": str(e), **_empty_result()}

    if not race_results:
        return _empty_result()

    # Compute cumulative P&L for drawdown
    cumulative = []
    running_pnl = 0
    for r in race_results:
        running_pnl += r["bet"]["payout"] - r["bet"]["stake"]
        cumulative.append(running_pnl)
        r["cumulative_pnl"] = running_pnl

    # Aggregate metrics
    predicted_probs = [r["predicted_prob"] for r in race_results]
    actuals = [r["actual_win"] for r in race_results]
    bets = [r["bet"] for r in race_results]

    roi = compute_roi(bets)
    brier = compute_brier_score(predicted_probs, actuals)
    max_dd = compute_max_drawdown([1000 + c for c in cumulative])  # base capital 1000

    # Fold-level results (walk-forward)
    splits = walk_forward_splits(data)
    fold_results = []
    for split in splits:
        test_ids = {e["race_id"] for e in split["test"]}
        fold_races = [r for r in race_results if r["race_id"] in test_ids]
        if fold_races:
            fold_probs = [r["predicted_prob"] for r in fold_races]
            fold_actuals = [r["actual_win"] for r in fold_races]
            fold_bets = [r["bet"] for r in fold_races]
            fold_results.append({
                "fold": split["fold"],
                "num_races": len(fold_races),
                "roi": compute_roi(fold_bets),
                "brier": compute_brier_score(fold_probs, fold_actuals),
                "cutoff_ts": split["cutoff_ts"],
            })

    # Yearly results
    yearly_data = get_yearly_splits(data)
    yearly_results = {}
    for year, entries in yearly_data.items():
        year_ids = {e["race_id"] for e in entries}
        year_races = [r for r in race_results if r["race_id"] in year_ids]
        if year_races:
            yr_probs = [r["predicted_prob"] for r in year_races]
            yr_actuals = [r["actual_win"] for r in year_races]
            yr_bets = [r["bet"] for r in year_races]
            yr_cum = []
            yr_pnl = 0
            for r in year_races:
                yr_pnl += r["bet"]["payout"] - r["bet"]["stake"]
                yr_cum.append(yr_pnl)
            yearly_results[year] = {
                "num_races": len(year_races),
                "roi": compute_roi(yr_bets),
                "brier": compute_brier_score(yr_probs, yr_actuals),
                "max_drawdown": compute_max_drawdown([1000 + c for c in yr_cum]),
                "hit_rate": sum(yr_actuals) / len(yr_actuals) if yr_actuals else 0,
            }

    return {
        "roi": roi,
        "brier": brier,
        "max_drawdown": max_dd,
        "num_races": len(race_results),
        "fold_results": fold_results,
        "yearly_results": yearly_results,
        "race_results": race_results,
    }


def _empty_result() -> dict:
    return {
        "roi": 0.0,
        "brier": 1.0,
        "max_drawdown": 0.0,
        "num_races": 0,
        "fold_results": [],
        "yearly_results": {},
        "race_results": [],
    }


# --- Baseline ---

def baseline_score_runner(features: dict, context: dict) -> dict:
    """Baseline scoring: use raw confidence from prediction as-is.

    This represents the current production behavior.
    """
    horses = features.get("horse_features", [])
    if not horses:
        return {"top_confidence": 50}
    return {"top_confidence": horses[0].get("confidence", 50)}


def get_baseline_metrics() -> dict:
    """Compute baseline metrics using current production logic."""
    return evaluate_walk_forward(baseline_score_runner, mask_odds=False)


# --- Adoption Gate ---

def check_adoption(
    candidate_metrics: dict,
    baseline_metrics: dict,
    candidate_masked: dict,
    min_races: int = 300,
) -> dict:
    """Check ALL adoption conditions.

    Returns:
    {
        "adopted": bool,
        "checks": {
            "performance": {"passed": bool, "details": str},
            "safety": {"passed": bool, "details": str},
            "consistency": {"passed": bool, "details": str},
            "anti_bias": {"passed": bool, "details": str},
            "significance": {"passed": bool, "details": str},
        },
        "reason": str (if rejected),
    }
    """
    checks = {}

    # A. Performance: ROI >= baseline AND Brier <= baseline
    perf_roi = candidate_metrics["roi"] >= baseline_metrics["roi"]
    perf_brier = candidate_metrics["brier"] <= baseline_metrics["brier"]
    checks["performance"] = {
        "passed": perf_roi and perf_brier,
        "details": (
            f"ROI: {candidate_metrics['roi']:.4f} vs baseline {baseline_metrics['roi']:.4f} "
            f"({'PASS' if perf_roi else 'FAIL'}), "
            f"Brier: {candidate_metrics['brier']:.4f} vs baseline {baseline_metrics['brier']:.4f} "
            f"({'PASS' if perf_brier else 'FAIL'})"
        ),
    }

    # B. Safety: MaxDrawdown <= max(baseline_MDD * 1.1, 0.15)
    baseline_mdd = baseline_metrics["max_drawdown"]
    mdd_threshold = max(baseline_mdd * 1.1, 0.15)
    candidate_mdd = candidate_metrics["max_drawdown"]
    checks["safety"] = {
        "passed": candidate_mdd <= mdd_threshold,
        "details": (
            f"MaxDrawdown: {candidate_mdd:.4f} vs threshold {mdd_threshold:.4f} "
            f"(baseline MDD: {baseline_mdd:.4f}) "
            f"({'PASS' if candidate_mdd <= mdd_threshold else 'FAIL'})"
        ),
    }

    # C. Consistency: walk-forward improvement rate >= 0.7
    # AND at least 2 of last 3 years improved
    b_folds = baseline_metrics.get("fold_results", [])
    c_folds = candidate_metrics.get("fold_results", [])
    improved_folds = 0
    total_folds = min(len(b_folds), len(c_folds))
    for i in range(total_folds):
        b_roi = b_folds[i]["roi"] if i < len(b_folds) else 0
        c_roi = c_folds[i]["roi"] if i < len(c_folds) else 0
        if c_roi >= b_roi:
            improved_folds += 1
    wf_rate = improved_folds / total_folds if total_folds > 0 else 0

    # Yearly consistency
    b_yearly = baseline_metrics.get("yearly_results", {})
    c_yearly = candidate_metrics.get("yearly_results", {})
    years_sorted = sorted(set(b_yearly.keys()) & set(c_yearly.keys()))
    last_3_years = years_sorted[-3:] if len(years_sorted) >= 3 else years_sorted
    yearly_pass_count = 0
    for yr in last_3_years:
        roi_improved = c_yearly[yr]["roi"] >= b_yearly[yr]["roi"]
        brier_not_worse = c_yearly[yr]["brier"] <= b_yearly[yr]["brier"]
        if roi_improved and brier_not_worse:
            yearly_pass_count += 1

    yearly_ok = yearly_pass_count >= 2 if len(last_3_years) >= 3 else yearly_pass_count >= len(last_3_years)

    checks["consistency"] = {
        "passed": wf_rate >= 0.7 and yearly_ok,
        "details": (
            f"WalkForward improvement rate: {wf_rate:.2f} ({'PASS' if wf_rate >= 0.7 else 'FAIL'}), "
            f"Yearly pass: {yearly_pass_count}/{len(last_3_years)} "
            f"({'PASS' if yearly_ok else 'FAIL'})"
        ),
    }

    # D. Anti-bias (odds mask test)
    if candidate_masked.get("error"):
        checks["anti_bias"] = {
            "passed": False,
            "details": f"Masked evaluation error: {candidate_masked['error']}",
        }
    else:
        # Composite = -Brier (lower is better) + ROI (higher is better)
        composite_normal = candidate_metrics["roi"] - candidate_metrics["brier"]
        composite_masked = candidate_masked["roi"] - candidate_masked["brier"]

        ratio = composite_masked / composite_normal if composite_normal != 0 else 0
        roi_masked_ok = candidate_masked["roi"] >= baseline_metrics["roi"] * 0.9
        brier_masked_ok = candidate_masked["brier"] <= baseline_metrics["brier"] * 1.05

        checks["anti_bias"] = {
            "passed": ratio >= 0.8 and roi_masked_ok and brier_masked_ok,
            "details": (
                f"Composite ratio (masked/normal): {ratio:.4f} ({'PASS' if ratio >= 0.8 else 'FAIL'}), "
                f"ROI_masked: {candidate_masked['roi']:.4f} vs threshold {baseline_metrics['roi'] * 0.9:.4f} "
                f"({'PASS' if roi_masked_ok else 'FAIL'}), "
                f"Brier_masked: {candidate_masked['brier']:.4f} vs threshold {baseline_metrics['brier'] * 1.05:.4f} "
                f"({'PASS' if brier_masked_ok else 'FAIL'})"
            ),
        }

    # E. Significance: num_races >= min_races
    num_races = candidate_metrics["num_races"]
    checks["significance"] = {
        "passed": num_races >= min_races,
        "details": (
            f"Num races: {num_races} vs required {min_races} "
            f"({'PASS' if num_races >= min_races else 'FAIL'})"
        ),
    }

    # Overall
    all_passed = all(c["passed"] for c in checks.values())
    reason = ""
    if not all_passed:
        failed = [k for k, v in checks.items() if not v["passed"]]
        reason = f"DISCARD: Failed checks: {', '.join(failed)}"

    return {
        "adopted": all_passed,
        "checks": checks,
        "reason": reason,
    }
