"""Data preparation with strict leakage prevention.

READ-ONLY: This file must NOT be modified by candidate generation.
It loads predictions/results, validates temporal integrity,
and produces train/eval splits for walk-forward evaluation.
"""

import json
import copy
from pathlib import Path
from datetime import datetime
from typing import Any

from data_store import load_predictions, load_results, load_weights

# Features that are available BEFORE race time (<= T-15 minutes)
ALLOWED_PRE_RACE_FEATURES = frozenset([
    "confidence", "ev_gap", "reason", "bet", "rank",  # prediction features
    "odds",  # pre-race fixed odds snapshot
    "grade",  # race grade (G1/G2/G3)
    "race_name",
    "timestamp",
    "structured_features",  # rich pre-race feature dict (T-15 safe)
])

# Features that must NEVER be used in scoring (post-race / future data)
FORBIDDEN_FEATURES = frozenset([
    "finishing_order", "payouts", "result", "final_odds",
    "post_race", "actual_rank", "win_time",
])


def _parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp, fallback to epoch if unparseable."""
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return datetime(2000, 1, 1)


def _extract_year(ts: str) -> int:
    return _parse_timestamp(ts).year


def _sanitize_prediction(pred: dict) -> dict:
    """Remove any forbidden features that might leak future info."""
    clean = {}
    for key, val in pred.items():
        if key in FORBIDDEN_FEATURES:
            continue
        if key == "horses" and isinstance(val, list):
            clean_horses = []
            for h in val:
                clean_h = {k: v for k, v in h.items() if k not in FORBIDDEN_FEATURES}
                clean_horses.append(clean_h)
            clean["horses"] = clean_horses
        else:
            clean[key] = val
    return clean


def load_paired_data() -> list[dict]:
    """Load prediction-result pairs, sorted by timestamp.

    Each entry:
    {
        "race_id": str,
        "prediction": dict (sanitized, no future data),
        "result": dict (full result for evaluation only),
        "timestamp": str,
        "year": int,
    }

    Returns only races that have BOTH prediction AND result.
    """
    predictions = load_predictions()
    results = load_results()

    paired = []
    for race_id, pred in predictions.items():
        result = results.get(race_id)
        if not result:
            continue
        if not pred.get("horses"):
            continue

        ts = pred.get("timestamp", "")
        paired.append({
            "race_id": race_id,
            "prediction": _sanitize_prediction(pred),
            "result": copy.deepcopy(result),
            "timestamp": ts,
            "year": _extract_year(ts),
        })

    paired.sort(key=lambda x: x["timestamp"])
    return paired


def walk_forward_splits(
    data: list[dict],
    min_train_size: int = 10,
    step_size: int = 1,
) -> list[dict]:
    """Generate expanding-window walk-forward splits.

    Each split:
    {
        "fold": int,
        "train": list[dict],  # all data up to cutoff
        "test": list[dict],   # next step_size entries
        "cutoff_ts": str,
    }

    This ensures training data NEVER includes future races.
    """
    if len(data) < min_train_size + step_size:
        # Not enough data for even one fold; return single split
        if len(data) >= 2:
            split_point = max(1, len(data) - 1)
            return [{
                "fold": 0,
                "train": data[:split_point],
                "test": data[split_point:],
                "cutoff_ts": data[split_point - 1]["timestamp"],
            }]
        return []

    splits = []
    fold = 0
    for i in range(min_train_size, len(data), step_size):
        train = data[:i]
        test = data[i:i + step_size]
        if not test:
            break
        splits.append({
            "fold": fold,
            "train": train,
            "test": test,
            "cutoff_ts": data[i - 1]["timestamp"],
        })
        fold += 1

    return splits


def get_yearly_splits(data: list[dict]) -> dict[int, list[dict]]:
    """Group data by year for annual consistency checks."""
    yearly = {}
    for entry in data:
        yr = entry["year"]
        yearly.setdefault(yr, []).append(entry)
    return yearly


def extract_features(prediction: dict) -> dict:
    """Extract numeric features from a sanitized prediction.

    Returns feature dict usable by score_runner().
    Features are strictly pre-race only.

    If the prediction contains a 'structured_features' dict (rich
    pre-race data persisted alongside Gemini output), it is passed
    through as-is for score_runner() to consume.
    """
    horses = prediction.get("horses", [])
    features = {
        "grade": prediction.get("grade", ""),
        "race_name": prediction.get("race_name", ""),
        "num_horses": len(horses),
        "horse_features": [],
    }

    for h in horses:
        hf = {
            "name": h.get("name", ""),
            "rank": h.get("rank", 0),
            "confidence": h.get("confidence", 0),
            "ev_gap": _parse_ev_gap(h.get("ev_gap", "0")),
            "odds": _parse_odds(h.get("odds", 0)),
            "bet": h.get("bet", ""),
        }
        features["horse_features"].append(hf)

    # Pass through structured pre-race features if present
    sf = prediction.get("structured_features")
    if sf and isinstance(sf, dict):
        features["structured_features"] = sf

    return features


def _parse_ev_gap(val) -> float:
    """Parse ev_gap string like '+12' or '-2' to float."""
    try:
        return float(str(val).replace("+", "").strip())
    except (ValueError, TypeError):
        return 0.0


def _parse_odds(val) -> float:
    """Parse odds value, handling Japanese format."""
    try:
        return float(str(val).replace("倍", "").strip())
    except (ValueError, TypeError):
        return 0.0


def get_context() -> dict:
    """Return context dict (current weights) for score_runner."""
    return {
        "weights": load_weights(),
    }


def validate_no_leakage(features: dict, result: dict) -> bool:
    """Verify that features contain no data from the result.

    Returns True if clean, False if leakage detected.
    """
    features_str = json.dumps(features, ensure_ascii=False)
    finishing = result.get("finishing_order", [])

    # Check that actual finishing positions don't appear in features
    for h in finishing:
        actual_rank = h.get("rank", 0)
        actual_time = h.get("time", "")
        if actual_time and actual_time in features_str:
            return False

    # Check forbidden keys
    for key in FORBIDDEN_FEATURES:
        if key in features_str:
            return False

    return True
