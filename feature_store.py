"""Structured pre-race feature extraction for persistence.

Extracts T-15-safe features from enriched entry data and race info,
producing a structured_features dict suitable for storage alongside
prediction records.

FEATURE POLICY:
    Only features available >= 15 minutes before race start.
    No post-race data. No Gemini-derived scores.

SCHEMA:
    structured_features = {
        "race": {race-level features},
        "horses": {name: {per-horse features}},
        "version": int,  # schema version for migration
    }
"""

from __future__ import annotations

import re


# Schema version — increment when adding/removing fields
SCHEMA_VERSION = 1

# Fields that must NEVER appear (post-race / Gemini-derived)
_FORBIDDEN_KEYS = frozenset([
    "finishing_order", "payouts", "result", "final_odds",
    "actual_rank", "win_time", "post_race",
    "confidence", "ev_gap", "reason", "bet", "gemini_comment",
])


def extract_structured_features(
    entries: list[dict],
    race_info: dict | None = None,
    track_condition: str = "",
    weather: str = "",
    temperature: str = "",
    cushion_value: str = "",
    venue: str = "",
) -> dict:
    """Extract structured pre-race features from enriched entries.

    Args:
        entries: list of enriched horse dicts from scraper
        race_info: dict from fetch_race_info (optional)
        track_condition: going condition string
        weather: weather string
        temperature: temperature string
        cushion_value: cushion value string
        venue: venue name

    Returns:
        structured_features dict ready for persistence
    """
    race_features = _extract_race_features(
        race_info=race_info,
        track_condition=track_condition,
        weather=weather,
        temperature=temperature,
        cushion_value=cushion_value,
        venue=venue,
        num_horses=len(entries),
    )

    horse_features = {}
    for h in entries:
        name = h.get("name", "")
        if not name:
            continue
        horse_features[name] = _extract_horse_features(h)

    result = {
        "version": SCHEMA_VERSION,
        "race": race_features,
        "horses": horse_features,
    }

    # Safety check: ensure no forbidden keys leaked in
    _validate_no_forbidden(result)

    return result


def _extract_race_features(
    race_info: dict | None,
    track_condition: str,
    weather: str,
    temperature: str,
    cushion_value: str,
    venue: str,
    num_horses: int,
) -> dict:
    """Extract race-level features."""
    info = race_info or {}

    # Parse distance from race_info surface field (e.g., "芝2000m")
    distance = 0
    surface = ""
    surface_text = info.get("surface", "")
    m = re.search(r"(芝|ダート)(\d+)", surface_text)
    if m:
        surface = m.group(1)
        distance = int(m.group(2))

    # Parse temperature to float
    temp_val = _safe_float(temperature or info.get("temperature", ""))

    # Parse cushion value to float
    cushion_val = _safe_float(cushion_value or info.get("cushion_value", ""))

    return {
        "venue": venue,
        "surface": surface,           # "芝" or "ダート"
        "distance": distance,         # meters (int)
        "track_condition": track_condition or info.get("track_condition", ""),
        "weather": weather or info.get("weather", ""),
        "temperature": temp_val,      # celsius (float, 0 if unknown)
        "cushion_value": cushion_val,  # JRA cushion value (float, 0 if unknown)
        "num_horses": num_horses,
    }


def _extract_horse_features(h: dict) -> dict:
    """Extract per-horse features from an enriched entry dict.

    All features are pre-race (T-15 safe).
    """
    features = {}

    # Basic entry data
    features["number"] = _safe_int(h.get("number", 0))
    features["waku"] = _safe_int(h.get("waku", 0))
    features["odds"] = _parse_odds(h.get("odds", "0"))

    # Age / sex
    age_str = h.get("age", "")
    features["age_str"] = age_str
    features["age"] = _parse_age(age_str)

    # Carried weight (斤量)
    features["carried_weight"] = _safe_float(h.get("weight", "0"))

    # Horse body weight and delta
    hw_str = h.get("horse_weight", "")
    weight_kg, weight_delta = _parse_horse_weight(hw_str)
    features["horse_weight_kg"] = weight_kg
    features["horse_weight_delta"] = weight_delta

    # Stable / transport
    features["stable"] = h.get("stable", "")
    features["transport_stress"] = h.get("transport_stress", "")

    # Jockey stats (pre-race public data)
    features["jockey_win_rate"] = _parse_percentage(h.get("jockey_win_rate", ""))
    features["jockey_g1_wins"] = _parse_wins(h.get("jockey_g1_wins", ""))

    # Trainer stats
    features["trainer_win_rate"] = _parse_percentage(h.get("trainer_win_rate", ""))

    # Training physics (computed from pre-race training times)
    tp = h.get("training_physics", {})
    features["training_final_split"] = _safe_float(tp.get("final_split", 0))
    features["training_acceleration"] = _safe_float(tp.get("acceleration_rate", 0))
    features["training_cardio_index"] = _safe_float(tp.get("cardio_index", 0))

    # Training NLP scores (from pre-race evaluation text)
    tnlp = h.get("training_nlp", {})
    features["training_coat_gloss"] = _safe_float(tnlp.get("coat_gloss", 0))
    features["training_stride_quality"] = _safe_float(tnlp.get("stride_quality", 0))
    features["training_weight_status"] = _safe_float(tnlp.get("weight_status", 0))

    # Paddock scores (from pre-race paddock observation)
    ps = h.get("paddock_scores", {})
    features["paddock_vascularity"] = _safe_float(ps.get("vascularity_index", 0))
    features["paddock_hindquarter"] = _safe_float(ps.get("hindquarter_power", 0))
    features["paddock_gait"] = _safe_float(ps.get("gait_fluidity", 0))

    # Best weight analysis (pre-race: how close to historical best weight)
    bwa = h.get("best_weight_analysis", {})
    features["best_weight_diff"] = _safe_float(bwa.get("diff_from_best", 0))
    features["best_weight_count"] = _safe_int(bwa.get("record_count", 0))

    # External farm (外厩)
    features["ritto"] = h.get("ritto", "")

    # Recent form summary (pre-race public info)
    # Store as-is string for now; future versions can parse
    features["recent_form"] = h.get("recent_form", "")

    # Weight trend (pre-race public info)
    features["weight_trend"] = h.get("weight_trend", "")

    return features


# ── Parsing utilities ──

def _safe_float(val) -> float:
    try:
        return float(str(val).strip())
    except (ValueError, TypeError):
        return 0.0


def _safe_int(val) -> int:
    try:
        return int(str(val).strip())
    except (ValueError, TypeError):
        return 0


def _parse_odds(val) -> float:
    try:
        return float(str(val).replace("倍", "").replace("---", "0").strip())
    except (ValueError, TypeError):
        return 0.0


def _parse_age(age_str: str) -> int:
    """Parse age from Japanese format like '牡4' or 'セ5'."""
    m = re.search(r"(\d+)", str(age_str))
    return int(m.group(1)) if m else 0


def _parse_horse_weight(hw_str: str) -> tuple[float, float]:
    """Parse '480(+4)' or '480kg(-2)' to (weight, delta)."""
    weight = 0.0
    delta = 0.0
    m = re.search(r"(\d{3,4})", str(hw_str))
    if m:
        weight = float(m.group(1))
    m2 = re.search(r"\(([+-]?\d+)\)", str(hw_str))
    if m2:
        delta = float(m2.group(1))
    return weight, delta


def _parse_percentage(val: str) -> float:
    """Parse '18%' or '18.5%' to 0.185."""
    m = re.search(r"([\d.]+)%", str(val))
    return float(m.group(1)) / 100.0 if m else 0.0


def _parse_wins(val: str) -> int:
    """Parse '3勝' to 3."""
    m = re.search(r"(\d+)", str(val))
    return int(m.group(1)) if m else 0


def _validate_no_forbidden(features: dict):
    """Recursively check that no forbidden keys are present."""
    if isinstance(features, dict):
        for key in features:
            if key in _FORBIDDEN_KEYS:
                raise ValueError(
                    f"Forbidden key '{key}' found in structured_features. "
                    "This is post-race or Gemini-derived data that must not be persisted."
                )
            _validate_no_forbidden(features[key])
    elif isinstance(features, list):
        for item in features:
            _validate_no_forbidden(item)
