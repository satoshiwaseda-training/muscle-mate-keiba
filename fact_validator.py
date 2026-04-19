"""Strict validation & transformation layer between fact_extractor and merge.

Responsibilities (in pipeline order):

  1. validate_fact(f)         — reject predictions/opinions/betting language
                                (second-layer defense on top of
                                 fact_schema.sentence_is_opinion)
  2. classify_fact(f)         — tag each fact with a state bucket:
                                {observation, condition, fatigue, stress, pain}
  3. canonical_key(f)         — produce a normalized merge key so that
                                phrase variants collapse cleanly
  4. detect_contradictions()  — for each horse, when both polarities
                                appear in the same state bucket, the
                                weaker side gets its confidence
                                multiplicatively reduced
  5. compute_state_scores()   — condense the fact list into four [0, 1]
                                scores: condition / fatigue / stress / pain_risk

Public entry point:

  validate_and_transform(facts) -> list[Fact]
      one-shot wrapper that applies steps 1-4 and returns a cleaned list.
      Step 5 is called separately per horse from the aggregator layer.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from fact_schema import Fact


# ── Extra blacklist — second-layer defense ─────────────

_EXTRA_OPINION_PHRASES: tuple[str, ...] = (
    # additional prediction / betting / popularity lexicon that the
    # sentence-level blacklist in fact_schema may miss on the raw_text
    "本命馬", "注目馬", "勝負馬", "買い", "買う", "押さえ",
    "軸馬", "相手馬", "穴馬", "人気上位",
    "勝ちそう", "勝てそう", "負けない", "鉄板",
    "狙い撃ち", "外せない", "軸", "ヒモ",
    # odds-derived betting frames
    "オッズ妙味", "美味しい", "甘い",
    # emotional hype
    "期待大", "期待値高", "イチオシ", "推し",
)

# Minimum raw per-fact confidence to accept (noise floor).
# Negatives get a LOWER floor because we deliberately want to amplify
# detection of fatigue / stress / pain signals — even weak hints are
# worth surfacing.
MIN_ACCEPT_CONFIDENCE_POSITIVE = 0.30
MIN_ACCEPT_CONFIDENCE_NEGATIVE = 0.20


def validate_fact(f: Fact) -> tuple[bool, str]:
    """Validate a single fact. Returns (ok, reason_if_rejected)."""
    if not f or not f.type:
        return (False, "empty-fact")
    # Polarity == 0 is uninformative (we only propagate positive/negative)
    if f.polarity == 0:
        return (False, "neutral-polarity")
    # Confidence noise floor — asymmetric for positives vs negatives
    try:
        conf = float(f.confidence)
    except (ValueError, TypeError):
        return (False, "non-numeric-confidence")
    floor = (MIN_ACCEPT_CONFIDENCE_NEGATIVE if f.polarity < 0
             else MIN_ACCEPT_CONFIDENCE_POSITIVE)
    if conf < floor:
        return (False, f"below-noise-floor ({conf:.2f} < {floor})")
    # Raw text betting-language sweep
    raw = f.raw_text or ""
    for p in _EXTRA_OPINION_PHRASES:
        if p in raw:
            return (False, f"betting-language: {p}")
    return (True, "ok")


# ── State classification ───────────────────────────────
# Every canonical fact type maps to ONE state bucket.
# State buckets:
#   - observation: neutral factual data (numeric, non-polarized)
#   - condition:   overall readiness / appearance / physique / ride-quality
#   - fatigue:     signs the horse is tired or heavy
#   - stress:      mental / behavioral tension
#   - pain:        movement restriction suggesting discomfort

_STATE_BY_FACT_TYPE: dict[str, str] = {
    # Condition (general readiness)
    "coat_good":               "condition",
    "coat_bad":                "condition",
    "body_sharp":              "condition",
    "hindquarter_strong":      "condition",
    "hindquarter_weak":        "condition",
    "gait_good":               "condition",
    "mental_calm":             "condition",
    "good_weight_stable":      "condition",
    "stable_positive_comment": "condition",
    "stable_concern_comment":  "condition",

    # Fatigue (heaviness / reduced readiness — observational not structural)
    "body_heavy":              "fatigue",
    "weight_up_large":         "fatigue",
    "weight_down_large":       "fatigue",

    # Stress (mental / behavioral)
    "mental_tense":            "stress",
    "sweating_concern":        "stress",

    # Pain (movement restriction — hints at discomfort)
    "gait_bad":                "pain",

    # OBSERVATION — excluded from state scoring even though polarity≠0.
    # high_carried_weight fires on any horse carrying ≥58kg (common
    # for graded-race top weights) so routing it to "fatigue" would
    # flag ~30% of horses as fatigued. It's structurally a race-setup
    # fact, not a physiological state.
    "high_carried_weight":     "observation",

    # Pure observational (polarity 0 → filtered by validator anyway)
    "heavy_track_fit":         "observation",
    "distance_fit_mile":       "observation",
    "track_firm":              "observation",
    "track_soft":              "observation",
    "track_heavy":             "observation",
    "track_very_heavy":        "observation",
    "track_firm_cushion":      "observation",
    "track_soft_cushion":      "observation",
    "scratched":               "observation",
}

# Direction: +1 means a positive fact of this type IMPROVES its state;
# -1 means a positive fact of this type WORSENS its state.
# For example, body_heavy.polarity = -1 in fact_schema (it's a bad sign),
# so we want fatigue_score to INCREASE when body_heavy is observed.
# Translation: fatigue_contribution = -polarity × confidence.

_STATE_DIRECTION: dict[str, int] = {
    "condition": +1,   # positive-polarity fact → +condition
    "fatigue":   -1,   # negative-polarity fact → +fatigue
    "stress":    -1,   # negative-polarity fact → +stress
    "pain":      -1,   # negative-polarity fact → +pain_risk
    "observation": 0,
}


def classify_fact(f: Fact) -> str:
    """Return the state bucket for a fact. Defaults to 'observation'."""
    if not f or not f.type:
        return "observation"
    return _STATE_BY_FACT_TYPE.get(f.type, "observation")


# ── Canonical key — merge hint ─────────────────────────

def canonical_key(f: Fact) -> tuple[str, int]:
    """Return the normalization key used to group fuzzy-variant facts.

    Currently (type, polarity). Kept simple — the fuzzy cluster logic
    already lives in `merge_fact_layers`, so this layer focuses on
    dropping predictions and contradictions rather than re-merging.
    """
    return (f.type, f.polarity)


# ── Contradiction detection ───────────────────────────

# Penalty multiplier applied to the weaker side of a contradiction.
# 0.50 = halve the confidence. 0.0 = drop entirely. We use 0.50 so
# the contradicted fact is still visible for audit but cannot dominate.
CONTRADICTION_PENALTY = 0.50


def detect_contradictions(facts_for_horse: list[Fact]) -> list[Fact]:
    """Down-weight the weaker side of any contradiction for one horse.

    Two detection axes:
      1. Same ORIGINAL category (fact.category from extraction). Catches
         mental_calm vs mental_tense even though they map to different
         state buckets.
      2. Same STATE bucket. Catches e.g. body_sharp vs body_heavy inside
         the condition/fatigue split.

    When BOTH polarities appear along either axis, the side with lower
    summed confidence gets multiplied by CONTRADICTION_PENALTY. Returns
    a NEW list (does not mutate input).
    """
    if not facts_for_horse:
        return []

    # Build two grouping maps. Both use (group_key, polarity) -> facts.
    by_category: dict[str, dict[int, list[Fact]]] = defaultdict(lambda: defaultdict(list))
    by_bucket: dict[str, dict[int, list[Fact]]] = defaultdict(lambda: defaultdict(list))
    for f in facts_for_horse:
        by_category[f.category or "?"][f.polarity].append(f)
        by_bucket[classify_fact(f)][f.polarity].append(f)

    weakened_ids: set[int] = set()

    def _flag_weaker(grouping: dict):
        for _, by_pol in grouping.items():
            pos = by_pol.get(+1, [])
            neg = by_pol.get(-1, [])
            if not pos or not neg:
                continue
            pos_w = sum(float(f.confidence) for f in pos)
            neg_w = sum(float(f.confidence) for f in neg)
            weaker = neg if pos_w >= neg_w else pos
            for f in weaker:
                weakened_ids.add(id(f))

    _flag_weaker(by_category)
    _flag_weaker(by_bucket)

    out: list[Fact] = []
    for f in facts_for_horse:
        if id(f) in weakened_ids:
            new_conf = round(float(f.confidence) * CONTRADICTION_PENALTY, 3)
            out.append(Fact(
                type=f.type, horse=f.horse, polarity=f.polarity,
                confidence=new_conf, source=f.source, raw_text=f.raw_text,
                category=f.category,
                meta={**(f.meta or {}), "contradicted": True,
                      "original_confidence": float(f.confidence)},
            ))
        else:
            out.append(f)
    return out


# ── State scores ──────────────────────────────────────

_STATE_FLOOR_N = 3.0   # shrinkage prior for each state bucket


def compute_state_scores(facts_for_horse: list[Fact]) -> dict:
    """Condense a fact list into four [0, 1] state scores.

    Each state score uses the same shrinkage formula as the composite
    so that a single fact cannot produce a saturated score.

      condition_score ∈ [0, 1]   higher = better readiness
      fatigue_score   ∈ [0, 1]   higher = more fatigued
      stress_score    ∈ [0, 1]   higher = more stressed
      pain_risk       ∈ [0, 1]   higher = more likely in discomfort

    Plus:
      n_facts_by_state  — fact count per bucket (for audit)
      observation_count — neutral observations captured
    """
    buckets = {"condition": [], "fatigue": [], "stress": [], "pain": []}
    obs_count = 0
    for f in facts_for_horse:
        b = classify_fact(f)
        if b == "observation":
            obs_count += 1
            continue
        buckets[b].append(f)

    def _score(facts: list[Fact], direction: int) -> float:
        """Return a 0.5-centered score in [0, 1].

        direction=+1 (condition): 0.5 + 0.5 × net polarity × shrinkage
        direction=-1 (fatigue/stress/pain): AMPLIFIED negative sensitivity.
                    One strong negative fact of confidence 0.80 alone
                    produces a state score of ≈ 0.72. This is intentional:
                    negative signals are rare and valuable — we want the
                    detector to light up on the first clear hit, not
                    wait for accumulation.
        """
        if not facts:
            return 0.0 if direction == -1 else 0.5
        pos_w = sum(float(f.confidence) for f in facts if f.polarity > 0)
        neg_w = sum(float(f.confidence) for f in facts if f.polarity < 0)
        total = pos_w + neg_w
        if total == 0:
            return 0.0 if direction == -1 else 0.5
        n = len(facts)
        shrink = n / (n + _STATE_FLOOR_N)
        if direction == +1:
            net = (pos_w - neg_w) / total
            return round(0.5 + 0.5 * net * shrink, 3)
        # direction == -1 — AMPLIFIED negative sensitivity
        if neg_w == 0:
            return 0.0
        n_neg = sum(1 for f in facts if f.polarity < 0)
        # Use raw negative weight (capped at 1.0), with LIGHT shrinkage
        # so a single strong negative still produces ~0.7 score.
        # shrink_neg = n_neg / (n_neg + 1.5)
        #   n_neg=1 → 0.40,  n_neg=2 → 0.57,  n_neg=3 → 0.67
        shrink_neg = n_neg / (n_neg + 1.5)
        raw = min(1.0, neg_w)
        # Amplification factor 1.8 pushes a 0.80-conf solo negative
        # from 0.40*0.80 = 0.32 up to 0.72
        amplified = min(1.0, raw * shrink_neg * 1.8 + raw * 0.2)
        return round(amplified, 3)

    return {
        "condition_score": _score(buckets["condition"], +1),
        "fatigue_score":   _score(buckets["fatigue"],   -1),
        "stress_score":    _score(buckets["stress"],    -1),
        "pain_risk":       _score(buckets["pain"],      -1),
        "n_facts_by_state": {k: len(v) for k, v in buckets.items()},
        "observation_count": obs_count,
    }


# ── One-shot entrypoint ───────────────────────────────

def validate_and_transform(
    facts: Iterable[Fact],
    drop_report: list | None = None,
) -> list[Fact]:
    """Validate + contradiction-detect a fact list.

    Steps:
      1. Drop facts that fail `validate_fact`.
      2. Group by horse.
      3. Apply `detect_contradictions` per horse.
      4. Flatten back to a list preserving order as much as possible.

    Pass an empty list to `drop_report` to collect per-fact drop reasons
    for auditing; callers that don't care can leave it None.
    """
    kept: list[Fact] = []
    for f in facts or []:
        ok, reason = validate_fact(f)
        if not ok:
            if drop_report is not None:
                drop_report.append({
                    "type": getattr(f, "type", ""),
                    "horse": getattr(f, "horse", ""),
                    "raw_text": getattr(f, "raw_text", ""),
                    "confidence": getattr(f, "confidence", 0),
                    "reason": reason,
                })
            continue
        kept.append(f)

    # Contradiction-detect per horse
    by_horse: dict[str | None, list[Fact]] = defaultdict(list)
    for f in kept:
        by_horse[f.horse].append(f)

    out: list[Fact] = []
    for horse, group in by_horse.items():
        out.extend(detect_contradictions(group))
    return out
