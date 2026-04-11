"""Dual-mode scoring: break odds dominance when facts are strong enough.

  if strong_fact_signal(horse):
      score = fact_weighted_score(horse)
  else:
      score = odds_weighted_score(horse)   # = existing score_runner output

A horse passes `strong_fact_signal` when ALL of:
  - consensus_fact_count ≥ 3   (facts corroborated by ≥2 sources)
  - composite_condition ≥ 0.7
  - no strong negative facts   (no Fact with polarity=-1 and confidence > 0.6)

`fact_weighted_score` REDUCES odds from ~80% dominance to 30%, and gives
the remaining 70% to fact-derived components weighted as requested:

  fact_score =
      0.30 · market_prob
    + 0.25 · composite_condition
    + 0.15 · hindquarter_strength
    + 0.12 · gait_score
    + 0.10 · vascularity
    + 0.08 · mental_state

Final output is rescaled to the same 0-100 range score_runner emits so
both modes can share a softmax and selection layer without rescaling.
"""

from __future__ import annotations

from typing import Any


STRONG_FACT_MIN_CONSENSUS = 3
STRONG_FACT_MIN_COMPOSITE = 0.7
STRONG_FACT_MAX_NEG_CONF = 0.6   # a negative fact with conf > this blocks override


# ── LOOSE TRIGGER (experimental betting rule) ──────────
# Rule ID — include in every persisted loose-bet entry so we can tell
# which version of the rule produced historical decisions.
LOOSE_RULE_VERSION = "cons>=1_comp>=0.60_odds<=15_no_strongneg_v1"

LOOSE_MIN_CONSENSUS = 1
LOOSE_MIN_COMPOSITE = 0.60
LOOSE_MAX_ODDS = 15.0


def trigger_loose_capped(horse: dict) -> tuple[bool, str]:
    """Loose trigger — the empirically best offline betting rule.

    Parallel and INDEPENDENT from `strong_fact_signal`. Strict trigger
    must continue to work exactly as before; this function only adds
    a second, experimental signal.

    Expects a dict with:
      - odds:              float | None
      - consensus_count:   int
      - composite_condition: float
      - strong_negative_present: bool (optional; defaults to False)

    Returns (flag, reason). `reason` is a human-readable string either
    explaining the rejection or, on True, summarising the key metrics.
    """
    if not isinstance(horse, dict):
        return (False, "invalid-horse-dict")
    odds = horse.get("odds")
    if odds is None:
        return (False, "missing-odds")
    if not isinstance(odds, (int, float)):
        return (False, "non-numeric-odds")
    if odds <= 0:
        return (False, f"non-positive-odds ({odds})")
    if odds > LOOSE_MAX_ODDS:
        return (False, f"odds {odds:.1f} > {LOOSE_MAX_ODDS} (extreme longshot)")

    consensus = int(horse.get("consensus_count", 0) or 0)
    if consensus < LOOSE_MIN_CONSENSUS:
        return (False, f"consensus {consensus} < {LOOSE_MIN_CONSENSUS}")

    composite = float(horse.get("composite_condition", 0.5) or 0.5)
    if composite < LOOSE_MIN_COMPOSITE:
        return (False, f"composite {composite:.2f} < {LOOSE_MIN_COMPOSITE:.2f}")

    if bool(horse.get("strong_negative_present", False)):
        return (False, "strong negative contradiction present")

    return (True,
            f"cons={consensus} comp={composite:.2f} odds={odds:.1f} clean")


# ── Mode predicate ─────────────────────────────────────

def strong_fact_signal(
    consensus_count: int,
    composite_condition: float,
    negative_facts: list,          # list of Fact objects with polarity=-1
) -> tuple[bool, str]:
    """Returns (pass, reason)."""
    if consensus_count < STRONG_FACT_MIN_CONSENSUS:
        return (False, f"consensus_count {consensus_count} < {STRONG_FACT_MIN_CONSENSUS}")
    if composite_condition < STRONG_FACT_MIN_COMPOSITE:
        return (False, f"composite {composite_condition:.2f} < {STRONG_FACT_MIN_COMPOSITE}")
    strong_negs = [f for f in negative_facts if float(f.confidence) > STRONG_FACT_MAX_NEG_CONF]
    if strong_negs:
        types = ",".join(f.type for f in strong_negs[:2])
        return (False, f"strong negative facts: {types}")
    return (True, f"consensus={consensus_count} composite={composite_condition:.2f}")


# ── Fact-weighted score ────────────────────────────────

MARKET_OVERROUND = 1.20


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _market_prob(odds: float) -> float:
    if odds <= 1.0:
        return 0.02
    return _clamp(1.0 / (odds * MARKET_OVERROUND), 0.02, 0.80)


def fact_weighted_score(h_sf: dict, composite_condition: float) -> float:
    """Return a 0-100 score with REDUCED odds dominance.

    Uses the 4-dim paddock schema (gait_score_01, hindquarter_01,
    vascularity_01, mental_state_01) if present; falls back to 0.5 neutral
    for missing dims. The composite_condition is passed in (from
    aggregate_horse_score) rather than recomputed here.
    """
    odds = float(h_sf.get("odds", 0) or 0)
    mp = _market_prob(odds)

    # 4-dim fact features, defaulting to 0.5 when missing
    g  = float(h_sf.get("paddock_gait_score_01", 0.5) or 0.5)
    hq = float(h_sf.get("paddock_hindquarter_01", 0.5) or 0.5)
    v  = float(h_sf.get("paddock_vascularity_01", 0.5) or 0.5)
    m  = float(h_sf.get("paddock_mental_state_01", 0.5) or 0.5)

    c = float(composite_condition or 0.5)

    prob = (
        0.30 * mp
        + 0.25 * c
        + 0.15 * hq
        + 0.12 * g
        + 0.10 * v
        + 0.08 * m
    )
    prob = _clamp(prob, 0.02, 0.95)
    return prob * 100.0


# ── Odds-weighted score ────────────────────────────────
# We keep using score_runner's existing output for this mode. The
# harness below runs score_runner once per horse and stores the result.


# ── Dual-mode decision ─────────────────────────────────

def dual_mode_score(
    h_sf: dict,
    odds_score: float,              # already-computed score_runner output
    consensus_count: int,
    composite_condition: float,
    negative_facts: list,
) -> dict:
    """Return the chosen score + mode + reason for one horse."""
    is_strong, reason = strong_fact_signal(
        consensus_count, composite_condition, negative_facts,
    )
    if is_strong:
        score = fact_weighted_score(h_sf, composite_condition)
        return {"score": score, "mode": "fact", "reason": reason,
                "odds_score": odds_score, "fact_score": score}
    return {"score": odds_score, "mode": "odds", "reason": reason,
            "odds_score": odds_score,
            "fact_score": fact_weighted_score(h_sf, composite_condition)}
