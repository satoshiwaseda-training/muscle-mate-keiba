"""Composite condition score — fuses free public data into a single [0, 1] number.

Inputs (each computed on what's actually available per horse; missing
dimensions are EXCLUDED from the weighted mean rather than filled with
neutral, so the composite reflects real information only):

  - weight_fitness      from horse_weight_delta (JRA body-weight change)
  - track_fit           from weight × track_condition × distance
  - paddock_aggregate   from the 4-dim paddock_features schema
  - odds_momentum       from intraday odds drift (live-only; backtest = absent)

Weights are declarative and can be tuned later. The composite is mapped
into score_runner's existing bio pathway (paddock_vascularity/hindquarter/
gait) so it affects ranking without requiring `train.py` changes.
"""

from __future__ import annotations

import math
from typing import Any


# ── Per-dimension scoring ─────────────────────────────────

def weight_fitness_score(delta_kg: float | int | None) -> tuple[float, bool]:
    """Bell curve centered at 0 kg change (σ = 5 kg).

    Returns (score_in_[0,1], has_signal).

      Δ =  0   → 1.00
      Δ = ±4  → 0.73
      Δ = ±8  → 0.28
      Δ = ±12 → 0.056
      Δ = ±16 → 0.007

    A |Δ| over 20 kg is probably data error; clamped to 0.02.
    `None` or zero delta is treated as NO SIGNAL (some horses show `--`
    on historical pages) to avoid rewarding missing data with a perfect 1.0.
    """
    if delta_kg is None:
        return (0.5, False)
    try:
        d = float(delta_kg)
    except (ValueError, TypeError):
        return (0.5, False)
    # Treat exactly 0 as "unknown" since historical pages use 0 as a NaN marker.
    if d == 0.0:
        return (0.5, False)
    sigma = 5.0
    score = math.exp(-(d * d) / (2 * sigma * sigma))
    return (max(0.02, min(1.0, score)), True)


def track_fit_score(h_sf: dict, race_sf: dict) -> tuple[float, bool]:
    """Very rough per-horse track fit proxy.

    Signal components:
      - On heavy/yielding tracks (重/不良), younger horses and heavier
        horses tend to handle wet turf slightly better.
      - On firm tracks (良), lighter carried weight helps marginally.
      - On 稍重, neutral.

    This is a weak directional prior, not a predictor. Returns
    (score_in_[0,1], has_signal).
    """
    track = (race_sf or {}).get("track_condition", "") or ""
    if not track:
        return (0.5, False)
    age = h_sf.get("age", 0) or 0
    horse_wt = h_sf.get("horse_weight_kg", 0) or 0
    carried = h_sf.get("carried_weight", 0) or 0
    if not (isinstance(age, (int, float)) and isinstance(horse_wt, (int, float))
            and isinstance(carried, (int, float))):
        return (0.5, False)

    score = 0.5
    if track in ("重", "不良"):
        # Heavier horses carry through mud better; older (5+) slightly favored
        if horse_wt >= 500: score += 0.08
        if horse_wt >= 520: score += 0.04
        if age >= 5: score += 0.04
        if carried >= 58: score -= 0.05   # heavy burden on soft = harder
    elif track == "稍重":
        score += 0.0  # neutral
    else:  # 良
        if carried <= 55: score += 0.03
        if age == 4: score += 0.02
    return (max(0.0, min(1.0, score)), True)


def paddock_aggregate(h_sf: dict) -> tuple[float, bool]:
    """Average of the 4-dim paddock_features schema when any dimension is
    non-neutral. Returns (score_in_[0,1], has_signal)."""
    keys = ("paddock_gait_score_01", "paddock_hindquarter_01",
            "paddock_vascularity_01", "paddock_mental_state_01")
    vals = []
    any_nonneutral = False
    for k in keys:
        v = h_sf.get(k)
        if isinstance(v, (int, float)):
            vals.append(float(v))
            if v != 0.5:
                any_nonneutral = True
    if not vals or not any_nonneutral:
        return (0.5, False)
    return (sum(vals) / len(vals), True)


def odds_momentum_score(h_sf: dict) -> tuple[float, bool]:
    """Intraday odds drift → [0, 1]. Needs two odds snapshots (morning +
    closing). For the historical backtest we have only the closing odds,
    so this always returns (0.5, False) and gets excluded from the
    composite. Wired here so the live pipeline can populate it.

    Expected fields when available:
      h_sf['odds_morning']  — decimal morning odds
      h_sf['odds_close']    — decimal closing odds (shutuba final)
    """
    m = h_sf.get("odds_morning")
    c = h_sf.get("odds_close") or h_sf.get("odds")
    if not isinstance(m, (int, float)) or not isinstance(c, (int, float)):
        return (0.5, False)
    if m <= 0 or c <= 0:
        return (0.5, False)
    # Negative drift (odds shortened) = smart money coming in → positive.
    # Normalize: 30% drop → 1.0, 0% → 0.5, 30% rise → 0.0
    drift = (m - c) / m          # positive when odds shortened
    score = 0.5 + drift * (0.5 / 0.30)
    return (max(0.0, min(1.0, score)), True)


# ── Composite ─────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "weight_fitness":  0.35,
    "track_fit":       0.15,
    "paddock_aggregate": 0.35,
    "odds_momentum":   0.15,
}


def composite_condition_score(
    h_sf: dict,
    race_sf: dict,
    weights: dict[str, float] | None = None,
) -> dict:
    """Weighted mean of the 4 dimensions, excluding dimensions with no signal.

    Returns a dict:
      {
        "composite": float in [0, 1],
        "has_signal": bool,
        "dimensions": {
            "weight_fitness":  (value, has_signal, weight_used),
            "track_fit":       (...),
            "paddock_aggregate": (...),
            "odds_momentum":   (...),
        },
        "n_dims_used": int,
      }
    """
    w = weights or DEFAULT_WEIGHTS

    dims = {
        "weight_fitness":    weight_fitness_score(h_sf.get("horse_weight_delta")),
        "track_fit":         track_fit_score(h_sf, race_sf),
        "paddock_aggregate": paddock_aggregate(h_sf),
        "odds_momentum":     odds_momentum_score(h_sf),
    }

    total_w = 0.0
    weighted_sum = 0.0
    details = {}
    used = 0
    for name, (val, has_sig) in dims.items():
        wk = w.get(name, 0.0)
        if has_sig and wk > 0:
            weighted_sum += wk * val
            total_w += wk
            used += 1
            details[name] = (val, True, wk)
        else:
            details[name] = (val, False, 0.0)

    if total_w == 0:
        return {
            "composite": 0.5,
            "has_signal": False,
            "dimensions": details,
            "n_dims_used": 0,
        }
    return {
        "composite": weighted_sum / total_w,
        "has_signal": True,
        "dimensions": details,
        "n_dims_used": used,
    }


# ── score_runner integration ──────────────────────────────
# score_runner reads paddock_vascularity/hindquarter/gait in [-1, 1] and
# averages them into bio_raw. We inject the composite into all three
# fields only when paddock coverage was previously missing, so real
# paddock scores (when present) are preserved.

def inject_composite_into_bio(
    h_sf: dict,
    composite: float,
    only_when_empty: bool = True,
) -> dict:
    """Write composite into score_runner's paddock keys.

    Returns a new dict; does not mutate input.
    - composite in [0, 1] is mapped to [-1, 1] via 2c - 1.
    - If `only_when_empty=True`, existing non-zero paddock values are kept.
    """
    out = dict(h_sf)
    centered = round(2.0 * composite - 1.0, 3)

    def needs_fill(key):
        v = out.get(key, 0)
        if only_when_empty:
            return not isinstance(v, (int, float)) or v == 0
        return True

    for key in ("paddock_gait", "paddock_hindquarter", "paddock_vascularity"):
        if needs_fill(key):
            out[key] = centered
    return out
