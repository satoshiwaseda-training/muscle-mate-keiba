"""Pedigree & camp feature extraction.

Computes structured features from bloodline, breeder, owner, and
trainer data. All features are pre-race safe (no leakage).

DESIGN PRINCIPLES:
  1. Every score maps to [0, 1] (neutral = 0.5)
  2. Unknown entities → 0.5 (never 0, which would penalize)
  3. All lookups use entity_tier static tables
  4. Composite scores are bounded and auditable
  5. Each function returns (score, has_signal) tuples

COMPOSITE SCORES (for scoring integration):
  - pedigree_composite: sire_tier + damsire_tier + distance_fit + surface_fit
  - camp_composite: breeder_tier + owner_tier + external_stable
  - interaction scores: sire × distance, sire × surface

VERSION: 1
"""

from __future__ import annotations

import entity_tier as et


# ═══════════════════════════════════════════════════════════
# A. Bloodline scores
# ═══════════════════════════════════════════════════════════

def sire_tier_score(sire_name: str) -> tuple[float, bool]:
    """Normalize sire tier (1-5) → [0, 1]. Unknown → (0.5, False)."""
    tier = et.get_sire_tier(sire_name)
    if sire_name and sire_name.strip() and tier != 2:
        return ((tier - 1) / 4.0, True)
    if sire_name and sire_name.strip():
        # Known name but not in table → average, weak signal
        return (0.25, True)
    return (0.5, False)


def damsire_tier_score(damsire_name: str) -> tuple[float, bool]:
    """Normalize damsire tier (1-5) → [0, 1]. Unknown → (0.5, False)."""
    tier = et.get_damsire_tier(damsire_name)
    if damsire_name and damsire_name.strip() and tier != 2:
        return ((tier - 1) / 4.0, True)
    if damsire_name and damsire_name.strip():
        return (0.25, True)
    return (0.5, False)


def sire_distance_fit(sire_name: str, race_distance: int) -> tuple[float, bool]:
    """How well does the sire's distance profile match the race?

    Returns (score ∈ [0, 1], has_signal).
    1.0 = perfect match (race at peak distance)
    0.0 = completely outside range
    """
    if not sire_name or not sire_name.strip() or race_distance <= 0:
        return (0.5, False)

    d_min, d_peak, d_max = et.get_sire_distance_profile(sire_name)
    # Check if this is the default profile (unknown sire)
    if (d_min, d_peak, d_max) == et.DEFAULT_DISTANCE_PROFILE:
        # Sire exists but not in profile table
        if sire_name.strip():
            return (0.5, False)
        return (0.5, False)

    if race_distance < d_min:
        # Too short — linearly decay
        gap = d_min - race_distance
        score = max(0.0, 1.0 - gap / 400.0)
        return (score * 0.5, True)  # cap at 0.5 when out of range

    if race_distance > d_max:
        # Too long — linearly decay
        gap = race_distance - d_max
        score = max(0.0, 1.0 - gap / 400.0)
        return (score * 0.5, True)

    # Within [min, max] — triangular peak at d_peak
    if race_distance <= d_peak:
        # Rising slope
        span = d_peak - d_min
        if span > 0:
            ratio = (race_distance - d_min) / span
        else:
            ratio = 1.0
        score = 0.5 + 0.5 * ratio
    else:
        # Falling slope
        span = d_max - d_peak
        if span > 0:
            ratio = (d_max - race_distance) / span
        else:
            ratio = 1.0
        score = 0.5 + 0.5 * ratio

    return (round(score, 4), True)


def sire_surface_fit(sire_name: str, surface: str) -> tuple[float, bool]:
    """How well does the sire suit this surface (芝/ダート)?

    Returns (score ∈ [0, 1], has_signal).
    """
    if not sire_name or not sire_name.strip() or not surface:
        return (0.5, False)

    bias = et.get_sire_surface_bias(sire_name)
    if bias == et.DEFAULT_SURFACE_BIAS:
        return (0.5, False)

    if "芝" in surface:
        return (bias["turf"], True)
    elif "ダート" in surface or "ダ" in surface:
        return (bias["dirt"], True)
    return (0.5, False)


def sire_heavy_track_fit(sire_name: str, track_condition: str) -> tuple[float, bool]:
    """Sire's fitness on heavy track. Only fires on 重/不良.

    Returns (score ∈ [0, 1], has_signal).
    """
    if not track_condition or track_condition not in ("重", "不良"):
        return (0.5, False)
    if not sire_name or not sire_name.strip():
        return (0.5, False)

    bias = et.get_sire_heavy_track_bias(sire_name)
    if bias == et.DEFAULT_HEAVY_TRACK_BIAS:
        return (0.5, False)

    return (bias, True)


# ═══════════════════════════════════════════════════════════
# B. Camp scores (breeder, owner, external stable)
# ═══════════════════════════════════════════════════════════

def breeder_tier_score(breeder_name: str) -> tuple[float, bool]:
    """Normalize breeder tier (1-5) → [0, 1]."""
    tier = et.get_breeder_tier(breeder_name)
    if breeder_name and breeder_name.strip() and tier != 2:
        return ((tier - 1) / 4.0, True)
    if breeder_name and breeder_name.strip():
        return (0.25, True)
    return (0.5, False)


def owner_tier_score(owner_name: str) -> tuple[float, bool]:
    """Normalize owner tier (1-5) → [0, 1]."""
    tier = et.get_owner_tier(owner_name)
    if owner_name and owner_name.strip() and tier != 2:
        return ((tier - 1) / 4.0, True)
    if owner_name and owner_name.strip():
        return (0.25, True)
    return (0.5, False)


def external_stable_score(ritto: str) -> tuple[float, bool]:
    """External stable (外厩) signal.

    Horses using external training facilities (especially Ritto)
    have a statistically higher win rate in JRA. This is a weak
    but consistent signal.

    Returns (score ∈ [0, 1], has_signal).
    """
    if not ritto or ritto.strip() == "":
        return (0.5, False)
    # Known external stable → small positive
    return (0.60, True)


# ═══════════════════════════════════════════════════════════
# C. Composite scores
# ═══════════════════════════════════════════════════════════

# Weights for pedigree composite
_PEDIGREE_WEIGHTS = {
    "sire_tier":       0.30,
    "damsire_tier":    0.20,
    "distance_fit":    0.30,
    "surface_fit":     0.20,
}

# Weights for camp composite
_CAMP_WEIGHTS = {
    "breeder_tier":    0.40,
    "owner_tier":      0.35,
    "external_stable": 0.25,
}


def pedigree_composite(
    sire_name: str,
    damsire_name: str,
    race_distance: int,
    surface: str,
    track_condition: str = "",
) -> dict:
    """Compute pedigree composite score.

    Returns:
        {
            "composite": float ∈ [0, 1],
            "has_signal": bool,
            "n_dims_used": int,
            "dimensions": {name: (value, has_signal, weight)},
            "heavy_track_fit": (float, bool),  # separate, only on heavy
        }
    """
    dims = {}
    dims["sire_tier"] = (*sire_tier_score(sire_name), _PEDIGREE_WEIGHTS["sire_tier"])
    dims["damsire_tier"] = (*damsire_tier_score(damsire_name), _PEDIGREE_WEIGHTS["damsire_tier"])
    dims["distance_fit"] = (*sire_distance_fit(sire_name, race_distance), _PEDIGREE_WEIGHTS["distance_fit"])
    dims["surface_fit"] = (*sire_surface_fit(sire_name, surface), _PEDIGREE_WEIGHTS["surface_fit"])

    # Weighted mean excluding dims without signal
    total_w = 0.0
    weighted_sum = 0.0
    n_used = 0
    for name, (val, has_sig, w) in dims.items():
        if has_sig:
            weighted_sum += val * w
            total_w += w
            n_used += 1

    if total_w > 0:
        composite = weighted_sum / total_w
        has_signal = True
    else:
        composite = 0.5
        has_signal = False

    heavy = sire_heavy_track_fit(sire_name, track_condition)

    return {
        "composite": round(composite, 4),
        "has_signal": has_signal,
        "n_dims_used": n_used,
        "dimensions": dims,
        "heavy_track_fit": heavy,
    }


def camp_composite(
    breeder_name: str,
    owner_name: str,
    ritto: str,
) -> dict:
    """Compute camp (陣営) composite score.

    Returns:
        {
            "composite": float ∈ [0, 1],
            "has_signal": bool,
            "n_dims_used": int,
            "dimensions": {name: (value, has_signal, weight)},
        }
    """
    dims = {}
    dims["breeder_tier"] = (*breeder_tier_score(breeder_name), _CAMP_WEIGHTS["breeder_tier"])
    dims["owner_tier"] = (*owner_tier_score(owner_name), _CAMP_WEIGHTS["owner_tier"])
    dims["external_stable"] = (*external_stable_score(ritto), _CAMP_WEIGHTS["external_stable"])

    total_w = 0.0
    weighted_sum = 0.0
    n_used = 0
    for name, (val, has_sig, w) in dims.items():
        if has_sig:
            weighted_sum += val * w
            total_w += w
            n_used += 1

    if total_w > 0:
        composite = weighted_sum / total_w
        has_signal = True
    else:
        composite = 0.5
        has_signal = False

    return {
        "composite": round(composite, 4),
        "has_signal": has_signal,
        "n_dims_used": n_used,
        "dimensions": dims,
    }


# ═══════════════════════════════════════════════════════════
# D. Full extraction for a single horse
# ═══════════════════════════════════════════════════════════

def extract_pedigree_features(
    sire_name: str,
    dam_name: str,
    damsire_name: str,
    breeder_name: str,
    owner_name: str,
    ritto: str,
    race_distance: int,
    surface: str,
    track_condition: str = "",
) -> dict:
    """Extract all pedigree + camp features for one horse.

    Returns a flat dict suitable for feature_store persistence.
    Every value is a float or string — no nested objects.
    """
    ped = pedigree_composite(sire_name, damsire_name, race_distance,
                             surface, track_condition)
    camp = camp_composite(breeder_name, owner_name, ritto)

    # Individual scores for audit trail
    s_tier_val, s_tier_sig = sire_tier_score(sire_name)
    d_tier_val, d_tier_sig = damsire_tier_score(damsire_name)
    dist_val, dist_sig = sire_distance_fit(sire_name, race_distance)
    surf_val, surf_sig = sire_surface_fit(sire_name, surface)
    heavy_val, heavy_sig = ped["heavy_track_fit"]
    breed_val, breed_sig = breeder_tier_score(breeder_name)
    own_val, own_sig = owner_tier_score(owner_name)
    ext_val, ext_sig = external_stable_score(ritto)

    missing_count = sum(1 for sig in [
        s_tier_sig, d_tier_sig, dist_sig, surf_sig,
        breed_sig, own_sig, ext_sig,
    ] if not sig)

    return {
        # Raw entity names (for audit)
        "sire_name":            sire_name or "",
        "dam_name":             dam_name or "",
        "damsire_name":         damsire_name or "",
        "breeder_name":         breeder_name or "",
        "owner_name":           owner_name or "",
        # Individual scores
        "sire_tier_score":      round(s_tier_val, 4),
        "damsire_tier_score":   round(d_tier_val, 4),
        "sire_distance_fit":    round(dist_val, 4),
        "sire_surface_fit":     round(surf_val, 4),
        "sire_heavy_track_fit": round(heavy_val, 4),
        "breeder_tier_score":   round(breed_val, 4),
        "owner_tier_score":     round(own_val, 4),
        "external_stable_score": round(ext_val, 4),
        # Composite scores (for scoring integration)
        "pedigree_composite":   ped["composite"],
        "camp_composite":       camp["composite"],
        # Signal quality
        "pedigree_has_signal":  ped["has_signal"],
        "camp_has_signal":      camp["has_signal"],
        "pedigree_n_dims":      ped["n_dims_used"],
        "camp_n_dims":          camp["n_dims_used"],
        "missing_feature_count": missing_count,
    }
