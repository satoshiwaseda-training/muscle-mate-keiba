"""Bridges the fact layer + live scraper into the original structured
score_runner in train.py.

WHY THIS EXISTS
---------------
In live mode, `feature_store.extract_structured_features` reads raw
entries from the netkeiba shutuba page. Those entries carry odds,
horse_weight, carried weight, age, jockey_id — but NOT jockey_win_rate,
training_acceleration, training_cardio_index, or paddock_*. Those
fields are populated by `scraper.enrich_entries`, which live mode
deliberately skips for speed.

Result: when the live pipeline calls `score_runner`, 5 of its 9
structured signal channels are zero, and the output collapses to
essentially `(1/odds)/1.20 * 100`. The original coefficient-structure
model is present but starved of input.

WHAT THIS BRIDGE DOES
---------------------
1. `enrich_sf_horses_for_live(sf_horses, entries, race_id, facts_by_horse)`
   Mutates sf_horses in place, filling the missing score_runner fields
   from cached scraper stats, the oikiri page critic text, and the
   per-horse fact list.

2. `structured_edge_from_score(score, odds)`
   Recovers score_runner's non-odds adjustment from its [2, 95] output
   by subtracting the known base term. That adjustment becomes the
   fact edge consumed by `probability_engine.assign_calibrated_probs`.

No changes to `train.py` or `feature_store.py`. This module is a pure
adapter so the original core model logic remains untouched.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import scraper
import feature_store as fs


# ── Signal bridging ────────────────────────────────────

def _safe_num(v, default: float = 0.0) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    return default


def _harvest_jockey_win_rate(
    sf_horses: dict,
    entries: list[dict],
) -> int:
    """Fill sf_horses[name]['jockey_win_rate'] from cached jockey stats.

    Reads from the scraper's per-jockey disk cache
    (data/scraper_cache/jockey/*). If a jockey is not yet cached, a
    single HTTP call via `scraper._cached_jockey_stats` populates it.
    Returns the count of horses enriched.
    """
    enriched = 0
    for e in entries:
        name = (e.get("name") or "").strip()
        if not name or name not in sf_horses:
            continue
        h = sf_horses[name]
        if _safe_num(h.get("jockey_win_rate")) > 0:
            continue
        jid = e.get("jockey_id") or ""
        if not jid:
            continue
        try:
            jstats = scraper._cached_jockey_stats(jid) or {}
        except Exception:
            continue
        wr_str = jstats.get("win_rate", "") or ""
        wr = fs._parse_percentage(wr_str) if wr_str else 0.0
        if wr > 0:
            h["jockey_win_rate"] = wr
            enriched += 1
    return enriched


def _harvest_training_critics(
    sf_horses: dict,
    race_id: str,
) -> int:
    """Fetch the oikiri page, parse Training_Critic text per horse,
    and map the ordinal critic score into training_acceleration and
    training_cardio_index.

    Uses the scraper's race-level cached fetcher when available.
    Returns the count of horses enriched.
    """
    enriched = 0
    try:
        training_rows = scraper._cached_training_times(race_id) or []
    except Exception:
        training_rows = []
    if not training_rows:
        return 0
    for row in training_rows:
        name = (row.get("name") or "").strip()
        eval_text = row.get("evaluation") or ""
        if not name or name not in sf_horses or not eval_text:
            continue
        try:
            critic = scraper.parse_training_critic(eval_text)
        except Exception:
            critic = 0.0
        if critic <= 0:
            continue
        h = sf_horses[name]
        # Map critic [0,1] → training_acceleration [-0.15, 0.15]
        if not _safe_num(h.get("training_acceleration")):
            h["training_acceleration"] = round((critic - 0.5) * 0.30, 4)
            enriched += 1
        # cardio_index mirrors critic in [0,1]
        if not _safe_num(h.get("training_cardio_index")):
            h["training_cardio_index"] = round(critic, 4)
    return enriched


def _harvest_paddock_from_facts(
    sf_horses: dict,
    facts_by_horse: dict[str, list],
) -> int:
    """Fill sf_horses[name].paddock_{vascularity,hindquarter,gait} from
    the per-horse fact list, using category buckets from fact_schema.

    Mapping:
      paddock_vascularity ← facts in 'condition' category (net polarity)
      paddock_hindquarter ← facts in 'physique' category
      paddock_gait        ← facts in 'movement' category
    Values are in [-1, 1] where 0 is neutral.
    """
    enriched = 0
    for name, h in sf_horses.items():
        facts = facts_by_horse.get(name, []) if facts_by_horse else []
        if not facts:
            continue

        by_cat: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for f in facts:
            cat = getattr(f, "category", "") or ""
            pol = getattr(f, "polarity", 0) or 0
            conf = float(getattr(f, "confidence", 0) or 0)
            if cat not in ("condition", "physique", "movement"):
                continue
            if pol > 0:
                by_cat[cat][0] += conf
            elif pol < 0:
                by_cat[cat][1] += conf

        def _net(cat_name: str) -> float:
            pn = by_cat.get(cat_name, [0.0, 0.0])
            total = pn[0] + pn[1]
            if total == 0:
                return 0.0
            return round((pn[0] - pn[1]) / total, 3)

        any_changed = False
        v = _net("condition")
        hq = _net("physique")
        g = _net("movement")
        if v != 0 and not _safe_num(h.get("paddock_vascularity")):
            h["paddock_vascularity"] = v
            any_changed = True
        if hq != 0 and not _safe_num(h.get("paddock_hindquarter")):
            h["paddock_hindquarter"] = hq
            any_changed = True
        if g != 0 and not _safe_num(h.get("paddock_gait")):
            h["paddock_gait"] = g
            any_changed = True
        if any_changed:
            enriched += 1
    return enriched


def enrich_sf_horses_for_live(
    sf_horses: dict,
    entries: list[dict],
    race_id: str,
    facts_by_horse: dict[str, list],
) -> dict:
    """Populate sf_horses with the fields score_runner reads.

    Mutates sf_horses in place. Safe to call repeatedly (only fills
    empty fields, never overwrites real data). Returns a diagnostic
    dict describing what was enriched.
    """
    diag = {
        "jockey_win_rate": _harvest_jockey_win_rate(sf_horses, entries),
        "training_critic": _harvest_training_critics(sf_horses, race_id),
        "paddock_from_facts": _harvest_paddock_from_facts(sf_horses, facts_by_horse),
    }

    # Audit: count how many horses have each signal non-zero post-enrich
    def _count(key):
        return sum(1 for h in sf_horses.values() if _safe_num(h.get(key)) != 0)

    diag["post_non_zero"] = {
        "jockey_win_rate":       _count("jockey_win_rate"),
        "training_acceleration": _count("training_acceleration"),
        "training_cardio_index": _count("training_cardio_index"),
        "paddock_vascularity":   _count("paddock_vascularity"),
        "paddock_hindquarter":   _count("paddock_hindquarter"),
        "paddock_gait":          _count("paddock_gait"),
        "horse_weight_delta":    _count("horse_weight_delta"),
        "total_horses":          len(sf_horses),
    }
    return diag


# ── Score → edge decomposition ────────────────────────

def structured_edge_from_score(score: float, odds: float) -> float:
    """Recover score_runner's structured adjustment from its raw output.

    score_runner composition (train.py):
        base         = clamp((1/odds) / 1.20, 0.02, 0.80)   when odds > 1.0
        final_prob   = clamp(base + adjustment, 0.02, 0.95)
        top_confidence = final_prob * 100     # in [2, 95]

    So we recover:
        adjustment ≈ (score / 100) - base

    For odds <= 1.0 (scratched or missing), returns 0 — we cannot
    decompose a uniform-base fallback into a meaningful edge.
    """
    if not isinstance(score, (int, float)) or score <= 0:
        return 0.0
    if not isinstance(odds, (int, float)) or odds <= 1.0:
        return 0.0
    implied_base = max(0.02, min(0.80, (1.0 / odds) / 1.20))
    return round((score / 100.0) - implied_base, 4)
