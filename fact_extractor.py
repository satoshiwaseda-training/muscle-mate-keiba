"""Fact extraction and multi-source merging.

`extract_canonical_facts(text, source, horse=None)` maps a free-text blob
to a list of `Fact` objects, rejecting any sentence that looks like a
prediction / 印 / buy-list.

`merge_fact_layers(*layers)` combines facts from multiple sources, applies
the consensus bonus (+0.2 for 2 sources, +0.4 for 3+), and returns a
deduplicated fact list plus a per-horse aggregation suitable for the
scoring layer.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

from fact_schema import (
    CANONICAL_FACT_TYPES, SOURCE_BASE_CONFIDENCE,
    sentence_is_opinion, Fact, phrase_to_cluster,
)


# ── Sentence split ─────────────────────────────────────

_SENT_SPLIT = re.compile(r"[。．\n\r]+")


def _split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


# Phrase lookup: longest-first per type
_PHRASE_INDEX: dict[str, list[tuple[str, float]]] = {
    t: sorted(cfg.get("phrases", []), key=lambda kv: -len(kv[0]))
    for t, cfg in CANONICAL_FACT_TYPES.items()
}


# ── Negation handling ────────────────────────────────
# A matched phrase followed within _NEGATION_WINDOW characters by any of
# _NEGATION_MARKERS is dropped, because negated phrases are ambiguous.
# Example: 気負い + なく → "気負いなく" = calm, NOT tense → skip the
# mental_tense emission.
_NEGATION_WINDOW = 8
_NEGATION_MARKERS = (
    "ない", "なく", "なさそう", "ず", "ぬ", "ではな", "せず",
    "わけでは", "ほど", "まで", "というほど",
)


def _is_negated(text: str, match_end: int) -> bool:
    tail = text[match_end: match_end + _NEGATION_WINDOW]
    return any(m in tail for m in _NEGATION_MARKERS)


# ── Core extractor ─────────────────────────────────────

def extract_canonical_facts(
    text: str,
    source: str,
    horse: str | None = None,
) -> list[Fact]:
    """Extract canonical facts from a text blob.

    Opinion/print sentences are dropped entirely (no facts emitted from
    them). Phrase confidence is multiplied by the source's base
    confidence to get raw per-fact confidence.
    """
    if not text or not source:
        return []
    base = SOURCE_BASE_CONFIDENCE.get(source, 0.5)
    facts: list[Fact] = []

    for sentence in _split_sentences(text):
        if sentence_is_opinion(sentence):
            continue
        for fact_type, phrases in _PHRASE_INDEX.items():
            cfg = CANONICAL_FACT_TYPES[fact_type]
            polarity = cfg.get("polarity", 0)
            category = cfg.get("category", "")
            for phrase, strength in phrases:
                idx = sentence.find(phrase)
                if idx < 0:
                    continue
                # Negation check — skip if followed by a negation marker
                # within the local window. Prevents "気負いなく" (calm)
                # from emitting a mental_tense fact.
                if _is_negated(sentence, idx + len(phrase)):
                    break
                facts.append(Fact(
                    type=fact_type,
                    horse=horse,
                    polarity=polarity,
                    confidence=round(base * strength, 3),
                    source=source,
                    raw_text=phrase,
                    category=category,
                ))
                break  # one hit per type per sentence (avoid triple-counts)
    return facts


# ── Numeric-fact helpers (JRA-derived) ─────────────────
# These do NOT depend on text; they turn numeric JRA data into canonical facts.

def fact_from_weight_delta(horse: str, delta_kg: float | int | None) -> list[Fact]:
    if delta_kg is None:
        return []
    try:
        d = float(delta_kg)
    except (ValueError, TypeError):
        return []
    if d == 0.0:
        return []  # 0 is the historical "unknown" marker
    abs_d = abs(d)
    if abs_d >= 10:
        t = "weight_up_large" if d > 0 else "weight_down_large"
        return [Fact(type=t, horse=horse, polarity=-1,
                     confidence=SOURCE_BASE_CONFIDENCE["jra"] * 0.85,
                     source="jra", raw_text=f"{d:+.0f}kg",
                     category="weight", meta={"delta_kg": d})]
    if abs_d <= 4:
        return [Fact(type="good_weight_stable", horse=horse, polarity=+1,
                     confidence=SOURCE_BASE_CONFIDENCE["jra"] * 0.60,
                     source="jra", raw_text=f"{d:+.0f}kg",
                     category="weight", meta={"delta_kg": d})]
    return []


def fact_from_track_condition(condition: str) -> list[Fact]:
    """Race-level track fact. horse=None, applies to all."""
    if not condition:
        return []
    mapping = {"良": ("track_firm", 0), "稍重": ("track_soft", 0),
               "重": ("track_heavy", -1), "不良": ("track_very_heavy", -1)}
    if condition not in mapping:
        return []
    t, polarity = mapping[condition]
    return [Fact(type=t, horse=None, polarity=polarity,
                 confidence=SOURCE_BASE_CONFIDENCE["jra"],
                 source="jra", raw_text=condition, category="track")]


def fact_from_scratch(horse: str) -> Fact:
    return Fact(type="scratched", horse=horse, polarity=-1,
                confidence=SOURCE_BASE_CONFIDENCE["jra"],
                source="jra", raw_text="取消/除外", category="status")


# ── Multi-source merge with consensus bonus ────────────

def merge_fact_layers(*layers: Iterable[Fact]) -> list[Fact]:
    """Merge fact layers from multiple sources.

    Two-level consensus:

      1. Per-type: identical (type, horse) from ≥2 distinct sources earn
         a per-type bonus (+0.2 / +0.4 for 2 / 3+ sources).

      2. Per-category: different fact TYPES within the same CATEGORY
         (e.g. both `coat_good` and `body_sharp` in "condition") count as
         category-level corroboration when they come from different
         sources. Each fact in a consensed category receives an extra
         +0.10 bonus on top of its per-type bonus.

    Both bonuses stack and clamp to 1.0. The returned Fact's meta carries:
      - n_sources:              distinct sources for this (type, horse)
      - consensus_bonus:        per-type bonus applied
      - category_n_sources:     distinct sources in this (category, horse)
      - category_consensus_bonus: per-category bonus applied
      - in_consensed_category:  bool
    """
    by_key: dict[tuple[str, str | None], list[Fact]] = defaultdict(list)
    for layer in layers:
        for f in layer or []:
            by_key[(f.type, f.horse)].append(f)

    # First pass: compute category-level source sets per
    # (category, horse, polarity) so that "mental_calm from 2 sources"
    # does NOT also bonus a contradictory "mental_tense from 1 source".
    cat_pol_sources: dict[tuple[str, str | None, int], set[str]] = defaultdict(set)
    # Also: fuzzy cluster source sets. A cluster key groups near-synonym
    # phrases across canonical types (e.g. 毛艶 / 張り艶 / 光沢 all go into
    # `cluster_coat_shine`), so two sources using different-but-related
    # wording can corroborate at the cluster level.
    cluster_pol_sources: dict[tuple[str, str | None, int], set[str]] = defaultdict(set)
    for (t, h), group in by_key.items():
        cat = group[0].category if group else ""
        for f in group:
            if not f.source:
                continue
            cat_pol_sources[(cat, h, f.polarity)].add(f.source)
            # Cluster key is derived from the raw phrase that matched
            ck = phrase_to_cluster(f.raw_text)
            if ck:
                cluster_pol_sources[(ck, h, f.polarity)].add(f.source)

    merged: list[Fact] = []
    for (t, h), group in by_key.items():
        distinct_sources = {f.source for f in group}
        n_src = len(distinct_sources)
        per_type_bonus = 0.0
        if n_src >= 3:
            per_type_bonus = 0.4
        elif n_src == 2:
            per_type_bonus = 0.2

        best = max(group, key=lambda f: f.confidence)
        cat = best.category
        pol = best.polarity
        same_pol_cat_n = len(cat_pol_sources.get((cat, h, pol), set()))
        cat_bonus = 0.10 if same_pol_cat_n >= 2 else 0.0

        # Fuzzy cluster consensus — covers semantic near-synonyms that
        # land in different canonical types.
        ck = phrase_to_cluster(best.raw_text)
        cluster_n = len(cluster_pol_sources.get((ck, h, pol), set())) if ck else 0
        cluster_bonus = 0.08 if cluster_n >= 2 else 0.0

        merged_conf = min(
            1.0,
            best.confidence + per_type_bonus + cat_bonus + cluster_bonus,
        )

        # A fact is considered "consensed" if it has any form of
        # multi-source agreement — same type, same category+polarity, or
        # same fuzzy cluster+polarity.
        in_consensus = (
            n_src >= 2
            or same_pol_cat_n >= 2
            or cluster_n >= 2
        )

        merged.append(Fact(
            type=t, horse=h,
            polarity=best.polarity,
            confidence=round(merged_conf, 3),
            source="+".join(sorted(distinct_sources)),
            raw_text=best.raw_text,
            category=cat,
            meta={
                **best.meta,
                "n_sources": n_src,
                "consensus_bonus": per_type_bonus,
                "category_same_pol_n_sources": same_pol_cat_n,
                "category_consensus_bonus": cat_bonus,
                "cluster_key": ck,
                "cluster_n_sources": cluster_n,
                "cluster_consensus_bonus": cluster_bonus,
                "in_consensed_category": in_consensus,
            },
        ))
    return merged


# ── Per-horse aggregation for scoring ──────────────────

SHRINKAGE_PRIOR_N = 3.0   # "pseudo-count" pulling sparse horses toward neutral


def aggregate_horse_score(facts_for_horse: list[Fact]) -> dict:
    """Turn a list of facts for one horse into a composite score.

    The composite applies a **sample-size shrinkage** to the raw net
    polarity so that a single positive fact does not produce the same
    composite as ten positive facts:

        net_raw  = (pos_w - neg_w) / (pos_w + neg_w)   ∈ [-1, +1]
        shrink   = n / (n + prior_n)                   ∈ [0, 1), prior_n = 3
        composite = 0.5 + 0.5 * net_raw * shrink

    With prior_n = 3:
      n = 1  → shrink 0.25 → max composite 0.625
      n = 3  → shrink 0.50 → max composite 0.750
      n = 7  → shrink 0.70 → max composite 0.850
      n = 15 → shrink 0.83 → max composite 0.917
    """
    if not facts_for_horse:
        return {"composite_condition": 0.5,
                "composite_condition_all": 0.5,
                "n_facts": 0, "n_consensed": 0,
                "n_positive": 0, "n_negative": 0,
                "consensed_fact_count": 0, "consensed_categories": [],
                "by_category": {}}
    pos_w = 0.0              # all facts (for composite_all)
    neg_w = 0.0
    pos_w_c = 0.0            # CONSENSED facts only (for composite_consensed)
    neg_w_c = 0.0
    pos_n = 0
    neg_n = 0
    by_cat: dict[str, dict] = defaultdict(lambda: {"pos_weight": 0.0, "neg_weight": 0.0})
    consensed_fact_count = 0
    consensed_categories: set[str] = set()
    n_consensed = 0

    for f in facts_for_horse:
        w = float(f.confidence)
        in_consensus_category = bool(f.meta.get("in_consensed_category"))
        # A fact counts toward `composite_condition` (the strict-mode
        # number) if it has ≥ 2 distinct sources at EITHER type level
        # or fuzzy-cluster level. This covers synonyms across canonical
        # types (e.g. 毛艶冴え + 張り艶冴え) without lowering the strict
        # threshold itself.
        n_src = f.meta.get("n_sources", 1)
        cluster_n = f.meta.get("cluster_n_sources", 0)
        type_is_consensed = n_src >= 2 or cluster_n >= 2
        if f.polarity > 0:
            pos_w += w; pos_n += 1
            by_cat[f.category]["pos_weight"] += w
            if type_is_consensed: pos_w_c += w
        elif f.polarity < 0:
            neg_w += w; neg_n += 1
            by_cat[f.category]["neg_weight"] += w
            if type_is_consensed: neg_w_c += w
        if in_consensus_category:
            consensed_fact_count += 1
            consensed_categories.add(f.category)
        if type_is_consensed:
            n_consensed += 1

    n_facts = len(facts_for_horse)

    def _composite(pos_w_v, neg_w_v, n):
        total_w = pos_w_v + neg_w_v
        if total_w == 0 or n == 0:
            return 0.5
        net_raw = (pos_w_v - neg_w_v) / total_w
        shrink = n / (n + SHRINKAGE_PRIOR_N)
        return 0.5 + 0.5 * net_raw * shrink

    composite_all = _composite(pos_w, neg_w, n_facts)
    composite_consensed = _composite(pos_w_c, neg_w_c, n_consensed)

    return {
        # composite_condition is now the CONSENSED-only version — it is
        # what the strict trigger reads. Single-source observations do
        # not move this number at all. `composite_condition_all` remains
        # available for display/scoring layers that want the fuller view.
        "composite_condition": round(composite_consensed, 3),
        "composite_condition_all": round(composite_all, 3),
        "n_facts": n_facts,
        "n_consensed": n_consensed,
        "n_positive": pos_n,
        "n_negative": neg_n,
        "consensed_fact_count": consensed_fact_count,
        "consensed_categories": sorted(consensed_categories),
        "by_category": {k: {"pos_weight": round(v["pos_weight"], 3),
                            "neg_weight": round(v["neg_weight"], 3)}
                        for k, v in by_cat.items()},
    }
