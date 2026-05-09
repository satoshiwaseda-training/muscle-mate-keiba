"""Grade-specific TOP-3 selection strategies.

Live app is scoped to G1/G2 only. G2 uses a diversified market-rank
presentation; G1 keeps the win_prob baseline.

G2 診断結果 (tools/analyze_g2_misses.py):
  - G2 勝ち馬の 48% は市場 4-10 番人気
  - 現行モデルは本命 100% を市場 1-3 番人気に集中
  ⇒ 構造的に G2 の半分の勝利を取れない

対策 (pick_diversified_top3):
  本命 (◎): 市場 1-3 番人気の中で win_prob 最大の馬
  対抗 (○): 市場 4-7 番人気の中で win_prob 最大の馬
  単穴 (▲): 市場 8 番人気以下の中で win_prob 最大の馬

同じ 600 円 (単勝×3 + 馬連×3) で、買い目が市場全体に広がる。

注意: この module は推論ロジックではなく **買い方の提示層**。
モデルの `ranked` (win_prob 降順) は触らない。憲法 §1.3「勝てるのは
条件付き」の通り、提示法を変えるだけで勝率保証はしない。
"""

from __future__ import annotations

from typing import Optional


STRATEGY_VERSION = "grade-strategy-v5.11-2026-05-09-dual-candidates"
EXPERIMENTAL_STRATEGY_VERSION = "experimental-jockey-trainer-v1-2026-05-09"


# ─── Named strategies ───
# Each strategy is a list of (label, mark, (market_rank_lo, market_rank_hi))
#
# "win_prob" is not listed here — it's the baseline (take ranked[:3] as-is).

STRATEGIES: dict = {
    "diversified_1-3_4-7_8+": [
        ("本命", "◎", (1, 3)),
        ("対抗", "○", (4, 7)),
        ("単穴", "▲", (8, 99)),
    ],
    "tight_1-2_3-5_6+": [
        ("本命", "◎", (1, 2)),
        ("対抗", "○", (3, 5)),
        ("単穴", "▲", (6, 99)),
    ],
    "loose_1-4_5-9_10+": [
        ("本命", "◎", (1, 4)),
        ("対抗", "○", (5, 9)),
        ("単穴", "▲", (10, 99)),
    ],
    "mid_heavy_1-2_3-6_7+": [
        ("本命", "◎", (1, 2)),
        ("対抗", "○", (3, 6)),
        ("単穴", "▲", (7, 99)),
    ],
    "wide_穴_1-3_4-8_9+": [
        ("本命", "◎", (1, 3)),
        ("対抗", "○", (4, 8)),
        ("単穴", "▲", (9, 99)),
    ],
}

# Backward-compat aliases used by old call sites
DIVERSIFIED_BUCKETS = STRATEGIES["diversified_1-3_4-7_8+"]
BALANCED_BUCKETS    = STRATEGIES["tight_1-2_3-5_6+"]


# Per-grade strategy mapping (v5.8 tuned via tools/grid_search_strategy.py
# on 221 R backtest with backfilled snapshots).
#
# 結果サマリ:
#   G1  (n=39):  win_prob                 ROI -5.7%  (ex-big2 -19.8%, 最頑健)
#                diversified_1-3_4-7_8+   ROI +0.1%  (ex-big2 -48.3%, 過学習疑義)
#                → 現行維持 (小サンプル、robustness 重視)
#   G2  (n=63):  diversified_1-3_4-7_8+   ROI +20.8% (ex-big2 -36.3%) ← 採用
#
# 期待される改善: G1/G2 専用の live 運用で検証を継続
GRADE_STRATEGY: dict = {
    "G1":     "win_prob",
    "JpnI":   "win_prob",
    "G2":     "diversified_1-3_4-7_8+",
    "JpnII":  "diversified_1-3_4-7_8+",
}


def get_strategy_for_grade(grade: str) -> str:
    """Return the strategy name for this race grade."""
    if not grade:
        return "win_prob"
    g = grade.upper()
    for key in ("G1", "G2", "JPNI", "JPNII"):
        if key in g:
            # Normalize lookup key to our mapping (JPNII -> JpnII)
            lookup = {"JPNI": "JpnI", "JPNII": "JpnII", "JPNIII": "JpnIII"}.get(key, key)
            return GRADE_STRATEGY.get(lookup, "win_prob")
    return "win_prob"


def pick_diversified_top3(ranked: list[dict],
                           market_rank_map: dict[str, int],
                           strategy: str = "diversified_1-3_4-7_8+") -> list[dict]:
    """Pick 3 horses spread across market-rank buckets.

    Args:
      ranked:          model output, list of {name, win_prob, odds, ...}
                       sorted by win_prob desc.
      market_rank_map: {horse_name: market_rank (1=最低 odds)}.
      strategy:        one of STRATEGIES keys, or legacy "diversified"/"balanced".

    Returns:
      list of 3 dicts (or fewer if field too small), each with:
        {**ranked_entry, "bucket_label", "bucket_mark", "market_rank"}
    """
    # Legacy aliases
    if strategy == "diversified":
        strategy = "diversified_1-3_4-7_8+"
    elif strategy == "balanced":
        strategy = "tight_1-2_3-5_6+"

    buckets = STRATEGIES.get(strategy, DIVERSIFIED_BUCKETS)

    # ranked is already sorted by win_prob desc. For each bucket, pick the
    # first horse whose market_rank falls in that bucket and hasn't been
    # picked yet.
    picked: list[dict] = []
    picked_names: set[str] = set()

    for label, mark, (lo, hi) in buckets:
        chosen = None
        for h in ranked:
            nm = (h.get("name") or "").strip()
            if not nm or nm in picked_names:
                continue
            mr = market_rank_map.get(nm)
            if mr is None:
                continue
            if lo <= mr <= hi:
                chosen = dict(h)
                chosen["bucket_label"] = label
                chosen["bucket_mark"]  = mark
                chosen["market_rank"]  = mr
                break
        if chosen:
            picked.append(chosen)
            picked_names.add((chosen.get("name") or "").strip())

    # Fallback: if any bucket empty (e.g. field has only 6 horses so
    # market rank 8+ doesn't exist), fill from highest win_prob remaining
    while len(picked) < 3:
        filled = False
        for h in ranked:
            nm = (h.get("name") or "").strip()
            if nm and nm not in picked_names:
                fallback = dict(h)
                fallback["bucket_label"] = f"補欠 ({len(picked)+1})"
                fallback["bucket_mark"]  = "△"
                fallback["market_rank"]  = market_rank_map.get(nm, 99)
                picked.append(fallback)
                picked_names.add(nm)
                filled = True
                break
        if not filled:
            break

    return picked


def _median(values: list[float]) -> float:
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def jockey_trainer_experiment_score(
    horse: dict,
    jockey_median: float = 0.0,
    trainer_median: float = 0.0,
) -> float:
    """Score for the experimental jockey/trainer candidate panel.

    This is a display and paper-trading candidate. It does not change the
    model probability, loose trigger rule, or investment thresholds.
    """
    try:
        jockey_win = float(horse.get("jockey_win_rate", 0) or 0)
    except (TypeError, ValueError):
        jockey_win = 0.0
    try:
        trainer_win = float(horse.get("trainer_win_rate", 0) or 0)
    except (TypeError, ValueError):
        trainer_win = 0.0
    try:
        win_prob = float(horse.get("win_prob", 0) or 0)
    except (TypeError, ValueError):
        win_prob = 0.0

    score = 0.0
    if jockey_median > 0 and jockey_win >= jockey_median:
        score += 0.04
    if trainer_median > 0 and trainer_win >= trainer_median:
        score += 0.03
    # Tiny tie-breaker so a field with no jockey/trainer rates still returns
    # a stable order without pretending the experiment has extra evidence.
    return round(score + win_prob * 0.001, 6)


def pick_jockey_trainer_experimental_top3(
    ranked: list[dict],
    market_rank_map: dict[str, int],
    strategy: str = "diversified_1-3_4-7_8+",
) -> list[dict]:
    """Pick top3 by market buckets, choosing within each bucket by
    jockey/trainer strength instead of model win_prob.
    """
    if strategy == "diversified":
        strategy = "diversified_1-3_4-7_8+"
    elif strategy == "balanced":
        strategy = "tight_1-2_3-5_6+"

    buckets = STRATEGIES.get(strategy, DIVERSIFIED_BUCKETS)
    jockey_med = _median([
        float(h.get("jockey_win_rate", 0) or 0)
        for h in ranked or []
        if isinstance(h.get("jockey_win_rate", 0), (int, float))
    ])
    trainer_med = _median([
        float(h.get("trainer_win_rate", 0) or 0)
        for h in ranked or []
        if isinstance(h.get("trainer_win_rate", 0), (int, float))
    ])

    enriched = []
    for h in ranked or []:
        row = dict(h)
        row["experimental_score"] = jockey_trainer_experiment_score(
            row, jockey_med, trainer_med,
        )
        row["experimental_strategy_version"] = EXPERIMENTAL_STRATEGY_VERSION
        enriched.append(row)

    experimental_ranked = sorted(
        enriched,
        key=lambda h: (
            float(h.get("experimental_score", 0) or 0),
            float(h.get("win_prob", 0) or 0),
        ),
        reverse=True,
    )

    picked: list[dict] = []
    picked_names: set[str] = set()
    for label, mark, (lo, hi) in buckets:
        chosen = None
        for h in experimental_ranked:
            nm = (h.get("name") or "").strip()
            if not nm or nm in picked_names:
                continue
            mr = market_rank_map.get(nm)
            if mr is None:
                continue
            if lo <= mr <= hi:
                chosen = dict(h)
                chosen["bucket_label"] = label
                chosen["bucket_mark"] = mark
                chosen["market_rank"] = mr
                chosen["candidate_type"] = "experimental_jockey_trainer"
                break
        if chosen:
            picked.append(chosen)
            picked_names.add((chosen.get("name") or "").strip())

    while len(picked) < 3:
        filled = False
        for h in experimental_ranked:
            nm = (h.get("name") or "").strip()
            if nm and nm not in picked_names:
                fallback = dict(h)
                fallback["bucket_label"] = f"補欠 ({len(picked)+1})"
                fallback["bucket_mark"] = "△"
                fallback["market_rank"] = market_rank_map.get(nm, 99)
                fallback["candidate_type"] = "experimental_jockey_trainer"
                picked.append(fallback)
                picked_names.add(nm)
                filled = True
                break
        if not filled:
            break

    return picked


def build_prediction_variants(ranked: list[dict], grade: str) -> dict:
    """Return primary and experimental top3 candidates for storage/UI."""
    strategy = get_strategy_for_grade(grade)
    market = build_market_rank_map(ranked)
    if strategy == "win_prob":
        primary = [dict(h) for h in (ranked or [])[:3]]
    else:
        primary = pick_diversified_top3(ranked, market, strategy=strategy)

    experimental_strategy = (
        strategy if strategy != "win_prob" else "diversified_1-3_4-7_8+"
    )
    experimental = pick_jockey_trainer_experimental_top3(
        ranked, market, strategy=experimental_strategy,
    )

    return {
        "primary": {
            "candidate_id": "primary_current",
            "label": "第一候補",
            "strategy": strategy,
            "strategy_version": STRATEGY_VERSION,
            "description": "現行ロジックのtop3",
            "top3": primary,
        },
        "experimental": {
            "candidate_id": "experimental_jockey_trainer",
            "label": "実験候補",
            "strategy": "feature_only_jockey_trainer_combo",
            "strategy_version": EXPERIMENTAL_STRATEGY_VERSION,
            "description": "各人気帯で騎手/調教師勝率を優先",
            "top3": experimental,
        },
    }


def should_apply_diversified(grade: str) -> bool:
    """Return True if this race's grade calls for any strategy != win_prob.

    Live scope is G1/G2 only; this returns True only for configured
    non-baseline strategies such as G2's market-diversified presentation.
    """
    return get_strategy_for_grade(grade) != "win_prob"


def build_market_rank_map(ranked_or_entries: list[dict]) -> dict[str, int]:
    """Build {horse_name: market_rank} from ranked list or entries.

    Uses odds ascending (lowest odds = rank 1 = 人気最上位).
    Horses with odds <= 1.0 are excluded.
    """
    decorated = []
    for h in ranked_or_entries or []:
        try:
            od = float(h.get("odds", 0) or 0)
        except (TypeError, ValueError):
            continue
        nm = (h.get("name") or "").strip()
        if not nm or od <= 1.0:
            continue
        decorated.append((od, nm))
    decorated.sort(key=lambda x: x[0])
    return {nm: i + 1 for i, (_, nm) in enumerate(decorated)}
