"""Grade-specific TOP-3 selection strategies (v5.9 — 2026-05-08).

馬連 BOX 3点を主眼に、グレードごとに TOP-3 の提示方法を切り替える。
この module は推論ロジックではなく **買い方の提示層**。

G2 診断結果 (tools/analyze_g2_misses.py / grid_search_strategy.py):
  - G2 勝ち馬の 48% は市場 4-10 番人気
  - 現行モデルは本命 100% を市場 1-3 番人気に集中
  ⇒ 構造的に G2 の半分の勝利を取れない

2026-05-08 馬連専用レビュー:
  - 先週の G1/G2 はどちらも 2着人気薄で馬連高配当
    - 京王杯SC(G2): 6-16, 24,590円
    - 天皇賞春(G1): 7-15, 18,240円
  - 221 R backtest の馬連3点のみ評価では、
    G1 も win_prob TOP3 より市場分散のROIが高い
    - G1 win_prob: -11.3%
    - G1 diversified_1-3_4-7_8+: +22.0%
  - G2 は現行の市場分散が引き続き最上位

対策 (pick_diversified_top3):
  本命 (◎): 市場 1-3 番人気の中で win_prob 最大の馬
  対抗 (○): 市場 4-7 番人気の中で win_prob 最大の馬
  単穴 (▲): 市場 8 番人気以下の中で win_prob 最大の馬

馬連 BOX 3点で、買い目が市場全体に広がる。

モデルの `ranked` (win_prob 降順) は触らない。憲法 §1.3「勝てるのは
条件付き」の通り、提示法を変えるだけで勝率保証はしない。
"""

from __future__ import annotations

from typing import Optional


STRATEGY_VERSION = "grade-strategy-v5.9-2026-05-08-umaren"


# Recent 90-day review (2026-02-08..2026-05-08, local result data through
# 2026-04-05). This is not used to change the frozen loose trigger or model
# score. It only annotates the human-facing TOP-3 betting panel.
#
# The main lesson is that 馬連 alone is extremely volatile in small windows:
#   - ALL grades, 馬連3点 best ROI was still -52.4%
#   - 馬連+ワイド reduced the drawdown to -25.5% on the same 26 races
# Therefore the operational improvement is a stake discipline warning, not
# a threshold chase.
RECENT_UMAREN_GUARD: dict = {
    "G1": {
        "sample": 3,
        "umaren_roi": -0.011,
        "umaren_wide_roi": -0.111,
        "message": (
            "直近3ヶ月のG1はサンプル3Rで馬連ROI -1%。"
            "過信せず、馬連BOXは通常額まで。"
        ),
    },
    "G2": {
        "sample": 9,
        "umaren_roi": -0.607,
        "umaren_wide_roi": 0.163,
        "message": (
            "直近3ヶ月のG2は馬連単独が不安定。"
            "馬連BOXにワイド保険を併用すると同期間ROIは改善。"
        ),
    },
    "G3": {
        "sample": 14,
        "umaren_roi": -0.581,
        "umaren_wide_roi": -0.088,
        "message": (
            "直近3ヶ月のG3は高配当決着が多く、馬連単独はドローダウン大。"
            "見送りまたはワイド保険を検討。"
        ),
    },
}


# User's actual purchase style:
#   単勝 3点 + 馬連BOX 3点 = 600円 / race
#
# Current 221R backtest with G1/G2 diversified and G3 loose:
#   ALL +4.2% ROI, but the edge is concentrated in G2.
#   G1 +0.1% (flat, small n) / G2 +20.8% / G3 -3.2%.
#
# This is intentionally a betting-action layer. It does not change ranking,
# loose trigger rules, or model probabilities. Predictions remain visible for
# every grade; stake discipline decides whether the 600円 ticket is warranted.
BUYING_STYLE_VERSION = "buying-style-v1-2026-05-08"
BUYING_STYLE_STAKE_PLAN: dict = {
    "G1": {
        "action": "WATCH",
        "stake_yen": 0,
        "label": "予想のみ",
        "historical_roi": 0.001,
        "recent_roi": -0.667,
        "reason": (
            "G1は長期でほぼ損益ゼロ、直近3ヶ月は不調。"
            "馬券は強い追加根拠がある時だけ。"
        ),
    },
    "G2": {
        "action": "BET",
        "stake_yen": 600,
        "label": "単勝3点 + 馬連BOX3点",
        "historical_roi": 0.208,
        "recent_roi": 0.193,
        "reason": (
            "この買い方の収益源。221R検証でも直近3ヶ月でもG2だけはプラス。"
        ),
    },
    "G3": {
        "action": "WATCH",
        "stake_yen": 0,
        "label": "見送り優先",
        "historical_roi": -0.032,
        "recent_roi": -0.708,
        "reason": (
            "G3は波が大きく直近ドローダウンも深い。"
            "公開情報の上積みが薄い日は見送り。"
        ),
    },
}


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


# Per-grade strategy mapping (v5.9 tuned via:
#   - tools/grid_search_strategy.py for legacy 単勝×3 + 馬連×3
#   - 2026-05-08 ad-hoc 馬連3点 ROI review for the user's actual purchase mode
# on 221 R backtest with backfilled snapshots).
#
# 結果サマリ:
#   馬連3点のみ:
#     G1  (n=39):  diversified_1-3_4-7_8+ ROI +22.0% ← 採用
#                  win_prob               ROI -11.3%
#     G2  (n=63):  diversified_1-3_4-7_8+ ROI  +3.0% ← 採用
#                  win_prob               ROI -56.9%
#     G3  (n=119): loose_1-4_5-9_10+      ROI +33.2% ← 維持
#
# Note: legacy 600円/R 評価では G1 diversified の ex-big2 が弱い。
# ただしユーザ運用が馬連中心に変わったため、ここでは馬連ROIを優先。
GRADE_STRATEGY: dict = {
    "G1":     "diversified_1-3_4-7_8+",
    "JpnI":   "diversified_1-3_4-7_8+",
    "G2":     "diversified_1-3_4-7_8+",
    "JpnII":  "diversified_1-3_4-7_8+",
    "G3":     "loose_1-4_5-9_10+",
    "JpnIII": "loose_1-4_5-9_10+",
}


def get_strategy_for_grade(grade: str) -> str:
    """Return the strategy name for this race grade."""
    if not grade:
        return "win_prob"
    g = grade.upper()
    for key in ("G1", "G2", "G3", "JPNI", "JPNII", "JPNIII"):
        if key in g:
            # Normalize lookup key to our mapping (JPNII -> JpnII)
            lookup = {"JPNI": "JpnI", "JPNII": "JpnII", "JPNIII": "JpnIII"}.get(key, key)
            return GRADE_STRATEGY.get(lookup, "win_prob")
    return "win_prob"


def recent_umaren_guard_for_grade(grade: str) -> dict:
    """Return recent-performance warning metadata for the display layer."""
    if not grade:
        return {}
    g = grade.upper()
    for key in ("G1", "G2", "G3"):
        if key in g:
            return RECENT_UMAREN_GUARD.get(key, {})
    return {}


def buying_style_plan_for_grade(grade: str) -> dict:
    """Return the user's 600円 buying-style stake plan for a race grade."""
    if not grade:
        return {
            "action": "WATCH",
            "stake_yen": 0,
            "label": "予想のみ",
            "historical_roi": 0.0,
            "recent_roi": 0.0,
            "reason": "グレード不明のため投入判断を保留。",
        }
    g = grade.upper()
    for key in ("G1", "G2", "G3"):
        if key in g:
            return BUYING_STYLE_STAKE_PLAN.get(key, {})
    return {
        "action": "WATCH",
        "stake_yen": 0,
        "label": "予想のみ",
        "historical_roi": 0.0,
        "recent_roi": 0.0,
        "reason": "G1/G2/G3以外は現行検証対象外。",
    }


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


def should_apply_diversified(grade: str) -> bool:
    """Return True if this race's grade calls for any strategy != win_prob.

    v5.8 以降、G2 だけでなく G3 も diversified 系 (loose_1-4_5-9_10+) を
    使うので、単なる「diversified か否か」フラグではなく、
    「strategy != win_prob か」を返す設計。
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
