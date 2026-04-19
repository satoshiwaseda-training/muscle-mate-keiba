"""Grade-specific TOP-3 selection strategies (v5.7 — 2026-04-19).

G2 の買い方を「市場分散」に変更する機能。G1/G3 は従来の win_prob 基準。

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


STRATEGY_VERSION = "grade-strategy-v5.7-2026-04-19"


# ─── Market rank buckets for each strategy ───
DIVERSIFIED_BUCKETS = [
    ("本命", "◎", (1, 3)),     # 市場 1-3 番人気
    ("対抗", "○", (4, 7)),     # 市場 4-7 番人気
    ("単穴", "▲", (8, 99)),    # 市場 8 番人気以下
]

BALANCED_BUCKETS = [
    ("本命", "◎", (1, 2)),     # 市場 1-2 番人気 (厳しめ)
    ("対抗", "○", (3, 5)),
    ("単穴", "▲", (6, 99)),
]


def pick_diversified_top3(ranked: list[dict],
                           market_rank_map: dict[str, int],
                           strategy: str = "diversified") -> list[dict]:
    """Pick 3 horses spread across market-rank buckets.

    Args:
      ranked:          model output, list of {name, win_prob, odds, ...}
                       sorted by win_prob desc.
      market_rank_map: {horse_name: market_rank (1=最低 odds)}.
      strategy:        "diversified" | "balanced"

    Returns:
      list of 3 dicts (or fewer if field too small), each with:
        {
          **original_ranked_entry,
          "bucket_label": "本命"/"対抗"/"単穴",
          "bucket_mark":  "◎"/"○"/"▲",
          "market_rank":  int,
        }
      If a bucket has no eligible horse, the slot is filled by the
      highest-win_prob horse remaining in the ranked list (fallback).
    """
    if strategy == "balanced":
        buckets = BALANCED_BUCKETS
    else:
        buckets = DIVERSIFIED_BUCKETS

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
    """Return True if this race's grade calls for the diversified strategy.

    現在は G2 のみ (analyze_g2_misses.py の結果に基づく)。
    他 grade での効果が確認できたら拡張する。
    """
    if not grade:
        return False
    g = grade.upper()
    return "G2" in g or "JPNII" in g


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
