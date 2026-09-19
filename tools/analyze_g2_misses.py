"""G2 の TOP3 外しを統計診断する。

前提:
  - G2 ROI は -48.2% で全 grade 中最悪
  - G1 は +14.3pp vs takeout、G3 は -16.8pp
  - 同じモデル・同じ買い方 (単勝×3 + 馬連×3) なのに G2 だけ深く赤字
  ⇒ 何か G2 固有のパターンがあるはず

このスクリプトは G2 レースを対象に以下を集計する:

  1. 勝ち馬の model rank 分布 (1-18 のどこにいた?)
  2. 勝ち馬の market rank 分布 (市場 1-18 のどこにいた?)
  3. モデル本命 vs 勝ち馬の market_rank 差 (= 市場超過方向の偏り)
  4. 芝/ダート/距離別の的中率
  5. race_name pattern (重賞タイプ別)
  6. モデル TOP 3 の odds 傾向 (本命が何倍馬だったか)

出力は「**この方向に偏った外し方をしている**」を示すテーブル。
そこから改善方針 (calibration 調整 / feature 重み / fact 追加) を決める。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "data" / "results.json"
BACKTEST_DIR_V5 = ROOT / "data" / "backtest_predictions"
BACKTEST_DIR_V4 = ROOT / "data" / "backtest_predictions_v4_n221"
BACKTEST_DIR = BACKTEST_DIR_V5 if BACKTEST_DIR_V5.exists() else BACKTEST_DIR_V4


def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    raw = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {(k[3:] if k.startswith("bt_") else k): v for k, v in raw.items()}


def _parse_odds(raw) -> float:
    try:
        return float(str(raw).replace("---", "0").replace("--", "0").replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _winner(result: dict) -> Optional[str]:
    for h in (result.get("finishing_order") or []):
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return (h.get("name") or "").strip()
        except (TypeError, ValueError):
            continue
    return None


def _rank_second(result: dict) -> Optional[str]:
    for h in (result.get("finishing_order") or []):
        try:
            if int(h.get("rank", 0) or 0) == 2:
                return (h.get("name") or "").strip()
        except (TypeError, ValueError):
            continue
    return None


def _market_rank_map(result: dict) -> dict:
    """name → 市場人気順位 (1=最低 odds)."""
    fo = result.get("finishing_order") or []
    decorated = []
    for h in fo:
        nm = (h.get("name") or "").strip()
        od = _parse_odds(h.get("odds"))
        if nm and od > 0:
            decorated.append((od, nm))
    decorated.sort(key=lambda x: x[0])
    return {nm: i + 1 for i, (_, nm) in enumerate(decorated)}


def _race_attributes(race_name: str) -> dict:
    """Parse surface / distance from race_name if possible."""
    surface = ""
    distance = ""
    # Most netkeiba race names don't embed surface/distance, so this is
    # only best-effort.
    if "ダート" in race_name or "ダ" in race_name:
        surface = "ダ"
    elif "芝" in race_name:
        surface = "芝"
    m = re.search(r"(\d{3,4})m", race_name)
    if m:
        distance = m.group(1)
    return {"surface": surface, "distance": distance}


def analyze():
    results = _load_results()
    if not BACKTEST_DIR.exists():
        print("no backtest dir"); return

    g2_preds = []
    for p in sorted(BACKTEST_DIR.glob("*_on.json")):
        try:
            pred = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "G2" not in (pred.get("grade", "") or ""):
            continue
        g2_preds.append(pred)

    print(f"[analyze-g2] G2 preds: {len(g2_preds)}")

    miss_rows = []
    hit_rows = []
    winner_model_rank_counter = Counter()
    winner_market_rank_counter = Counter()
    model_top1_market_rank_dist = Counter()
    model_top1_odds_bucket = Counter()

    for pred in g2_preds:
        rid = pred.get("race_id", "")
        res = results.get(rid)
        if not res:
            continue
        winner = _winner(res)
        if not winner:
            continue
        ranked = pred.get("ranked") or []
        if len(ranked) < 3:
            continue
        model_names = [(r.get("name") or "").strip() for r in ranked]
        model_top3_set = set(model_names[:3])
        model_rank_of_winner = next(
            (i + 1 for i, nm in enumerate(model_names) if nm == winner),
            None,
        )
        winner_model_rank_counter[model_rank_of_winner or 99] += 1

        market_rank = _market_rank_map(res)
        winner_market_rank = market_rank.get(winner, 99)
        winner_market_rank_counter[winner_market_rank] += 1

        model_top1_name = model_names[0] if model_names else ""
        model_top1_market_rank = market_rank.get(model_top1_name, 99)
        model_top1_market_rank_dist[model_top1_market_rank] += 1

        # odds bucket of model top-1
        top1_odds = 0.0
        for r in ranked:
            if (r.get("name") or "").strip() == model_top1_name:
                top1_odds = float(r.get("odds", 0) or 0)
                break
        bucket = "<3" if top1_odds < 3 else \
                 "3-5" if top1_odds < 5 else \
                 "5-10" if top1_odds < 10 else \
                 "10-20" if top1_odds < 20 else \
                 ">=20"
        model_top1_odds_bucket[bucket] += 1

        row = {
            "race_id": rid,
            "race_name": pred.get("race_name", ""),
            "winner": winner,
            "model_top3": model_names[:3],
            "model_top1": model_top1_name,
            "model_rank_of_winner": model_rank_of_winner,
            "winner_market_rank": winner_market_rank,
            "model_top1_market_rank": model_top1_market_rank,
            "model_top1_odds": top1_odds,
        }
        if winner in model_top3_set:
            hit_rows.append(row)
        else:
            miss_rows.append(row)

    n = len(hit_rows) + len(miss_rows)
    print(f"\n[analyze-g2] evaluated: {n}  hit: {len(hit_rows)}  miss: {len(miss_rows)}")
    print(f"  hit rate: {len(hit_rows)/n*100:.1f}%" if n else "")

    print("\n── 勝ち馬の model rank 分布 (全 G2 レース) ──")
    for rank in sorted(winner_model_rank_counter):
        c = winner_model_rank_counter[rank]
        marker = "◎" if rank <= 3 else ""
        print(f"  model rank {rank:>2}: {c:>3d} 回 {marker}")

    print("\n── 勝ち馬の market rank 分布 (全 G2 レース) ──")
    for rank in sorted(winner_market_rank_counter):
        c = winner_market_rank_counter[rank]
        marker = "★" if rank <= 3 else ""
        print(f"  market rank {rank:>2}: {c:>3d} 回 {marker}")

    print("\n── モデル本命の market rank 分布 (モデルが市場の何番人気を本命にしたか) ──")
    for rank in sorted(model_top1_market_rank_dist):
        c = model_top1_market_rank_dist[rank]
        print(f"  本命 = 市場 {rank:>2} 番人気: {c:>3d} 回")

    print("\n── モデル本命の odds 帯 ──")
    for bucket in ("<3", "3-5", "5-10", "10-20", ">=20"):
        c = model_top1_odds_bucket.get(bucket, 0)
        print(f"  本命 odds {bucket:<6s}: {c:>3d} 回")

    # --- Misses: どこで外したか集中的に ---
    print("\n── MISS サンプル (先頭 15 件) ──")
    miss_rows.sort(key=lambda r: r["model_rank_of_winner"] or 999, reverse=True)
    for r in miss_rows[:15]:
        top3 = " / ".join(r["model_top3"])
        print(f"  {r['race_name'][:20]:<20s} "
              f"winner={r['winner'][:12]:<12s} "
              f"winner model_rank={r['model_rank_of_winner']} "
              f"market_rank={r['winner_market_rank']:>2} "
              f"model_top1_odds={r['model_top1_odds']:.1f}")

    # --- 診断: モデル TOP1 は市場の何番人気が多いか? ---
    below_market_avg = [r["model_top1_market_rank"]
                        for r in miss_rows
                        if r["model_top1_market_rank"] < 99]
    if below_market_avg:
        print(f"\n[diagnosis] miss 時の model top1 平均市場順位: "
              f"{sum(below_market_avg)/len(below_market_avg):.1f}")
        print(f"[diagnosis] hit 時:")
        hit_top1_mk = [r["model_top1_market_rank"] for r in hit_rows
                       if r["model_top1_market_rank"] < 99]
        if hit_top1_mk:
            print(f"  {sum(hit_top1_mk)/len(hit_top1_mk):.1f}")

    winner_mk_miss_avg = sum(r["winner_market_rank"] for r in miss_rows) / len(miss_rows) if miss_rows else 0
    winner_mk_hit_avg = sum(r["winner_market_rank"] for r in hit_rows) / len(hit_rows) if hit_rows else 0
    print(f"\n[diagnosis] miss レースの winner 平均市場順位: {winner_mk_miss_avg:.1f}")
    print(f"[diagnosis] hit  レースの winner 平均市場順位: {winner_mk_hit_avg:.1f}")


if __name__ == "__main__":
    analyze()
