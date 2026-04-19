"""Back-test the diversified-TOP3 strategy on past G2 races.

Policy:
  - Pick 本命/対抗/単穴 via grade_strategy.pick_diversified_top3
  - Buy 単勝 × 3 + 馬連 × 3 combinations (C(3,2) = 3 pairs)
  - Cost: 600 yen/race
  - Compare ROI vs current win_prob-only TOP3 strategy.

Usage:
  python tools/simulate_diversified_g2.py
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_FILE = ROOT / "data" / "results.json"
BACKTEST_DIR_V5 = ROOT / "data" / "backtest_predictions"
BACKTEST_DIR_V4 = ROOT / "data" / "backtest_predictions_v4_n221"
BACKTEST_DIR = BACKTEST_DIR_V5 if BACKTEST_DIR_V5.exists() else BACKTEST_DIR_V4

import grade_strategy as gs

BET_PER_TICKET = 100.0
TOTAL_COST_PER_RACE = 600.0


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


def _rank_map(result: dict) -> dict:
    m = {}
    for h in (result.get("finishing_order") or []):
        try:
            r = int(h.get("rank", 0) or 0)
        except (TypeError, ValueError):
            continue
        nm = (h.get("name") or "").strip()
        if nm and r > 0:
            m[nm] = r
    return m


def _payout(result: dict, bet_type: str) -> float:
    try:
        return float((result.get("payouts") or {}).get(bet_type, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def simulate_race(pred: dict, result: dict,
                   strategy: str) -> dict | None:
    """Return per-race PnL + detail for a given strategy.

    strategy: "baseline" (current win_prob TOP3) or "diversified"
    """
    ranked = pred.get("ranked") or []
    if len(ranked) < 3:
        return None
    rank_map = _rank_map(result)
    if not rank_map:
        return None

    # Determine TOP 3 based on strategy
    if strategy == "baseline":
        top3 = [(r.get("name") or "").strip() for r in ranked[:3]]
    else:
        # Build market rank from the ranked list's odds
        mk_map = gs.build_market_rank_map(ranked)
        picked = gs.pick_diversified_top3(ranked, mk_map, strategy="diversified")
        top3 = [(h.get("name") or "").strip() for h in picked[:3]]

    if len([n for n in top3 if n]) < 3:
        return None
    top3_set = set(top3)

    # Find winner & second
    winner = next((nm for nm, r in rank_map.items() if r == 1), None)
    second = next((nm for nm, r in rank_map.items() if r == 2), None)
    if not winner:
        return None

    # Tansho (win)
    tansho_pay = _payout(result, "単勝") if winner in top3_set else 0.0

    # Umaren
    umaren_pay = 0.0
    if second:
        actual = frozenset([winner, second])
        my_pairs = {frozenset(c) for c in combinations(top3, 2)}
        if actual in my_pairs:
            umaren_pay = _payout(result, "馬連")

    return {
        "race_id":   pred.get("race_id", ""),
        "race_name": pred.get("race_name", ""),
        "grade":     pred.get("grade", ""),
        "strategy":  strategy,
        "top3":      top3,
        "winner":    winner,
        "second":    second,
        "tansho":    tansho_pay,
        "umaren":    umaren_pay,
        "payout":    tansho_pay + umaren_pay,
        "cost":      TOTAL_COST_PER_RACE,
        "pnl":       (tansho_pay + umaren_pay) - TOTAL_COST_PER_RACE,
    }


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    n = len(records)
    cost = sum(r["cost"] for r in records)
    payout = sum(r["payout"] for r in records)
    pnl = payout - cost
    t_hits = sum(1 for r in records if r["tansho"] > 0)
    u_hits = sum(1 for r in records if r["umaren"] > 0)
    return {
        "n":      n,
        "cost":   cost,
        "payout": payout,
        "pnl":    pnl,
        "roi":    round(pnl / cost, 4) if cost else 0.0,
        "tansho_hit_rate": round(t_hits / n, 4),
        "umaren_hit_rate": round(u_hits / n, 4),
        "any_hit_rate":    round(sum(1 for r in records
                                     if r["payout"] > 0) / n, 4),
    }


def main() -> int:
    results = _load_results()

    g2_preds = []
    for p in sorted(BACKTEST_DIR.glob("*_on.json")):
        try:
            pred = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "G2" not in (pred.get("grade", "") or ""):
            continue
        g2_preds.append(pred)

    print(f"[simulate] G2 preds: {len(g2_preds)}")

    baseline_recs = []
    diverse_recs  = []
    detail = []
    for pred in g2_preds:
        rid = pred.get("race_id", "")
        res = results.get(rid)
        if not res:
            continue
        b = simulate_race(pred, res, "baseline")
        d = simulate_race(pred, res, "diversified")
        if not b or not d:
            continue
        baseline_recs.append(b)
        diverse_recs.append(d)
        detail.append((b, d))

    print(f"[simulate] evaluable: {len(detail)}")
    print("")
    print("=" * 72)
    print("BASELINE (現行 win_prob TOP3) — G2 のみ")
    print("=" * 72)
    b_agg = aggregate(baseline_recs)
    _print_agg(b_agg)

    print("")
    print("=" * 72)
    print("DIVERSIFIED (市場分散 TOP3) — G2 のみ")
    print("=" * 72)
    d_agg = aggregate(diverse_recs)
    _print_agg(d_agg)

    print("")
    print("=" * 72)
    print("DELTA (diversified − baseline)")
    print("=" * 72)
    print(f"  cost:       {d_agg['cost']-b_agg['cost']:+,.0f} 円 "
          f"(同じ 600円/R × {d_agg['n']}R)")
    print(f"  payout:     {d_agg['payout']-b_agg['payout']:+,.0f} 円")
    print(f"  pnl:        {d_agg['pnl']-b_agg['pnl']:+,.0f} 円")
    print(f"  ROI:        {(d_agg['roi']-b_agg['roi'])*100:+.2f} pp")
    print(f"  単勝 hit:    {(d_agg['tansho_hit_rate']-b_agg['tansho_hit_rate'])*100:+.2f} pp")
    print(f"  馬連 hit:    {(d_agg['umaren_hit_rate']-b_agg['umaren_hit_rate'])*100:+.2f} pp")

    # Side-by-side comparison for races where strategies differ
    print("\n── race-level diff (first 15) ──")
    print(f"  {'race_name':<25s}   {'baseline':>10s} {'diverse':>10s}   delta")
    for b, d in detail[:15]:
        if b["pnl"] != d["pnl"]:
            print(f"  {b['race_name'][:25]:<25s}   "
                  f"{b['pnl']:>+10,.0f} {d['pnl']:>+10,.0f}   "
                  f"{d['pnl']-b['pnl']:>+7,.0f}")

    return 0


def _print_agg(agg):
    print(f"  races:         {agg['n']}")
    print(f"  cost:          {agg['cost']:,.0f} 円")
    print(f"  payout:        {agg['payout']:,.0f} 円")
    print(f"  pnl:           {agg['pnl']:+,.0f} 円")
    print(f"  ROI:           {agg['roi']*100:+.1f}%")
    print(f"  単勝 hit rate:  {agg['tansho_hit_rate']*100:.1f}%")
    print(f"  馬連 hit rate:  {agg['umaren_hit_rate']*100:.1f}%")
    print(f"  any hit:       {agg['any_hit_rate']*100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
