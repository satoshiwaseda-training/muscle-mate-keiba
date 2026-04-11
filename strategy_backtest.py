"""Portfolio-strategy backtest on the 50-race enriched dataset.

Reads data/enriched_backtest_results.json (ranked + selected_top3 per race)
and data/results.json (payouts + finishing order). Runs five strategies:

  1. WIN model   : 1u on model's highest-win_prob horse
  2. WIN odds    : 1u on lowest-odds horse (baseline)
  3. 3連複 box   : 1u ticket on model's selected_top3 set
  4. 馬連 box    : 3u cost (C(3,2)) on model's selected_top3 pairs
  5. Value WIN   : 1u on each horse whose model_prob > market_prob*(1+edge)

All figures are yen-normalized to 100-yen tickets. ROI = PnL / total cost.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent
ENR_FILE = ROOT / "data" / "enriched_backtest_results.json"
RES_FILE = ROOT / "data" / "results.json"

TICKET = 100  # yen per ticket
MARKET_OVERROUND = 1.20   # JRA typical takeout on win pool
VALUE_EDGE = 0.00         # require model_prob > market_prob to bet


def _norm(n): return (n or "").strip()


def _po(s) -> float:
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_dataset():
    enriched = json.loads(ENR_FILE.read_text(encoding="utf-8"))
    results = json.loads(RES_FILE.read_text(encoding="utf-8"))
    rows = []
    for er in enriched:
        rid = er["race_id"]
        res = results.get(f"bt_{rid}")
        if not res:
            continue
        fo = res.get("finishing_order") or []
        if not fo:
            continue
        winner = second = third = None
        for h in fo:
            try:
                rk = int(h.get("rank", 0) or 0)
            except ValueError:
                continue
            nm = _norm(h.get("name"))
            if rk == 1: winner = nm
            elif rk == 2: second = nm
            elif rk == 3: third = nm
        if not winner:
            continue
        odds_map = {}
        for h in fo:
            nm = _norm(h.get("name"))
            od = _po(h.get("odds"))
            if nm and od > 0 and nm not in odds_map:
                odds_map[nm] = od
        rows.append({
            "race_id": rid,
            "race_name": er.get("race_name", ""),
            "grade": er.get("grade", ""),
            "ranked": er["ranked"],
            "selected_top3": er["selected_top3"],
            "winner": winner, "second": second, "third": third,
            "odds_map": odds_map,
            "payouts": res.get("payouts") or {},
        })
    return rows


# ── Strategies ─────────────────────────────────────────

def strategy_win_model(row: dict) -> tuple[float, float, bool]:
    """Bet 1 ticket on model's highest-win_prob horse."""
    if not row["ranked"]:
        return (0.0, 0.0, False)
    pick = row["ranked"][0]["name"]
    cost = TICKET
    hit = (pick == row["winner"])
    payout = row["payouts"].get("単勝", 0) if hit else 0
    return (cost, payout, hit)


def strategy_win_odds_fav(row: dict) -> tuple[float, float, bool]:
    """Bet 1 ticket on lowest-odds horse."""
    if not row["odds_map"]:
        return (0.0, 0.0, False)
    pick = min(row["odds_map"], key=lambda n: row["odds_map"][n])
    cost = TICKET
    hit = (pick == row["winner"])
    payout = row["payouts"].get("単勝", 0) if hit else 0
    return (cost, payout, hit)


def strategy_trio_box(row: dict) -> tuple[float, float, bool]:
    """3連複 box on selected_top3: 1 ticket, hits if top 3 finishers ⊆ set."""
    if len(row["selected_top3"]) < 3:
        return (0.0, 0.0, False)
    s = {h["name"] for h in row["selected_top3"]}
    top3 = {row["winner"], row["second"], row["third"]}
    cost = TICKET                       # 3連複 is any-order → 1 combo per triple
    hit = top3.issubset(s) and None not in top3
    payout = row["payouts"].get("3連複", 0) if hit else 0
    return (cost, payout, hit)


def strategy_trio_box_topk(row: dict, k: int = 5) -> tuple[float, float, bool]:
    """3連複 box on top-k horses by model win_prob (wider net, higher hit rate).

    Cost = C(k, 3) tickets (any-order combinations). Wins the full payout
    on the single matching combination, minus the cost of all other combos.
    """
    if len(row["ranked"]) < k or k < 3:
        return (0.0, 0.0, False)
    pool = [h["name"] for h in row["ranked"][:k]]
    n_combos = len(list(combinations(pool, 3)))
    cost = TICKET * n_combos
    top3 = {row["winner"], row["second"], row["third"]}
    hit = top3.issubset(set(pool)) and None not in top3
    payout = row["payouts"].get("3連複", 0) if hit else 0
    return (cost, payout, hit)


def strategy_quinella_topk(row: dict, k: int = 4) -> tuple[float, float, bool]:
    """馬連 box on top-k horses — wider 2-of-k net."""
    if len(row["ranked"]) < k or k < 2:
        return (0.0, 0.0, False)
    pool = [h["name"] for h in row["ranked"][:k]]
    n_pairs = len(list(combinations(pool, 2)))
    cost = TICKET * n_pairs
    pair = {row["winner"], row["second"]}
    hit = pair.issubset(set(pool)) and None not in pair
    payout = row["payouts"].get("馬連", 0) if hit else 0
    return (cost, payout, hit)


def strategy_quinella_box(row: dict) -> tuple[float, float, bool]:
    """馬連 box on selected_top3: 3 pairs, 3 tickets, hits if {winner,second} ⊆ set."""
    if len(row["selected_top3"]) < 3:
        return (0.0, 0.0, False)
    s = [h["name"] for h in row["selected_top3"]]
    n_pairs = len(list(combinations(s, 2)))  # C(3,2) = 3
    cost = TICKET * n_pairs
    pair = {row["winner"], row["second"]}
    hit = pair.issubset(set(s)) and None not in pair
    payout = row["payouts"].get("馬連", 0) if hit else 0
    return (cost, payout, hit)


def strategy_value_win(row: dict, edge: float = VALUE_EDGE,
                       max_odds: float = 20.0) -> tuple[float, float, int]:
    """Bet 1 ticket on every horse where model_prob > market_prob*(1+edge)
    AND the horse's decimal odds are ≤ max_odds.

    The max_odds cap matters because the softmax floors every horse at ~2%
    while the market correctly prices 100:1 shots at <1%. Without the cap
    every longshot looks "mispriced" purely from the probability floor.
    """
    cost = 0.0
    payout = 0.0
    n_bets = 0
    for h in row["ranked"]:
        odds = h.get("odds", 0) or 0
        if odds <= 1.0 or odds > max_odds:
            continue
        market_prob = 1.0 / (odds * MARKET_OVERROUND)
        model_prob = h.get("win_prob", 0)
        if model_prob > market_prob * (1.0 + edge):
            cost += TICKET
            n_bets += 1
            if h["name"] == row["winner"]:
                payout += row["payouts"].get("単勝", 0)
    return (cost, payout, n_bets)


def strategy_best_value_only(row: dict, min_edge: float = 0.05,
                             max_odds: float = 20.0) -> tuple[float, float, int]:
    """Take only the single largest positive-edge bet per race, if any.

    Ranks candidates by (model_prob - market_prob) / market_prob and picks
    the best one, if that edge exceeds min_edge. Single-bet-per-race keeps
    variance controlled.
    """
    best = None
    for h in row["ranked"]:
        odds = h.get("odds", 0) or 0
        if odds <= 1.0 or odds > max_odds:
            continue
        market_prob = 1.0 / (odds * MARKET_OVERROUND)
        if market_prob <= 0:
            continue
        model_prob = h.get("win_prob", 0)
        edge = (model_prob - market_prob) / market_prob
        if edge < min_edge:
            continue
        if best is None or edge > best[0]:
            best = (edge, h)
    if best is None:
        return (0.0, 0.0, 0)
    cost = TICKET
    payout = row["payouts"].get("単勝", 0) if best[1]["name"] == row["winner"] else 0
    return (cost, payout, 1)


def strategy_place_model_top(row: dict) -> tuple[float, float, bool]:
    """複勝 (place) on model's top pick.

    NOTE: JRA payout data gives 複勝 as a single number = the winning
    horse's place payout. So we can only score 複勝 bets on the WINNER
    (i.e. when model_top actually won). For a true place bet where
    finishing 2nd or 3rd also pays, we'd need per-horse place payouts
    which aren't in the result file.
    """
    if not row["ranked"]:
        return (0.0, 0.0, False)
    pick = row["ranked"][0]["name"]
    cost = TICKET
    hit = (pick == row["winner"])
    payout = row["payouts"].get("複勝", 0) if hit else 0
    return (cost, payout, hit)


# ── Runner ─────────────────────────────────────────────

def run():
    rows = load_dataset()
    print(f"Loaded {len(rows)} races with payouts\n")

    strategies = [
        ("WIN model (baseline)",        strategy_win_model),
        ("WIN odds favorite",           strategy_win_odds_fav),
        ("複勝 model top (winner-only)", strategy_place_model_top),
        ("3連複 box top-3",              strategy_trio_box),
        ("馬連 box top-3 (3u)",          strategy_quinella_box),
    ]

    print(f"{'strategy':28} {'cost':>9} {'payout':>9} {'pnl':>9} {'roi':>9} {'hits':>7}")
    print("-" * 76)
    results = {}
    for name, fn in strategies:
        total_cost = total_pay = 0.0
        hits = 0
        for row in rows:
            c, p, h = fn(row)
            total_cost += c
            total_pay += p
            if h: hits += 1
        pnl = total_pay - total_cost
        roi = (pnl / total_cost) if total_cost else 0.0
        results[name] = (total_cost, total_pay, pnl, roi, hits)
        print(f"{name:28} {total_cost:9.0f} {total_pay:9.0f} {pnl:+9.0f} {roi*100:+8.2f}% {hits:>4}/{len(rows)}")

    # Value-bet strategies with varying edge thresholds, odds capped ≤ 20
    print("\n── Value WIN (model_prob > market_prob × (1+edge)), odds ≤ 20 ──")
    print(f"{'edge':>8} {'n_bets':>8} {'cost':>9} {'payout':>9} {'pnl':>9} {'roi':>9} {'hit_rate':>10}")
    print("-" * 76)
    for edge in (0.00, 0.05, 0.10, 0.20, 0.30, 0.50):
        total_cost = total_pay = 0.0
        n_bets = winners = 0
        for row in rows:
            c, p, n = strategy_value_win(row, edge=edge, max_odds=20.0)
            total_cost += c; total_pay += p
            n_bets += n
            if p > 0: winners += 1
        pnl = total_pay - total_cost
        roi = (pnl / total_cost) if total_cost else 0.0
        rate = (winners / n_bets * 100) if n_bets else 0.0
        print(f"{edge*100:+7.0f}% {n_bets:>8} {total_cost:9.0f} {total_pay:9.0f} {pnl:+9.0f} {roi*100:+8.2f}% {rate:>8.1f}%")

    # Single-best-value-per-race strategy
    print("\n── BEST value only (1 bet/race, by relative-edge rank) ──")
    print(f"{'min_edge':>10} {'n_bets':>8} {'cost':>9} {'payout':>9} {'pnl':>9} {'roi':>9}")
    print("-" * 68)
    for me in (0.01, 0.05, 0.10, 0.20, 0.30):
        total_cost = total_pay = 0.0
        n_bets = 0
        for row in rows:
            c, p, n = strategy_best_value_only(row, min_edge=me, max_odds=20.0)
            total_cost += c; total_pay += p; n_bets += n
        pnl = total_pay - total_cost
        roi = (pnl / total_cost) if total_cost else 0.0
        print(f"{me*100:+9.0f}% {n_bets:>8} {total_cost:9.0f} {total_pay:9.0f} {pnl:+9.0f} {roi*100:+8.2f}%")

    # Wider top-k box strategies
    print("\n── 3連複 box top-k (wider hit rate at higher cost) ──")
    print(f"{'k':>4} {'cost/race':>11} {'total_cost':>11} {'total_pay':>11} {'pnl':>9} {'roi':>9} {'hits':>7}")
    print("-" * 72)
    for k in (3, 4, 5, 6):
        total_cost = total_pay = 0.0
        hits = 0
        for row in rows:
            c, p, h = strategy_trio_box_topk(row, k=k)
            total_cost += c; total_pay += p
            if h: hits += 1
        pnl = total_pay - total_cost
        roi = (pnl / total_cost) if total_cost else 0.0
        avg_cost = total_cost / len(rows) if rows else 0
        print(f"{k:>4} {avg_cost:>11.0f} {total_cost:>11.0f} {total_pay:>11.0f} {pnl:+9.0f} {roi*100:+8.2f}% {hits:>4}/{len(rows)}")

    print("\n── 馬連 box top-k ──")
    print(f"{'k':>4} {'cost/race':>11} {'total_cost':>11} {'total_pay':>11} {'pnl':>9} {'roi':>9} {'hits':>7}")
    print("-" * 72)
    for k in (3, 4, 5, 6):
        total_cost = total_pay = 0.0
        hits = 0
        for row in rows:
            c, p, h = strategy_quinella_topk(row, k=k)
            total_cost += c; total_pay += p
            if h: hits += 1
        pnl = total_pay - total_cost
        roi = (pnl / total_cost) if total_cost else 0.0
        avg_cost = total_cost / len(rows) if rows else 0
        print(f"{k:>4} {avg_cost:>11.0f} {total_cost:>11.0f} {total_pay:>11.0f} {pnl:+9.0f} {roi*100:+8.2f}% {hits:>4}/{len(rows)}")

    # Break-even benchmark: JRA takeout
    print("\n── Context: JRA takeout (minimum loss for random betting) ──")
    print("  単勝/複勝:   ≈ 20% takeout → ROI floor −20% on any unbiased flat strategy")
    print("  馬連/ワイド:  ≈ 22.5%")
    print("  馬単/3連複:   ≈ 25%")
    print("  3連単:       ≈ 27.5%")
    print("  A strategy that beats takeout has found genuine mispricing.")


if __name__ == "__main__":
    run()
