"""Temperature sweep over the saved 50-race enriched dataset.

Resoftmax raw scores with T ∈ {4, 5, 6, 8, 12}, rerun value-bet
strategies with max_odds ∈ {∞, 20}, and report the effect on:

  - probability distribution (mean top prob, mean floor prob, spread)
  - longshot bet share (bets at odds > 20)
  - value-bet count, hit rate, ROI
  - flat WIN baseline (for reference — unaffected by T)

No network calls. Reads only data/enriched_backtest_results.json and
data/results.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent
ENR = ROOT / "data" / "enriched_backtest_results.json"
RES = ROOT / "data" / "results.json"

TICKET = 100
OVERROUND = 1.20


def softmax(scores, T):
    T = max(T, 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / T) for s in scores]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]


def _norm(n): return (n or "").strip()


def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def load_dataset():
    enriched = json.loads(ENR.read_text(encoding="utf-8"))
    results = json.loads(RES.read_text(encoding="utf-8"))
    rows = []
    for er in enriched:
        rid = er["race_id"]
        res = results.get(f"bt_{rid}")
        if not res:
            continue
        fo = res.get("finishing_order") or []
        winner = second = None
        for h in fo:
            try:
                rk = int(h.get("rank", 0) or 0)
            except ValueError:
                continue
            nm = _norm(h.get("name"))
            if rk == 1: winner = nm
            elif rk == 2: second = nm
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
            "ranked_scores": [(h["name"], h["score"], h["odds"]) for h in er["ranked"]],
            "winner": winner,
            "second": second,
            "odds_map": odds_map,
            "payouts": res.get("payouts") or {},
        })
    return rows


def eval_temperature(rows, T, max_odds=None, min_edge=0.01):
    """Run value-bet strategy under a given softmax T.

    Returns dict with all the key metrics."""
    total_cost = total_pay = 0.0
    n_bets = winners = 0
    longshot_bets = 0

    # prob distribution tracking
    top_probs = []
    floor_probs = []
    prob_spreads = []
    value_win_bet_count_dist = []

    # flat baseline
    flat_cost = flat_pay = 0.0
    flat_hits = 0

    for row in rows:
        names = [t[0] for t in row["ranked_scores"]]
        scores = [t[1] for t in row["ranked_scores"]]
        odds = [t[2] for t in row["ranked_scores"]]
        if not names:
            continue
        probs = softmax(scores, T)

        # stats
        top_probs.append(probs[0])
        floor_probs.append(min(probs))
        prob_spreads.append(probs[0] - min(probs))

        # Flat model-top (for reference; doesn't depend on T except tiebreak)
        model_top = names[0]   # already ranked by score desc in the saved file
        flat_cost += TICKET
        if model_top == row["winner"]:
            flat_pay += row["payouts"].get("単勝", 0)
            flat_hits += 1

        # BEST value only: find the single largest positive edge
        best = None  # (edge, name, odds)
        per_race_bets = 0
        for name, score, od in zip(names, scores, odds):
            if od <= 1.0:
                continue
            if max_odds is not None and od > max_odds:
                continue
            mp = probs[names.index(name)]
            mkt = 1.0 / (od * OVERROUND)
            if mkt <= 0:
                continue
            edge = (mp - mkt) / mkt
            if edge < min_edge:
                continue
            per_race_bets += 1
            if best is None or edge > best[0]:
                best = (edge, name, od)
        value_win_bet_count_dist.append(per_race_bets)

        if best is not None:
            _, pick, pick_odds = best
            total_cost += TICKET
            n_bets += 1
            if pick_odds > 20.0:
                longshot_bets += 1
            if pick == row["winner"]:
                total_pay += row["payouts"].get("単勝", 0)
                winners += 1

    pnl = total_pay - total_cost
    roi = (pnl / total_cost) if total_cost else 0.0
    flat_pnl = flat_pay - flat_cost
    flat_roi = (flat_pnl / flat_cost) if flat_cost else 0.0

    return {
        "T": T,
        "max_odds": max_odds if max_odds is not None else float("inf"),
        "n_bets": n_bets,
        "cost": total_cost,
        "payout": total_pay,
        "pnl": pnl,
        "roi": roi,
        "winners": winners,
        "hit_rate": (winners / n_bets) if n_bets else 0.0,
        "longshot_bets": longshot_bets,
        "mean_top_prob": sum(top_probs) / len(top_probs),
        "mean_floor_prob": sum(floor_probs) / len(floor_probs),
        "mean_spread": sum(prob_spreads) / len(prob_spreads),
        "mean_bet_candidates_per_race": sum(value_win_bet_count_dist) / len(value_win_bet_count_dist),
        "flat_model_roi": flat_roi,
        "flat_model_hits": flat_hits,
    }


def fmt(r):
    return (f"T={r['T']:>4}  max_odds={r['max_odds']:>5}  "
            f"top={r['mean_top_prob']:.3f}  floor={r['mean_floor_prob']:.3f}  "
            f"spread={r['mean_spread']:.3f}  "
            f"cand/race={r['mean_bet_candidates_per_race']:4.1f}  "
            f"bets={r['n_bets']:>3}  "
            f"hit={r['hit_rate']*100:5.1f}%  "
            f"roi={r['roi']*100:+7.2f}%  "
            f"ls={r['longshot_bets']}")


def main():
    rows = load_dataset()
    print(f"Loaded {len(rows)} races\n")

    print("Distribution stats + BEST value only (edge ≥ 1%)")
    print("=" * 110)
    for T in (12, 8, 6, 5, 4):
        for max_odds in (None, 20.0):
            r = eval_temperature(rows, T=T, max_odds=max_odds, min_edge=0.01)
            print(fmt(r))
        print()

    # Reference: flat model win betting (T-independent since ranking doesn't change)
    r = eval_temperature(rows, T=12, max_odds=None, min_edge=0.01)
    print(f"Reference flat WIN model (T-independent): ROI {r['flat_model_roi']*100:+.2f}%  hits={r['flat_model_hits']}/50")

    # Also test tighter min_edge with sharp T
    print("\nFine sweep around best config (T=5, max_odds=20) with edge thresholds")
    print("-" * 80)
    for me in (0.00, 0.02, 0.05, 0.08, 0.12, 0.20):
        r = eval_temperature(rows, T=5, max_odds=20.0, min_edge=me)
        print(f"  min_edge={me*100:+4.0f}%  bets={r['n_bets']:>3}  "
              f"hit={r['hit_rate']*100:5.1f}%  roi={r['roi']*100:+7.2f}%  ls={r['longshot_bets']}")


if __name__ == "__main__":
    main()
