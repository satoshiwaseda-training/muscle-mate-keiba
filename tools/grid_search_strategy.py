"""Grid search: per-grade best strategy for 単勝×3 + 馬連×3 (600円/R).

For each grade (G1 / G2 / G3) we evaluate:
  - "win_prob"                   (baseline: ranked[:3])
  - "diversified_1-3_4-7_8+"     (market bucket, current G2 config)
  - "tight_1-2_3-5_6+"
  - "loose_1-4_5-9_10+"
  - "mid_heavy_1-2_3-6_7+"
  - "wide_穴_1-3_4-8_9+"

For each (grade, strategy) pair we compute:
  - ROI (cost vs payout)
  - 単勝 hit rate / 馬連 hit rate
  - PnL
  - Robustness: ROI if top 2 big wins removed (overfitting check)

Report best strategy per grade.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import grade_strategy as gs

RESULTS_FILE = ROOT / "data" / "results.json"
BACKTEST_DIR_V5 = ROOT / "data" / "backtest_predictions"
BACKTEST_DIR_V4 = ROOT / "data" / "backtest_predictions_v4_n221"
BACKTEST_DIR = BACKTEST_DIR_V5 if BACKTEST_DIR_V5.exists() else BACKTEST_DIR_V4

COST_PER_RACE = 600.0


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


def _grade_bucket(grade_str: str) -> str:
    g = (grade_str or "").upper()
    if "G1" in g or "GI" == g:
        return "G1"
    if "G2" in g or "GII" == g:
        return "G2"
    if "G3" in g or "GIII" == g:
        return "G3"
    return "OTHER"


def simulate(pred: dict, result: dict, strategy: str) -> dict | None:
    ranked = pred.get("ranked") or []
    if len(ranked) < 3:
        return None
    rank_map = _rank_map(result)
    if not rank_map:
        return None

    if strategy == "win_prob":
        top3_names = [(r.get("name") or "").strip() for r in ranked[:3]]
    else:
        mk_map = gs.build_market_rank_map(ranked)
        picked = gs.pick_diversified_top3(ranked, mk_map, strategy=strategy)
        top3_names = [(h.get("name") or "").strip() for h in picked[:3]]

    if len([n for n in top3_names if n]) < 3:
        return None
    top3_set = set(top3_names)

    winner = next((nm for nm, r in rank_map.items() if r == 1), None)
    second = next((nm for nm, r in rank_map.items() if r == 2), None)
    if not winner:
        return None

    tansho = _payout(result, "単勝") if winner in top3_set else 0.0
    umaren = 0.0
    if second and frozenset([winner, second]) in {
        frozenset(c) for c in combinations(top3_names, 2)
    }:
        umaren = _payout(result, "馬連")

    return {
        "race_name": pred.get("race_name", ""),
        "grade": pred.get("grade", ""),
        "tansho": tansho,
        "umaren": umaren,
        "payout": tansho + umaren,
        "pnl":    (tansho + umaren) - COST_PER_RACE,
        "top3":   top3_names,
    }


def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    n = len(records)
    cost = n * COST_PER_RACE
    payout = sum(r["payout"] for r in records)
    pnl = payout - cost

    # Robustness: sort by pnl desc, remove top 2 wins
    sorted_by_pnl = sorted(records, key=lambda r: r["pnl"], reverse=True)
    top2_pnl = sum(r["pnl"] for r in sorted_by_pnl[:2])
    if n > 2:
        ex_big = {
            "cost":   (n - 2) * COST_PER_RACE,
            "payout": sum(r["payout"] for r in sorted_by_pnl[2:]),
            "pnl":    pnl - top2_pnl,
        }
        ex_big["roi"] = ex_big["pnl"] / ex_big["cost"] if ex_big["cost"] else 0.0
    else:
        ex_big = None

    return {
        "n":      n,
        "cost":   cost,
        "payout": payout,
        "pnl":    pnl,
        "roi":    pnl / cost if cost else 0.0,
        "tansho_hit": sum(1 for r in records if r["tansho"] > 0) / n,
        "umaren_hit": sum(1 for r in records if r["umaren"] > 0) / n,
        "ex_big2":    ex_big,      # ROI after removing top-2 big wins
    }


def main() -> int:
    results = _load_results()
    all_preds = []
    for p in sorted(BACKTEST_DIR.glob("*_on.json")):
        try:
            all_preds.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    print(f"[grid] predictions loaded: {len(all_preds)}")

    # Group by grade bucket
    by_grade: dict = {"G1": [], "G2": [], "G3": []}
    for p in all_preds:
        b = _grade_bucket(p.get("grade", ""))
        if b in by_grade:
            by_grade[b].append(p)
    for g, lst in by_grade.items():
        print(f"  {g}: {len(lst)} races")

    strategies = ["win_prob"] + list(gs.STRATEGIES.keys())
    print(f"\n[grid] strategies tested: {len(strategies)}")
    for s in strategies:
        print(f"  - {s}")

    # Run
    results_table: dict[str, dict[str, dict]] = {}
    for grade, preds in by_grade.items():
        results_table[grade] = {}
        for strat in strategies:
            recs = []
            for p in preds:
                res = results.get(p.get("race_id", ""))
                if not res:
                    continue
                r = simulate(p, res, strat)
                if r:
                    recs.append(r)
            agg = aggregate(recs)
            results_table[grade][strat] = agg

    # Print results table
    print("\n" + "=" * 96)
    print(f"{'Grade':<6s}{'Strategy':<30s}{'n':>4s}{'ROI%':>10s}{'単勝':>8s}"
          f"{'馬連':>8s}{'PnL':>10s}{'ROI ex-big2':>14s}")
    print("=" * 96)
    best_per_grade = {}
    for grade in ("G1", "G2", "G3"):
        grade_results = results_table[grade]
        # Find best by ROI
        best = max(grade_results.items(),
                   key=lambda kv: kv[1].get("roi", -99))
        best_per_grade[grade] = best[0]
        for strat in strategies:
            a = grade_results.get(strat, {})
            if a.get("n", 0) == 0:
                continue
            marker = " ★" if strat == best[0] else ""
            ex_big = a.get("ex_big2") or {}
            print(f"{grade:<6s}{strat:<30s}"
                  f"{a['n']:>4d}"
                  f"{a['roi']*100:>+9.1f}%"
                  f"{a['tansho_hit']*100:>7.1f}%"
                  f"{a['umaren_hit']*100:>7.1f}%"
                  f"{a['pnl']:>+10,.0f}"
                  f"{ex_big.get('roi', 0)*100:>+13.1f}%"
                  f"{marker}")
        print()

    print("=" * 96)
    print("BEST STRATEGY PER GRADE")
    print("=" * 96)
    for g, s in best_per_grade.items():
        a = results_table[g][s]
        print(f"  {g}: {s}")
        print(f"      ROI = {a['roi']*100:+.1f}%  (n={a['n']}, "
              f"PnL={a['pnl']:+,.0f}円)")
        ex = a.get("ex_big2")
        if ex:
            print(f"      ROI ex-big2 = {ex['roi']*100:+.1f}%  (robustness check)")
        print(f"      hit rates: 単勝 {a['tansho_hit']*100:.1f}% · "
              f"馬連 {a['umaren_hit']*100:.1f}%")

    # Config snippet for copy-paste
    print("\n" + "=" * 96)
    print("GRADE_STRATEGY mapping (copy to grade_strategy.py)")
    print("=" * 96)
    print("GRADE_STRATEGY = {")
    for g in ("G1", "G2", "G3"):
        s = best_per_grade.get(g, "win_prob")
        print(f'    "{g}":     {s!r},')
    # Jpn variants inherit G2/G3
    print(f'    "JpnI":   {best_per_grade.get("G1", "win_prob")!r},')
    print(f'    "JpnII":  {best_per_grade.get("G2", "win_prob")!r},')
    print(f'    "JpnIII": {best_per_grade.get("G3", "win_prob")!r},')
    print("}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
