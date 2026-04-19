"""Stability KPIs: ROI variance, weekly ROI, max drawdown, losing streaks.

Computes for both v3 and v4:
  - Overall ROI
  - Weekly ROI (by ISO week)
  - ROI variance / stdev across weeks
  - Max drawdown (running cumulative pnl low)
  - Max consecutive losses

USAGE:
  python tools/stability_kpis.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_v2_v3 import _load_results, _winner, _norm, _win_pay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_DIR = PROJECT_ROOT / "data" / "backtest_predictions_v3"
V4_DIR = PROJECT_ROOT / "data" / "backtest_predictions"


def _load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_bets(pred_dir, results_map, label):
    """Return list of bets sorted by race_date."""
    bets = []
    for p in sorted(pred_dir.glob("*_on.json")):
        rid = p.stem.replace("_on", "")
        pred = _load(p)
        ranked = pred.get("ranked") or []
        if not ranked:
            continue
        result = results_map.get(rid, {})
        winner = _winner(result)
        if not winner:
            continue
        top1 = _norm(ranked[0].get("name"))
        win_pay = _win_pay(result)
        is_hit = top1 == winner
        pnl = (win_pay - 100) if is_hit else -100
        race_date = pred.get("race_date", "")
        try:
            d = dt.date.fromisoformat(race_date)
        except Exception:
            d = None

        bets.append({
            "race_id":   rid,
            "race_date": race_date,
            "race_name": pred.get("race_name", ""),
            "top1":      top1,
            "winner":    winner,
            "odds":      ranked[0].get("odds", 0),
            "is_hit":    is_hit,
            "pnl":       pnl,
            "win_pay":   win_pay if is_hit else 0,
            "date":      d,
        })
    bets.sort(key=lambda b: b["date"] or dt.date.min)
    return bets


def compute_kpis(bets):
    n = len(bets)
    if n == 0:
        return {}

    total_cost = n * 100
    total_payout = sum(b["pnl"] + 100 for b in bets)  # payout = pnl + cost
    roi = (total_payout - total_cost) / total_cost if total_cost else 0

    # Weekly ROI
    weekly = defaultdict(list)
    for b in bets:
        if b["date"]:
            iso = b["date"].isocalendar()
            key = f"{iso.year}-W{iso.week:02d}"
            weekly[key].append(b)
    weekly_roi = {}
    for wk, wk_bets in sorted(weekly.items()):
        cost = len(wk_bets) * 100
        payout = sum(b["pnl"] + 100 for b in wk_bets)
        weekly_roi[wk] = {
            "n_bets": len(wk_bets),
            "wins":   sum(1 for b in wk_bets if b["is_hit"]),
            "pnl":    sum(b["pnl"] for b in wk_bets),
            "roi":    ((payout - cost) / cost) if cost else 0,
        }

    # ROI variance across weeks
    weekly_rois = [w["roi"] for w in weekly_roi.values()]
    if len(weekly_rois) >= 2:
        mean_wk = sum(weekly_rois) / len(weekly_rois)
        var_wk = sum((r - mean_wk) ** 2 for r in weekly_rois) / len(weekly_rois)
        std_wk = var_wk ** 0.5
    else:
        mean_wk = weekly_rois[0] if weekly_rois else 0
        std_wk = 0

    # Running pnl & max drawdown
    cum = 0
    peak = 0
    max_dd = 0
    dd_start_idx = None
    dd_end_idx = None
    current_peak_idx = 0
    for i, b in enumerate(bets):
        cum += b["pnl"]
        if cum > peak:
            peak = cum
            current_peak_idx = i
        dd = peak - cum
        if dd > max_dd:
            max_dd = dd
            dd_start_idx = current_peak_idx
            dd_end_idx = i

    # Max consecutive losses
    max_losing_streak = 0
    cur_streak = 0
    for b in bets:
        if not b["is_hit"]:
            cur_streak += 1
            max_losing_streak = max(max_losing_streak, cur_streak)
        else:
            cur_streak = 0

    # Max consecutive wins (bonus)
    max_winning_streak = 0
    cur = 0
    for b in bets:
        if b["is_hit"]:
            cur += 1
            max_winning_streak = max(max_winning_streak, cur)
        else:
            cur = 0

    # Final cumulative
    final_pnl = sum(b["pnl"] for b in bets)
    n_wins = sum(1 for b in bets if b["is_hit"])

    return {
        "n_bets":    n,
        "n_wins":    n_wins,
        "win_rate":  n_wins / n,
        "roi":       roi,
        "final_pnl": final_pnl,
        "weekly": {
            "n_weeks":          len(weekly_roi),
            "mean_roi":         mean_wk,
            "std_roi":          std_wk,
            "min_week_roi":     min(weekly_rois) if weekly_rois else 0,
            "max_week_roi":     max(weekly_rois) if weekly_rois else 0,
            "detail":           weekly_roi,
        },
        "drawdown": {
            "max_drawdown_yen": max_dd,
            "dd_start_date":    bets[dd_start_idx]["race_date"] if dd_start_idx is not None else "",
            "dd_end_date":      bets[dd_end_idx]["race_date"] if dd_end_idx is not None else "",
        },
        "streaks": {
            "max_losing":  max_losing_streak,
            "max_winning": max_winning_streak,
        },
    }


def print_kpis(label, k):
    print(f"\n-- {label} --")
    print(f"  bets:       {k['n_bets']}")
    print(f"  wins:       {k['n_wins']}")
    print(f"  win_rate:   {k['win_rate']*100:.1f}%")
    print(f"  ROI:        {k['roi']*100:+.1f}%")
    print(f"  final_pnl:  {k['final_pnl']:+,}")
    print(f"  weekly mean_roi: {k['weekly']['mean_roi']*100:+.1f}%")
    print(f"  weekly std_roi:  {k['weekly']['std_roi']*100:.1f}pt")
    print(f"  weekly roi range: "
          f"[{k['weekly']['min_week_roi']*100:+.1f}%, "
          f"{k['weekly']['max_week_roi']*100:+.1f}%]")
    print(f"  n_weeks:    {k['weekly']['n_weeks']}")
    print(f"  max drawdown: {k['drawdown']['max_drawdown_yen']:,}円 "
          f"({k['drawdown']['dd_start_date']} → {k['drawdown']['dd_end_date']})")
    print(f"  max losing streak:  {k['streaks']['max_losing']}")
    print(f"  max winning streak: {k['streaks']['max_winning']}")


def main() -> int:
    results_map = _load_results()
    v3_bets = collect_bets(V3_DIR, results_map, "v3")
    v4_bets = collect_bets(V4_DIR, results_map, "v4")

    print("=" * 78)
    print("  STABILITY KPIs (v3 vs v4)")
    print("=" * 78)
    v3_k = compute_kpis(v3_bets)
    v4_k = compute_kpis(v4_bets)
    print_kpis("v3 (camp absolute)", v3_k)
    print_kpis("v4 (camp z-score)", v4_k)

    print("\n-- DELTAS --")
    print(f"  ROI:                  {(v4_k['roi']-v3_k['roi'])*100:+.1f}pt")
    print(f"  weekly std_roi:       {(v4_k['weekly']['std_roi']-v3_k['weekly']['std_roi'])*100:+.1f}pt "
          f"(lower=more stable)")
    print(f"  max drawdown:         {v4_k['drawdown']['max_drawdown_yen']-v3_k['drawdown']['max_drawdown_yen']:+,}円 "
          f"(lower=better)")
    print(f"  max losing streak:    {v4_k['streaks']['max_losing']-v3_k['streaks']['max_losing']:+d} "
          f"(lower=better)")

    out = PROJECT_ROOT / "data" / "stability_kpis_v3_v4.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"v3": v3_k, "v4": v4_k}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nReport saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
