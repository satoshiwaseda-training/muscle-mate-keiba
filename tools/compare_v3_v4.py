"""Compare v3 (camp absolute) vs v4 (camp z-score race-relative).

USAGE:
  python tools/compare_v3_v4.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_v2_v3 import collect_metrics, _load_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_DIR = PROJECT_ROOT / "data" / "backtest_predictions_v3"
V4_DIR = PROJECT_ROOT / "data" / "backtest_predictions"


def main() -> int:
    results_map = _load_results()
    if not V3_DIR.exists():
        print(f"[error] No v3 backup at {V3_DIR}")
        return 1

    v3 = collect_metrics(V3_DIR, results_map, "v3 (camp absolute)")
    v4 = collect_metrics(V4_DIR, results_map, "v4 (camp z-score)")

    v3_top = {r["race_id"]: r for r in v3["races"]}
    v4_top = {r["race_id"]: r for r in v4["races"]}
    common_ids = sorted(set(v3_top.keys()) & set(v4_top.keys()))

    top1_diffs = []
    for rid in common_ids:
        r3, r4 = v3_top[rid], v4_top[rid]
        if r3["top1"] != r4["top1"]:
            top1_diffs.append({
                "race_id": rid,
                "race_name": r3["race_name"],
                "v3_top1": r3["top1"],
                "v4_top1": r4["top1"],
                "winner": r3["winner"],
                "v3_hit": r3["top1"] == r3["winner"],
                "v4_hit": r4["top1"] == r3["winner"],
                "win_pay": r3["win_pay"],
            })

    print("=" * 78)
    print("  v3 vs v4 COMPARISON — camp z-score normalization effect")
    print("=" * 78)
    print(f"  v3: camp uses absolute (n_camp - 0.5)")
    print(f"  v4: camp uses race-relative z-score (camp - mean) / std")
    print()

    print(f"-- Volume --")
    print(f"  v3 races={v3['n_races']}  v4 races={v4['n_races']}")
    print()

    print(f"-- Win rate / ROI --")
    print(f"  v3: {v3['wins']:>3} wins  win_rate={v3['win_rate']*100:>5.1f}%  "
          f"ROI={v3['roi']*100:>+6.1f}%  pnl={v3['payout']-v3['cost']:>+8,.0f}")
    print(f"  v4: {v4['wins']:>3} wins  win_rate={v4['win_rate']*100:>5.1f}%  "
          f"ROI={v4['roi']*100:>+6.1f}%  pnl={v4['payout']-v4['cost']:>+8,.0f}")
    print(f"  Δ:  wins={v4['wins']-v3['wins']:+d}  "
          f"win_rate={(v4['win_rate']-v3['win_rate'])*100:+.1f}pt  "
          f"ROI={(v4['roi']-v3['roi'])*100:+.1f}pt")
    print()

    print(f"-- Market follow rate --")
    print(f"  v3: {v3['market_follow']}/{v3['n_with_result']} = "
          f"{v3['market_follow_rate']*100:.1f}%")
    print(f"  v4: {v4['market_follow']}/{v4['n_with_result']} = "
          f"{v4['market_follow_rate']*100:.1f}%")
    print(f"  Δ:  {(v4['market_follow_rate']-v3['market_follow_rate'])*100:+.1f}pt")
    print()

    print(f"-- Non-favorite top1 --")
    print(f"  v3: {v3['non_fav_count']:>3} races  "
          f"hit_rate={v3['non_fav_hit_rate']*100:>5.1f}%  "
          f"ROI={v3['non_fav_roi']*100:>+6.1f}%")
    print(f"  v4: {v4['non_fav_count']:>3} races  "
          f"hit_rate={v4['non_fav_hit_rate']*100:>5.1f}%  "
          f"ROI={v4['non_fav_roi']*100:>+6.1f}%")
    print(f"  Δ:  count={v4['non_fav_count']-v3['non_fav_count']:+d}  "
          f"hit_rate={(v4['non_fav_hit_rate']-v3['non_fav_hit_rate'])*100:+.1f}pt  "
          f"ROI={(v4['non_fav_roi']-v3['non_fav_roi'])*100:+.1f}pt")
    print()

    print(f"-- Top1 differences (v3 → v4) --")
    n_diff = len(top1_diffs)
    diff_pct = n_diff / len(common_ids) * 100 if common_ids else 0
    print(f"  {n_diff} / {len(common_ids)} races ({diff_pct:.1f}%)")
    if top1_diffs:
        new_hits = sum(1 for d in top1_diffs if d["v4_hit"])
        lost_hits = sum(1 for d in top1_diffs if d["v3_hit"] and not d["v4_hit"])
        gained_pnl = sum((d["win_pay"] - 100) for d in top1_diffs if d["v4_hit"])
        lost_pnl = sum((d["win_pay"] - 100) for d in top1_diffs if d["v3_hit"] and not d["v4_hit"])
        print(f"  v4 newly hit: {new_hits} (gained pnl={gained_pnl:+,.0f})")
        print(f"  v4 lost (v3 was hit, v4 missed): {lost_hits} (lost pnl={lost_pnl:+,.0f})")
        print()
        for d in top1_diffs[:20]:
            v3mark = " v3WIN" if d["v3_hit"] else ""
            v4mark = " v4WIN" if d["v4_hit"] else ""
            print(f"    {d['race_name']:30}: {d['v3_top1']:18} → {d['v4_top1']:18}"
                  f" (winner={d['winner']}){v3mark}{v4mark}")

    print()
    print("=" * 78)
    print("  SUCCESS CRITERIA")
    print("=" * 78)
    crit_top1 = diff_pct > 5.0
    crit_roi = v4["roi"] > v3["roi"]
    crit_nf = v4["non_fav_roi"] >= v3["non_fav_roi"]
    print(f"  [{('PASS' if crit_top1 else 'FAIL')}] Top1 changes > 5%: "
          f"{n_diff}/{len(common_ids)} = {diff_pct:.1f}%")
    print(f"  [{('PASS' if crit_roi else 'NEUTRAL' if v4['roi']==v3['roi'] else 'FAIL')}] ROI improved: "
          f"v3 {v3['roi']*100:+.1f}% → v4 {v4['roi']*100:+.1f}%")
    print(f"  [{('PASS' if crit_nf else 'FAIL')}] Non-fav ROI not worsened: "
          f"v3 {v3['non_fav_roi']*100:+.1f}% → v4 {v4['non_fav_roi']*100:+.1f}%")

    out = PROJECT_ROOT / "data" / "v3_v4_comparison.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "v3": {k: v for k, v in v3.items() if k != "races"},
            "v4": {k: v for k, v in v4.items() if k != "races"},
            "top1_diffs": top1_diffs,
            "success_criteria": {
                "top1_changes_over_5pct": crit_top1,
                "roi_improved": crit_roi,
                "non_fav_roi_not_worsened": crit_nf,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
