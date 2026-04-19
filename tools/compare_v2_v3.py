"""Compare v2 (W_CAMP=0.015, no odds-band) vs v3 (W_CAMP=0.028, odds-band).

Reads predictions from:
  - data/backtest_predictions_v2/*_on.json   (v2 ON mode)
  - data/backtest_predictions/*_on.json      (v3 ON mode, current)

Joins with results.json and computes:
  - Win rate / ROI for v2 vs v3
  - Non-favorite ROI for v2 vs v3
  - Market follow rate for v2 vs v3
  - Top1 differences (v2 picked X, v3 picked Y)

Also compares v3 ON vs v3 OFF for completeness.

USAGE:
  python tools/compare_v2_v3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V2_DIR = PROJECT_ROOT / "data" / "backtest_predictions_v2"
V3_DIR = PROJECT_ROOT / "data" / "backtest_predictions"
RESULTS_FILE = PROJECT_ROOT / "data" / "results.json"


def _norm(s):
    return (s or "").strip()


def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        out[k[3:] if k.startswith("bt_") else k] = v
    return out


def _winner(result):
    fo = (result or {}).get("finishing_order") or []
    for h in fo:
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return _norm(h.get("name"))
        except ValueError:
            pass
    return ""


def _odds_favorite(result):
    fo = (result or {}).get("finishing_order") or []
    best = None
    for h in fo:
        try:
            od = float(str(h.get("odds", 0)).replace("---", "0"))
        except Exception:
            od = 0.0
        if od > 1.0 and (best is None or od < best[1]):
            best = (_norm(h.get("name")), od)
    return best[0] if best else ""


def _win_pay(result):
    pay = (result or {}).get("payouts") or {}
    raw = pay.get("\u5358\u52dd", 0)
    try:
        return float(str(raw).replace(",", "").replace("\u5186", "").strip() or 0)
    except Exception:
        return 0.0


def collect_metrics(pred_dir: Path, results_map: dict, label: str) -> dict:
    on_files = sorted(pred_dir.glob("*_on.json"))

    n_races = 0
    n_with_result = 0
    wins = 0
    cost = 0.0
    payout = 0.0
    market_follow = 0
    non_fav_count = 0
    non_fav_wins = 0
    non_fav_cost = 0.0
    non_fav_payout = 0.0
    races = []

    for p in on_files:
        rid = p.stem.replace("_on", "")
        with open(p, "r", encoding="utf-8") as f:
            pred = json.load(f)
        ranked = pred.get("ranked") or []
        if not ranked:
            continue
        n_races += 1
        top1 = _norm(ranked[0].get("name"))
        top1_odds = ranked[0].get("odds", 0)
        result = results_map.get(rid, {})
        winner = _winner(result)
        odds_fav = _odds_favorite(result)
        win_pay = _win_pay(result)

        races.append({
            "race_id": rid,
            "race_name": pred.get("race_name", ""),
            "top1": top1,
            "top1_odds": top1_odds,
            "winner": winner,
            "odds_fav": odds_fav,
            "win_pay": win_pay,
            "is_market_follow": top1 == odds_fav,
            "is_non_fav": top1 != odds_fav and bool(odds_fav),
        })

        if winner:
            n_with_result += 1
            cost += 100
            if top1 == winner:
                wins += 1
                payout += win_pay
            if top1 == odds_fav:
                market_follow += 1
            else:
                non_fav_count += 1
                non_fav_cost += 100
                if top1 == winner:
                    non_fav_wins += 1
                    non_fav_payout += win_pay

    win_rate = wins / n_with_result if n_with_result else 0
    roi = (payout - cost) / cost if cost else 0
    market_follow_rate = market_follow / n_with_result if n_with_result else 0
    nf_hit = non_fav_wins / non_fav_count if non_fav_count else 0
    nf_roi = (non_fav_payout - non_fav_cost) / non_fav_cost if non_fav_cost else 0

    return {
        "label": label,
        "n_races": n_races,
        "n_with_result": n_with_result,
        "wins": wins,
        "win_rate": win_rate,
        "cost": cost,
        "payout": payout,
        "roi": roi,
        "market_follow": market_follow,
        "market_follow_rate": market_follow_rate,
        "non_fav_count": non_fav_count,
        "non_fav_wins": non_fav_wins,
        "non_fav_hit_rate": nf_hit,
        "non_fav_roi": nf_roi,
        "races": races,
    }


def main() -> int:
    results_map = _load_results()
    if not V2_DIR.exists():
        print(f"[error] No v2 backup at {V2_DIR}")
        return 1
    if not V3_DIR.exists():
        print(f"[error] No v3 predictions at {V3_DIR}")
        return 1

    v2 = collect_metrics(V2_DIR, results_map, "v2 (W_CAMP=0.015, no dampener)")
    v3 = collect_metrics(V3_DIR, results_map, "v3 (W_CAMP=0.028, odds-band dampener)")

    # Top1 differences
    v2_top = {r["race_id"]: r for r in v2["races"]}
    v3_top = {r["race_id"]: r for r in v3["races"]}
    common_ids = sorted(set(v2_top.keys()) & set(v3_top.keys()))

    top1_diffs = []
    for rid in common_ids:
        r2, r3 = v2_top[rid], v3_top[rid]
        if r2["top1"] != r3["top1"]:
            top1_diffs.append({
                "race_id": rid,
                "race_name": r2["race_name"],
                "v2_top1": r2["top1"],
                "v3_top1": r3["top1"],
                "winner": r2["winner"],
                "v3_hit": r3["top1"] == r2["winner"],
                "v2_hit": r2["top1"] == r2["winner"],
                "win_pay": r2["win_pay"],
            })

    # Print
    print("=" * 78)
    print("  v2 vs v3 COMPARISON (same snapshots, different coefficients)")
    print("=" * 78)
    print(f"  v2: W_CAMP=0.015, no odds-band dampener")
    print(f"  v3: W_CAMP=0.028, pedigree×weight by odds band:")
    print(f"      odds<3 → 0.0,  3-15 → 1.0,  >15 → 0.5")
    print()

    print(f"-- Volume --")
    print(f"  v2 races={v2['n_races']}  with_result={v2['n_with_result']}")
    print(f"  v3 races={v3['n_races']}  with_result={v3['n_with_result']}")
    print()

    print(f"-- Win rate / ROI --")
    print(f"  v2: {v2['wins']:>3} wins  win_rate={v2['win_rate']*100:>5.1f}%  "
          f"ROI={v2['roi']*100:>+6.1f}%  pnl={v2['payout']-v2['cost']:>+8,.0f}")
    print(f"  v3: {v3['wins']:>3} wins  win_rate={v3['win_rate']*100:>5.1f}%  "
          f"ROI={v3['roi']*100:>+6.1f}%  pnl={v3['payout']-v3['cost']:>+8,.0f}")
    print(f"  Δ:  wins={v3['wins']-v2['wins']:+d}  "
          f"win_rate={(v3['win_rate']-v2['win_rate'])*100:+.1f}pt  "
          f"ROI={(v3['roi']-v2['roi'])*100:+.1f}pt")
    print()

    print(f"-- Market follow rate (1番人気追従) --")
    print(f"  v2: {v2['market_follow']}/{v2['n_with_result']} = "
          f"{v2['market_follow_rate']*100:.1f}%")
    print(f"  v3: {v3['market_follow']}/{v3['n_with_result']} = "
          f"{v3['market_follow_rate']*100:.1f}%")
    print(f"  Δ:  {(v3['market_follow_rate']-v2['market_follow_rate'])*100:+.1f}pt "
          f"(下がるほど独自性向上)")
    print()

    print(f"-- Non-favorite top1 (1番人気以外を本命) --")
    print(f"  v2: {v2['non_fav_count']:>3} races  "
          f"hit_rate={v2['non_fav_hit_rate']*100:>5.1f}%  "
          f"ROI={v2['non_fav_roi']*100:>+6.1f}%")
    print(f"  v3: {v3['non_fav_count']:>3} races  "
          f"hit_rate={v3['non_fav_hit_rate']*100:>5.1f}%  "
          f"ROI={v3['non_fav_roi']*100:>+6.1f}%")
    print(f"  Δ:  count={v3['non_fav_count']-v2['non_fav_count']:+d}  "
          f"hit_rate={(v3['non_fav_hit_rate']-v2['non_fav_hit_rate'])*100:+.1f}pt  "
          f"ROI={(v3['non_fav_roi']-v2['non_fav_roi'])*100:+.1f}pt")
    print()

    print(f"-- Top1 differences --")
    print(f"  Races where v2 top1 != v3 top1: {len(top1_diffs)} / {len(common_ids)}")
    if top1_diffs:
        new_hits = sum(1 for d in top1_diffs if d["v3_hit"])
        lost_hits = sum(1 for d in top1_diffs if d["v2_hit"] and not d["v3_hit"])
        print(f"  v3 newly hit: {new_hits}")
        print(f"  v3 lost (v2 was hit, v3 missed): {lost_hits}")
        print()
        for d in top1_diffs[:15]:
            v2mark = " v2WIN" if d["v2_hit"] else ""
            v3mark = " v3WIN" if d["v3_hit"] else ""
            print(f"    {d['race_name']:30}: {d['v2_top1']:15} → {d['v3_top1']:15}"
                  f" (winner={d['winner']}){v2mark}{v3mark}")

    # ── Success criteria check ──
    print()
    print("=" * 78)
    print("  SUCCESS CRITERIA")
    print("=" * 78)
    crit_roi = v3["roi"] > v2["roi"]
    crit_nf = v3["non_fav_roi"] >= v2["non_fav_roi"]
    crit_mf = v3["market_follow_rate"] < v2["market_follow_rate"]
    print(f"  [{('PASS' if crit_roi else 'FAIL')}] ROI improved: "
          f"v2 {v2['roi']*100:+.1f}% → v3 {v3['roi']*100:+.1f}%")
    print(f"  [{('PASS' if crit_nf else 'FAIL')}] Non-fav ROI not worsened: "
          f"v2 {v2['non_fav_roi']*100:+.1f}% → v3 {v3['non_fav_roi']*100:+.1f}%")
    print(f"  [{('PASS' if crit_mf else 'FAIL')}] Market follow decreased: "
          f"v2 {v2['market_follow_rate']*100:.1f}% → v3 {v3['market_follow_rate']*100:.1f}%")

    out = PROJECT_ROOT / "data" / "v2_v3_comparison.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "v2": {k: v for k, v in v2.items() if k != "races"},
            "v3": {k: v for k, v in v3.items() if k != "races"},
            "top1_diffs": top1_diffs,
            "success_criteria": {
                "roi_improved": crit_roi,
                "non_fav_roi_not_worsened": crit_nf,
                "market_follow_decreased": crit_mf,
            },
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
