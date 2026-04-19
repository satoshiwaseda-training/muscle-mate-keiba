"""Snapshot-based v1 vs v2 comparison.

Reads `{race_id}_on.json` and `{race_id}_off.json` from
data/backtest_predictions/ and compares them:

- Top1 change rate
- Rank changes in top10
- Market follow rate (both)
- Non-favorite picks (both)
- Overall win rate / ROI (if results available)
- Pedigree composite contribution

USAGE:
  python tools/snapshot_compare.py
  python tools/snapshot_compare.py --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "data" / "backtest_predictions"
RESULTS_FILE = PROJECT_ROOT / "data" / "results.json"


def _norm(s):
    return (s or "").strip()


def _load_results() -> dict:
    """Load results.json and return a dict keyed by race_id.

    results.json keys are prefixed with 'bt_' historically; also
    live_predictions has raw race_id keys. Handle both.
    """
    if not RESULTS_FILE.exists():
        return {}
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    out = {}
    for k, v in raw.items():
        # Strip bt_ prefix if present
        if k.startswith("bt_"):
            out[k[3:]] = v
        else:
            out[k] = v
    return out


def _winner_of(result: dict) -> str | None:
    fo = (result or {}).get("finishing_order") or []
    for h in fo:
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return _norm(h.get("name"))
        except ValueError:
            pass
    return None


def _odds_favorite(result: dict) -> str | None:
    fo = (result or {}).get("finishing_order") or []
    best = None
    for h in fo:
        try:
            od = float(str(h.get("odds", 0)).replace("---", "0"))
        except Exception:
            od = 0.0
        if od > 1.0 and (best is None or od < best[1]):
            best = (_norm(h.get("name")), od)
    return best[0] if best else None


def _win_pay(result: dict) -> float:
    pay = (result or {}).get("payouts") or {}
    raw = pay.get("\u5358\u52dd", 0)
    try:
        return float(str(raw).replace(",", "").replace("\u5186", "").strip() or 0)
    except Exception:
        return 0.0


def compare_one(on: dict, off: dict, result: dict | None) -> dict:
    """Compare pedigree-on vs pedigree-off for a single race."""
    on_ranked = on.get("ranked") or []
    off_ranked = off.get("ranked") or []
    if not on_ranked or not off_ranked:
        return {}

    on_top1 = _norm(on_ranked[0].get("name"))
    off_top1 = _norm(off_ranked[0].get("name"))

    # Rank changes (top10)
    off_order = [_norm(h.get("name")) for h in off_ranked[:10]]
    on_order = [_norm(h.get("name")) for h in on_ranked[:10]]

    changes = []
    for i, name in enumerate(on_order):
        if name in off_order:
            off_pos = off_order.index(name)
            if off_pos != i:
                changes.append({
                    "name": name,
                    "off_rank": off_pos + 1,
                    "on_rank": i + 1,
                    "delta": off_pos - i,
                })

    winner = _winner_of(result) if result else None
    odds_fav = _odds_favorite(result) if result else None
    win_pay = _win_pay(result) if result else 0.0

    detail = {
        "race_id":    on.get("race_id", ""),
        "race_name":  on.get("race_name", ""),
        "race_date":  on.get("race_date", ""),
        "off_top1":   off_top1,
        "on_top1":    on_top1,
        "top1_changed": off_top1 != on_top1,
        "odds_fav":   odds_fav or "",
        "winner":     winner or "",
        "win_pay":    win_pay,
        "has_result": winner is not None,
        "rank_changes_top10": len(changes),
        "changes":    changes,
        "on_off_market_follow": {
            "off": off_top1 == odds_fav if odds_fav else None,
            "on":  on_top1 == odds_fav if odds_fav else None,
        },
        "on_top1_pedigree": {
            "pedigree_composite": on_ranked[0].get("pedigree_composite", 0.5),
            "camp_composite":     on_ranked[0].get("camp_composite", 0.5),
            "sire_distance_fit":  on_ranked[0].get("sire_distance_fit", 0.5),
            "sire_name":          on_ranked[0].get("sire_name", ""),
            "damsire_name":       on_ranked[0].get("damsire_name", ""),
            "breeder_name":       on_ranked[0].get("breeder_name", ""),
        },
    }
    return detail


def aggregate(details: list[dict]) -> dict:
    total = len(details)
    with_result = sum(1 for d in details if d.get("has_result"))

    top1_changes = sum(1 for d in details if d.get("top1_changed"))
    total_rank_changes = sum(d.get("rank_changes_top10", 0) for d in details)

    # Market follow
    mf_off = sum(1 for d in details
                 if d.get("on_off_market_follow", {}).get("off") is True)
    mf_on = sum(1 for d in details
                if d.get("on_off_market_follow", {}).get("on") is True)
    # Denominator: races where odds_fav is known
    mf_denom = sum(1 for d in details if d.get("odds_fav"))

    # Non-favorite picks
    nf_off_count = 0
    nf_on_count = 0
    nf_off_wins = 0
    nf_on_wins = 0
    nf_off_cost = 0.0
    nf_on_cost = 0.0
    nf_off_payout = 0.0
    nf_on_payout = 0.0

    # Overall win rate / ROI
    off_wins = 0
    on_wins = 0
    off_cost = 0.0
    on_cost = 0.0
    off_payout = 0.0
    on_payout = 0.0

    for d in details:
        if not d.get("has_result"):
            continue
        winner = d["winner"]
        odds_fav = d.get("odds_fav", "")
        win_pay = d.get("win_pay", 0.0)

        # Overall
        off_cost += 100
        on_cost += 100
        if d["off_top1"] == winner:
            off_wins += 1
            off_payout += win_pay
        if d["on_top1"] == winner:
            on_wins += 1
            on_payout += win_pay

        # Non-favorite
        if odds_fav and d["off_top1"] != odds_fav:
            nf_off_count += 1
            nf_off_cost += 100
            if d["off_top1"] == winner:
                nf_off_wins += 1
                nf_off_payout += win_pay
        if odds_fav and d["on_top1"] != odds_fav:
            nf_on_count += 1
            nf_on_cost += 100
            if d["on_top1"] == winner:
                nf_on_wins += 1
                nf_on_payout += win_pay

    def rate(a, b):
        return (a / b) if b > 0 else 0.0

    def roi(cost, payout):
        return ((payout - cost) / cost) if cost > 0 else 0.0

    return {
        "total_races":        total,
        "races_with_result":  with_result,
        "top1_changed":       top1_changes,
        "top1_change_rate":   rate(top1_changes, total),
        "total_rank_changes": total_rank_changes,
        "market_follow": {
            "denom":      mf_denom,
            "off_count":  mf_off,
            "on_count":   mf_on,
            "off_rate":   rate(mf_off, mf_denom),
            "on_rate":    rate(mf_on, mf_denom),
            "diff_pt":    rate(mf_on, mf_denom) - rate(mf_off, mf_denom),
        },
        "non_favorite": {
            "off_count":  nf_off_count,
            "on_count":   nf_on_count,
            "off_hit_rate": rate(nf_off_wins, nf_off_count),
            "on_hit_rate":  rate(nf_on_wins, nf_on_count),
            "off_roi":      roi(nf_off_cost, nf_off_payout),
            "on_roi":       roi(nf_on_cost, nf_on_payout),
        },
        "overall": {
            "off_win_rate": rate(off_wins, with_result),
            "on_win_rate":  rate(on_wins, with_result),
            "off_roi":      roi(off_cost, off_payout),
            "on_roi":       roi(on_cost, on_payout),
            "off_wins":     off_wins,
            "on_wins":      on_wins,
        },
    }


def print_report(agg: dict, details: list[dict], show_changes: int = 10):
    print("=" * 70)
    print("  SNAPSHOT-BASED v1 vs v2 COMPARISON (pedigree off vs on)")
    print("=" * 70)
    print(f"  Total races:          {agg['total_races']}")
    print(f"  Races with result:    {agg['races_with_result']}")
    print(f"  Top1 changed:         {agg['top1_changed']}/{agg['total_races']} "
          f"({agg['top1_change_rate']:.1%})")
    print(f"  Total rank changes:   {agg['total_rank_changes']}")
    print()
    mf = agg["market_follow"]
    print("-- Market Follow (1番人気追従) --")
    print(f"  OFF: {mf['off_count']}/{mf['denom']} = {mf['off_rate']:.1%}")
    print(f"  ON:  {mf['on_count']}/{mf['denom']} = {mf['on_rate']:.1%}")
    print(f"  diff: {mf['diff_pt']:+.1%}")
    print()
    nf = agg["non_favorite"]
    print("-- Non-Favorite Top1 (1番人気以外を本命) --")
    print(f"  OFF: {nf['off_count']} races, hit_rate={nf['off_hit_rate']:.1%}, ROI={nf['off_roi']:+.1%}")
    print(f"  ON:  {nf['on_count']} races, hit_rate={nf['on_hit_rate']:.1%}, ROI={nf['on_roi']:+.1%}")
    print()
    ov = agg["overall"]
    print("-- Overall Win Rate / ROI --")
    print(f"  OFF: {ov['off_wins']} wins  win_rate={ov['off_win_rate']:.1%}  ROI={ov['off_roi']:+.1%}")
    print(f"  ON:  {ov['on_wins']} wins  win_rate={ov['on_win_rate']:.1%}  ROI={ov['on_roi']:+.1%}")
    print()

    # Show races where top1 actually changed
    changed = [d for d in details if d.get("top1_changed")]
    if changed:
        print(f"-- Races where top1 changed ({len(changed)}) --")
        for d in changed[:show_changes]:
            mark_fav = " (=fav)" if d["odds_fav"] == d["on_top1"] else ""
            mark_win = " WIN" if d["winner"] == d["on_top1"] else \
                       (" (winner=" + d["winner"] + ")" if d["winner"] else "")
            print(f"  {d['race_date']} {d['race_name']}")
            print(f"    off: {d['off_top1']}")
            print(f"    on:  {d['on_top1']}{mark_fav}{mark_win}")
            ped = d["on_top1_pedigree"]
            print(f"    (ped={ped['pedigree_composite']:.2f} "
                  f"camp={ped['camp_composite']:.2f} "
                  f"dist={ped['sire_distance_fit']:.2f} "
                  f"sire={ped['sire_name']} breeder={ped['breeder_name']})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", default="data/snapshot_comparison_report.json")
    args = parser.parse_args()

    # Find all race_ids that have both _on and _off predictions
    on_files = {p.stem.replace("_on", ""): p
                for p in PRED_DIR.glob("*_on.json")}
    off_files = {p.stem.replace("_off", ""): p
                 for p in PRED_DIR.glob("*_off.json")}
    common_ids = sorted(set(on_files.keys()) & set(off_files.keys()))

    if args.limit > 0:
        common_ids = common_ids[:args.limit]

    if not common_ids:
        print("[error] No matching on/off prediction pairs found in "
              f"{PRED_DIR}. Run snapshot_predict.py first.")
        return 1

    results_map = _load_results()

    details = []
    for rid in common_ids:
        with open(on_files[rid], "r", encoding="utf-8") as f:
            on = json.load(f)
        with open(off_files[rid], "r", encoding="utf-8") as f:
            off = json.load(f)
        result = results_map.get(rid, {})
        d = compare_one(on, off, result)
        if d:
            details.append(d)

    agg = aggregate(details)
    print_report(agg, details)

    # Save JSON
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"aggregate": agg, "details": details},
                  f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
