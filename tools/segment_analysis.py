"""Segment v3 vs v4 by grade / distance / surface / field size.

Per segment, computes:
  - race count
  - v3 wins, v3 ROI
  - v4 wins, v4 ROI
  - delta (v4 - v3)

Requires:
  data/backtest_predictions_v3/  (v3 predictions)
  data/backtest_predictions/     (v4 current predictions)
  data/results.json

USAGE:
  python tools/segment_analysis.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_v2_v3 import _load_results, _winner, _odds_favorite, _win_pay, _norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_DIR = PROJECT_ROOT / "data" / "backtest_predictions_v3"
V4_DIR = PROJECT_ROOT / "data" / "backtest_predictions"


def _load_pred(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _distance_band(d):
    if not d or d <= 0:
        return "unknown"
    if d < 1600:
        return "short (<1600)"
    if d < 2200:
        return "middle (1600-2200)"
    return "long (>=2200)"


def _surface(info):
    s = info.get("surface", "") or ""
    if "\u82dd" in s:
        return "turf"
    if "\u30c0" in s:
        return "dirt"
    return "unknown"


def _parse_surface_distance(info):
    # race_info has "surface": "芝2400m" or similar
    s = info.get("surface", "")
    import re
    m = re.search(r"(\u82dd|\u30c0\u30fc\u30c8|\u30c0)([\d,]+)", s)
    if m:
        surf = "turf" if "\u82dd" in m.group(1) else "dirt"
        try:
            dist = int(m.group(2).replace(",", ""))
        except ValueError:
            dist = 0
        return surf, dist
    return "unknown", 0


def _field_band(n):
    if n < 12:
        return "small (<12)"
    if n <= 16:
        return "middle (12-16)"
    return "large (>16)"


def collect(results_map):
    """Join v3 + v4 predictions with results + race meta."""
    races = []
    v3_files = {p.stem.replace("_on", ""): p
                for p in V3_DIR.glob("*_on.json")}
    v4_files = {p.stem.replace("_on", ""): p
                for p in V4_DIR.glob("*_on.json")}
    common = sorted(set(v3_files) & set(v4_files))

    # Load race_meta for surface/distance
    meta_dir = PROJECT_ROOT / "data" / "race_meta"

    for rid in common:
        v3 = _load_pred(v3_files[rid])
        v4 = _load_pred(v4_files[rid])
        result = results_map.get(rid, {})
        winner = _winner(result)
        odds_fav = _odds_favorite(result)
        win_pay = _win_pay(result)

        meta_p = meta_dir / f"{rid}.json"
        surf = "unknown"
        dist = 0
        if meta_p.exists():
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            surf = meta.get("surface", "unknown")
            dist = meta.get("distance", 0)

        grade = v4.get("grade", "")
        num_horses = len(v4.get("ranked", []))

        v3_top1 = _norm((v3.get("ranked") or [{}])[0].get("name"))
        v4_top1 = _norm((v4.get("ranked") or [{}])[0].get("name"))

        races.append({
            "race_id":   rid,
            "race_name": v4.get("race_name", ""),
            "race_date": v4.get("race_date", ""),
            "grade":     grade or "other",
            "surface":   surf,
            "distance":  dist,
            "dist_band": _distance_band(dist),
            "field_band": _field_band(num_horses),
            "num_horses": num_horses,
            "winner":    winner,
            "odds_fav":  odds_fav,
            "win_pay":   win_pay,
            "v3_top1":   v3_top1,
            "v4_top1":   v4_top1,
            "v3_hit":    bool(winner) and v3_top1 == winner,
            "v4_hit":    bool(winner) and v4_top1 == winner,
        })

    return races


def segment_metrics(races, key_fn):
    groups = defaultdict(list)
    for r in races:
        groups[key_fn(r)].append(r)

    out = {}
    for seg, rs in sorted(groups.items()):
        n = len(rs)
        with_res = [r for r in rs if r["winner"]]
        n_res = len(with_res)

        v3_wins = sum(1 for r in with_res if r["v3_hit"])
        v4_wins = sum(1 for r in with_res if r["v4_hit"])
        v3_cost = n_res * 100.0
        v3_payout = sum(r["win_pay"] for r in with_res if r["v3_hit"])
        v4_cost = n_res * 100.0
        v4_payout = sum(r["win_pay"] for r in with_res if r["v4_hit"])

        def _roi(cost, pay):
            return (pay - cost) / cost if cost > 0 else 0.0

        out[seg] = {
            "n":         n,
            "n_result":  n_res,
            "v3_wins":   v3_wins,
            "v4_wins":   v4_wins,
            "v3_win_rate": v3_wins / n_res if n_res else 0,
            "v4_win_rate": v4_wins / n_res if n_res else 0,
            "v3_roi":    _roi(v3_cost, v3_payout),
            "v4_roi":    _roi(v4_cost, v4_payout),
            "delta_wins": v4_wins - v3_wins,
            "delta_roi": _roi(v4_cost, v4_payout) - _roi(v3_cost, v3_payout),
        }
    return out


def print_segment(title, metrics):
    print(f"\n-- {title} --")
    hdr = f"  {'segment':<20} {'n':>4} {'v3_wins':>8} {'v4_wins':>8} {'v3_roi':>9} {'v4_roi':>9} {'ΔROI':>9}"
    print(hdr)
    for seg, m in metrics.items():
        print(f"  {seg:<20} {m['n']:>4} {m['v3_wins']:>8} {m['v4_wins']:>8} "
              f"{m['v3_roi']*100:>+8.1f}% {m['v4_roi']*100:>+8.1f}% {m['delta_roi']*100:>+8.1f}pt")


def main() -> int:
    results_map = _load_results()
    races = collect(results_map)
    print(f"Total races loaded: {len(races)}")

    grade_m = segment_metrics(races, lambda r: r["grade"])
    dist_m = segment_metrics(races, lambda r: r["dist_band"])
    surf_m = segment_metrics(races, lambda r: r["surface"])
    field_m = segment_metrics(races, lambda r: r["field_band"])

    print("\n" + "=" * 78)
    print("  SEGMENT ANALYSIS — v3 vs v4 (camp absolute vs camp z-score)")
    print("=" * 78)
    print_segment("By Grade", grade_m)
    print_segment("By Distance", dist_m)
    print_segment("By Surface", surf_m)
    print_segment("By Field Size", field_m)

    # Save
    out = PROJECT_ROOT / "data" / "segment_analysis_v3_v4.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "total_races": len(races),
            "by_grade":   grade_m,
            "by_distance": dist_m,
            "by_surface": surf_m,
            "by_field_size": field_m,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
