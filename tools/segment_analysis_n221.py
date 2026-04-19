"""Segment analysis: v4 ON vs OFF on full n=221 backtest.

Segments by grade, distance, surface, field size.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_v2_v3 import _load_results, _winner, _norm, _win_pay

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "data" / "backtest_predictions"
META_DIR = PROJECT_ROOT / "data" / "race_meta"


def _dist_band(d):
    if d <= 0:  return "unknown"
    if d < 1600: return "short (<1600)"
    if d < 2200: return "middle (1600-2200)"
    return "long (>=2200)"


def _field_band(n):
    if n < 12:   return "small (<12)"
    if n <= 16:  return "middle (12-16)"
    return "large (>16)"


def collect():
    results_map = _load_results()
    on_files = {p.stem.replace("_on", ""): p for p in PRED_DIR.glob("*_on.json")}
    off_files = {p.stem.replace("_off", ""): p for p in PRED_DIR.glob("*_off.json")}
    common = sorted(set(on_files) & set(off_files))

    races = []
    for rid in common:
        with open(on_files[rid], "r", encoding="utf-8") as f:
            on = json.load(f)
        with open(off_files[rid], "r", encoding="utf-8") as f:
            off = json.load(f)
        result = results_map.get(rid, {})
        winner = _winner(result)
        win_pay = _win_pay(result)
        if not winner:
            continue

        on_top1 = _norm((on.get("ranked") or [{}])[0].get("name"))
        off_top1 = _norm((off.get("ranked") or [{}])[0].get("name"))
        num_horses = len(on.get("ranked", []))
        grade = on.get("grade", "") or "other"

        meta_p = META_DIR / f"{rid}.json"
        surf = "unknown"; dist = 0
        if meta_p.exists():
            with open(meta_p, "r", encoding="utf-8") as f:
                m = json.load(f)
            surf = m.get("surface", "unknown")
            dist = m.get("distance", 0)

        races.append({
            "race_id": rid,
            "race_date": on.get("race_date", ""),
            "grade": grade,
            "surface": surf,
            "dist_band": _dist_band(dist),
            "field_band": _field_band(num_horses),
            "winner": winner,
            "win_pay": win_pay,
            "off_top1": off_top1,
            "on_top1": on_top1,
            "off_hit": off_top1 == winner,
            "on_hit": on_top1 == winner,
        })
    return races


def segment(races, key_fn):
    groups = defaultdict(list)
    for r in races:
        groups[key_fn(r)].append(r)

    out = {}
    for k, rs in sorted(groups.items()):
        n = len(rs)
        off_wins = sum(1 for r in rs if r["off_hit"])
        on_wins = sum(1 for r in rs if r["on_hit"])
        off_cost = n * 100
        off_payout = sum(r["win_pay"] for r in rs if r["off_hit"])
        on_cost = n * 100
        on_payout = sum(r["win_pay"] for r in rs if r["on_hit"])

        def roi(c, p): return (p - c) / c if c else 0

        out[k] = {
            "n": n,
            "off_wins": off_wins,
            "on_wins": on_wins,
            "off_roi": roi(off_cost, off_payout),
            "on_roi": roi(on_cost, on_payout),
            "delta_roi": roi(on_cost, on_payout) - roi(off_cost, off_payout),
        }
    return out


def print_seg(title, m):
    print(f"\n-- {title} --")
    print(f"  {'segment':<22} {'n':>4} {'off_wins':>8} {'on_wins':>8} {'off_roi':>9} {'on_roi':>9} {'ΔROI':>9}")
    for k, d in m.items():
        print(f"  {k:<22} {d['n']:>4} {d['off_wins']:>8} {d['on_wins']:>8} "
              f"{d['off_roi']*100:>+8.1f}% {d['on_roi']*100:>+8.1f}% {d['delta_roi']*100:>+8.1f}pt")


def main():
    races = collect()
    print("=" * 78)
    print(f"  SEGMENT (v4 ON vs OFF) — full n={len(races)}")
    print("=" * 78)
    print_seg("By Grade", segment(races, lambda r: r["grade"]))
    print_seg("By Distance", segment(races, lambda r: r["dist_band"]))
    print_seg("By Surface", segment(races, lambda r: r["surface"]))
    print_seg("By Field Size", segment(races, lambda r: r["field_band"]))

    out = PROJECT_ROOT / "data" / "segment_analysis_n221.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "total_races": len(races),
            "by_grade":    segment(races, lambda r: r["grade"]),
            "by_distance": segment(races, lambda r: r["dist_band"]),
            "by_surface":  segment(races, lambda r: r["surface"]),
            "by_field_size": segment(races, lambda r: r["field_band"]),
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()
