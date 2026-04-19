"""Root-cause analysis of v3→v4 regressions.

For each race where v3 hit but v4 missed (or where v4 moved in a clearly
bad direction), dump the camp_composite distribution and the base/on
score differences to understand WHY camp z-score worked against us.

USAGE:
  python tools/loss_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.compare_v2_v3 import _load_results, _winner, _norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V3_DIR = PROJECT_ROOT / "data" / "backtest_predictions_v3"
V4_DIR = PROJECT_ROOT / "data" / "backtest_predictions"


def _load(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_race(v3, v4, result):
    """Produce a per-horse diff showing the effect of z-score on ranking."""
    v3_ranked = v3.get("ranked") or []
    v4_ranked = v4.get("ranked") or []

    # Build lookup by name
    v3_by_name = {_norm(h.get("name")): {
        "rank": i + 1, "score": h.get("score", 0), "odds": h.get("odds", 0),
        "camp": h.get("camp_composite", 0.5),
        "win_prob": h.get("win_prob", 0),
    } for i, h in enumerate(v3_ranked)}
    v4_by_name = {_norm(h.get("name")): {
        "rank": i + 1, "score": h.get("score", 0), "odds": h.get("odds", 0),
        "camp": h.get("camp_composite", 0.5),
        "win_prob": h.get("win_prob", 0),
    } for i, h in enumerate(v4_ranked)}

    names = list(set(v3_by_name.keys()) | set(v4_by_name.keys()))

    rows = []
    for name in names:
        v3h = v3_by_name.get(name, {})
        v4h = v4_by_name.get(name, {})
        rows.append({
            "name": name,
            "v3_rank": v3h.get("rank", 99),
            "v4_rank": v4h.get("rank", 99),
            "rank_delta": v3h.get("rank", 99) - v4h.get("rank", 99),
            "score_delta": v4h.get("score", 0) - v3h.get("score", 0),
            "camp": v4h.get("camp", v3h.get("camp", 0.5)),
            "odds": v4h.get("odds", v3h.get("odds", 0)),
        })

    # Compute camp statistics for this race
    camps = [r["camp"] for r in rows]
    mean_camp = sum(camps) / len(camps) if camps else 0.5
    var = sum((c - mean_camp) ** 2 for c in camps) / len(camps) if camps else 0
    std_camp = var ** 0.5 or 1.0

    # Annotate z-score per horse
    for r in rows:
        r["camp_z"] = (r["camp"] - mean_camp) / std_camp if std_camp > 0 else 0.0

    # Sort by v4 rank
    rows.sort(key=lambda r: r["v4_rank"])

    return {
        "v3_top1": _norm(v3_ranked[0].get("name")) if v3_ranked else "",
        "v4_top1": _norm(v4_ranked[0].get("name")) if v4_ranked else "",
        "mean_camp": round(mean_camp, 3),
        "std_camp":  round(std_camp, 3),
        "rows":      rows,
    }


def main() -> int:
    results_map = _load_results()
    v3_files = {p.stem.replace("_on", ""): p for p in V3_DIR.glob("*_on.json")}
    v4_files = {p.stem.replace("_on", ""): p for p in V4_DIR.glob("*_on.json")}

    regressions = []
    improvements = []

    for rid in sorted(v3_files.keys()):
        if rid not in v4_files:
            continue
        v3 = _load(v3_files[rid])
        v4 = _load(v4_files[rid])
        result = results_map.get(rid, {})
        winner = _winner(result)
        if not winner:
            continue

        v3_top1 = _norm((v3.get("ranked") or [{}])[0].get("name"))
        v4_top1 = _norm((v4.get("ranked") or [{}])[0].get("name"))

        if v3_top1 == v4_top1:
            continue

        detail = analyze_race(v3, v4, result)
        detail["race_id"] = rid
        detail["race_name"] = v4.get("race_name", "")
        detail["race_date"] = v4.get("race_date", "")
        detail["winner"] = winner

        v3_hit = v3_top1 == winner
        v4_hit = v4_top1 == winner

        if v3_hit and not v4_hit:
            regressions.append(detail)
        elif v4_hit and not v3_hit:
            improvements.append(detail)

    print("=" * 78)
    print("  LOSS ANALYSIS — v3 WIN → v4 LOSS")
    print("=" * 78)
    print(f"  Regressions: {len(regressions)}")
    print(f"  Improvements: {len(improvements)}")
    print()

    for r in regressions:
        print(f"-- {r['race_date']} {r['race_name']} ({r['race_id']}) --")
        print(f"  winner:  {r['winner']}")
        print(f"  v3_top1: {r['v3_top1']} (HIT)")
        print(f"  v4_top1: {r['v4_top1']} (MISS)")
        print(f"  race camp stats: mean={r['mean_camp']:.3f} std={r['std_camp']:.3f}")
        print(f"  {'name':18} {'v3r':>4} {'v4r':>4} {'Δrank':>6} {'score_Δ':>8} {'camp':>6} {'camp_z':>7} {'odds':>6}")
        for row in r["rows"][:8]:
            mark = ""
            if row["name"] == r["winner"]:
                mark = " ← winner"
            elif row["name"] == r["v3_top1"]:
                mark = " ← v3"
            elif row["name"] == r["v4_top1"]:
                mark = " ← v4"
            print(f"  {row['name']:18} {row['v3_rank']:>4} {row['v4_rank']:>4} "
                  f"{row['rank_delta']:>+6} "
                  f"{row['score_delta']:>+8.2f} "
                  f"{row['camp']:>6.2f} {row['camp_z']:>+7.2f} "
                  f"{row['odds']:>6.1f}{mark}")
        print()

    print()
    print("=" * 78)
    print("  BONUS: IMPROVEMENTS (v3 MISS → v4 WIN)")
    print("=" * 78)
    for r in improvements:
        print(f"  {r['race_date']} {r['race_name']}: {r['v3_top1']} → {r['v4_top1']} (WIN)")

    out = PROJECT_ROOT / "data" / "loss_analysis_v3_v4.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({
            "regressions": regressions,
            "improvements": improvements,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
