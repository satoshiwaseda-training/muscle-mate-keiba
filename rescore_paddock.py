"""Re-score paddock features on the 51 cached enrich_race files.

For each cached race:
  1. Get horse names from the cached entries
  2. Get the raw paddock_comment text (one-horse or whole-page)
  3. Extract per-horse segments (handles full-page chrome correctly)
  4. Score each segment with the new 4-dim extractor
  5. Compute score_runner-compatible keys

Then rebuild `structured_features.horses` with the new paddock values
and rerun the score_runner + softmax pipeline. Save to a new results
file so the old 50-race baseline is preserved for comparison.

No network calls.
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

import paddock_features as pf
import feature_store as fs
from train import score_runner
from data_store import load_weights
import probability_engine as pe

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "scraper_cache" / "enrich_race"
PRED = ROOT / "data" / "predictions.json"
RES = ROOT / "data" / "results.json"
ENR = ROOT / "data" / "enriched_backtest_results.json"
OUT = ROOT / "data" / "enriched_backtest_results_v2.json"

TICKET = 100


def _norm(n): return (n or "").strip()
def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def main():
    preds = json.loads(PRED.read_text(encoding="utf-8"))
    results = json.loads(RES.read_text(encoding="utf-8"))
    old_enriched = json.loads(ENR.read_text(encoding="utf-8"))
    old_by_rid = {e["race_id"]: e for e in old_enriched}

    # Only process races that are in the old 50-race output AND have a cache file
    rows_out = []
    cov_stats = {"total_horses": 0,
                 "nonzero_gait": 0, "nonzero_hq": 0, "nonzero_vasc": 0, "nonzero_mental": 0,
                 "phrase_hits": 0, "grade_fallback": 0, "zero_signal": 0}
    race_grade_dist = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "E": 0, "none": 0}

    ctx = {"weights": load_weights()}

    for old in old_enriched:
        rid = old["race_id"]
        cache_file = CACHE / f"{rid}.json"
        if not cache_file.exists():
            continue
        cached_horses = json.loads(cache_file.read_text(encoding="utf-8"))
        if not cached_horses:
            continue
        horse_names = [h.get("name", "") for h in cached_horses]
        # All horses share the same page text (stored per horse)
        page_text = ""
        for h in cached_horses:
            pt = h.get("paddock_comment") or ""
            if len(pt) > len(page_text):
                page_text = pt

        # Extract per-horse segments from the page text
        segments = pf.extract_per_horse_comments(page_text, horse_names)

        # Score each horse
        per_horse_scores = {}
        for name in horse_names:
            seg = segments.get(name, {})
            sc = pf.score_from_segment(seg) if seg else {
                "gait_score": 0.5, "hindquarter_strength": 0.5,
                "vascularity": 0.5, "mental_state": 0.5, "hits": 0,
            }
            per_horse_scores[name] = sc
            letter = seg.get("grade_letter") if seg else ""
            race_grade_dist[letter or "none"] = race_grade_dist.get(letter or "none", 0) + 1

            cov_stats["total_horses"] += 1
            if sc["gait_score"] != 0.5: cov_stats["nonzero_gait"] += 1
            if sc["hindquarter_strength"] != 0.5: cov_stats["nonzero_hq"] += 1
            if sc["vascularity"] != 0.5: cov_stats["nonzero_vasc"] += 1
            if sc["mental_state"] != 0.5: cov_stats["nonzero_mental"] += 1
            if sc["hits"] > 0: cov_stats["phrase_hits"] += 1
            elif sc["hits"] == -1: cov_stats["grade_fallback"] += 1
            else: cov_stats["zero_signal"] += 1

        # Rebuild structured_features.horses with new paddock values
        sf = dict(old.get("structured_features") or {})
        sf_horses = dict(sf.get("horses") or {})
        for name, h_sf in list(sf_horses.items()):
            if name not in per_horse_scores:
                continue
            new_h = dict(h_sf)
            sc = per_horse_scores[name]
            new_h["paddock_gait_score_01"] = sc["gait_score"]
            new_h["paddock_hindquarter_01"] = sc["hindquarter_strength"]
            new_h["paddock_vascularity_01"] = sc["vascularity"]
            new_h["paddock_mental_state_01"] = sc["mental_state"]
            # Map to score_runner's [-1, 1] keys
            centered = pf.to_score_runner_keys(sc)
            new_h["paddock_gait"] = centered["paddock_gait"]
            new_h["paddock_hindquarter"] = centered["paddock_hindquarter"]
            new_h["paddock_vascularity"] = centered["paddock_vascularity"]
            sf_horses[name] = new_h
        sf["horses"] = sf_horses

        # Re-score all horses with the updated structured_features
        entries = list(sf_horses.keys())
        grade = old.get("grade", "")
        scored = []
        for this_name in entries:
            rotated = [{"name": this_name, "rank": 1,
                        "odds": sf_horses[this_name].get("odds", 0),
                        "confidence": 0, "ev_gap": 0, "bet": ""}]
            for other in entries:
                if other == this_name:
                    continue
                rotated.append({"name": other, "rank": 2,
                                "odds": sf_horses[other].get("odds", 0),
                                "confidence": 0, "ev_gap": 0, "bet": ""})
            feat = {"grade": grade, "num_horses": len(entries),
                    "horse_features": rotated, "structured_features": sf}
            s = score_runner(feat, ctx).get("top_confidence", 50.0)
            scored.append({
                "name": this_name,
                "odds": sf_horses[this_name].get("odds", 0),
                "score": float(s),
            })
        ranked = pe.assign_win_probs(scored, temperature=pe.DEFAULT_TEMPERATURE)
        sel = pe.select_top3(ranked, alpha=pe.DEFAULT_ALPHA, beta=pe.DEFAULT_BETA)

        rows_out.append({
            "race_id": rid,
            "race_name": old.get("race_name", ""),
            "grade": grade,
            "ranked": ranked,
            "selected_top3": sel["selected"],
            "p1": sel["p1"], "p2": sel["p2"],
            "structured_features": sf,
        })

    # Save rescored output
    OUT.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows_out)} races to {OUT.relative_to(ROOT)}")

    # Coverage report
    print("\n" + "=" * 72)
    print("PADDOCK FEATURE COVERAGE (new extractor, offline rescore)")
    print("=" * 72)
    n = cov_stats["total_horses"] or 1
    print(f"  Total horses scored        : {cov_stats['total_horses']}")
    print(f"  Comments with phrase hits  : {cov_stats['phrase_hits']} ({cov_stats['phrase_hits']/n*100:.1f}%)")
    print(f"  Grade-letter fallback only : {cov_stats['grade_fallback']} ({cov_stats['grade_fallback']/n*100:.1f}%)")
    print(f"  Zero signal (pure neutral) : {cov_stats['zero_signal']} ({cov_stats['zero_signal']/n*100:.1f}%)")
    print(f"  Non-neutral gait           : {cov_stats['nonzero_gait']} ({cov_stats['nonzero_gait']/n*100:.1f}%)")
    print(f"  Non-neutral hindquarter    : {cov_stats['nonzero_hq']} ({cov_stats['nonzero_hq']/n*100:.1f}%)")
    print(f"  Non-neutral vascularity    : {cov_stats['nonzero_vasc']} ({cov_stats['nonzero_vasc']/n*100:.1f}%)")
    print(f"  Non-neutral mental_state   : {cov_stats['nonzero_mental']} ({cov_stats['nonzero_mental']/n*100:.1f}%)")
    print(f"\n  Grade-letter distribution  : {dict(sorted(race_grade_dist.items()))}")


if __name__ == "__main__":
    main()
