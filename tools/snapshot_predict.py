"""Predict from snapshot — leak-safe backtest prediction runner.

Runs live_pipeline.predict_live against a snapshot, with scraper
monkey-patched to serve snapshot data only. No network, no post-race
data.

Also supports pedigree ON/OFF ablation on the same snapshot for pure
feature-effect isolation.

USAGE:
  python tools/snapshot_predict.py --race-id 202503030511
  python tools/snapshot_predict.py --all
  python tools/snapshot_predict.py --race-id 202503030511 --pedigree off
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.snapshot_reader import patch_scraper, load_snapshot
from tools.build_snapshot import LeakError, leak_check

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = PROJECT_ROOT / "data" / "backtest_predictions"
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshot"


def _strip_pedigree(result: dict) -> dict:
    """Return a copy with pedigree composites neutralized.

    Used to re-score the same ranked list as if pedigree features didn't
    exist. Since live_pipeline already produced final ranked[] with
    pedigree injected, we need to re-run scoring with pedigree=0.5.

    This helper is called on the snapshot BEFORE predict_live, by
    re-invoking predict_live in a patched state that zeros pedigree.
    """
    # (placeholder — actual neutralization done via separate invocation)
    return result


def run_predict_from_snapshot(race_id: str, pedigree: str = "on",
                               save: bool = True) -> dict:
    """Run predict_live against snapshot[race_id] with scraper patched.

    pedigree:
      - "on"  : use actual pedigree_composite from entity_tier
      - "off" : force pedigree_composite = camp_composite = sire_distance_fit = 0.5
    """
    snap = load_snapshot(race_id)  # raises LeakError if stale/leaky

    # Import lazily so the patch applies to the right module objects
    import scraper
    import live_pipeline
    import feature_store as fs
    import pedigree_features as pf

    # For pedigree=off, we patch feature_store's pedigree extraction to
    # neutral. This is the cleanest isolation: we keep all other features
    # intact and only zero the pedigree layer.
    pf_original = pf.extract_pedigree_features

    def _neutral_pedigree(**kwargs):
        # Return the same shape but with all scores at 0.5 / neutral
        return {
            "sire_name":            kwargs.get("sire_name", ""),
            "dam_name":             kwargs.get("dam_name", ""),
            "damsire_name":         kwargs.get("damsire_name", ""),
            "breeder_name":         kwargs.get("breeder_name", ""),
            "owner_name":           kwargs.get("owner_name", ""),
            "sire_tier_score":      0.5,
            "damsire_tier_score":   0.5,
            "sire_distance_fit":    0.5,
            "sire_surface_fit":     0.5,
            "sire_heavy_track_fit": 0.5,
            "breeder_tier_score":   0.5,
            "owner_tier_score":     0.5,
            "external_stable_score": 0.5,
            "pedigree_composite":   0.5,
            "camp_composite":       0.5,
            "pedigree_has_signal":  False,
            "camp_has_signal":      False,
            "pedigree_n_dims":      0,
            "camp_n_dims":          0,
            "missing_feature_count": 7,
        }

    if pedigree == "off":
        pf.extract_pedigree_features = _neutral_pedigree

    try:
        with patch_scraper(snap):
            venue = snap.get("venue", "")
            race_name = snap.get("race_name", "")
            race_date = snap["race_date"]

            result = live_pipeline.predict_live(
                race_id=race_id,
                venue=venue,
                race_name=race_name,
                race_date=race_date,
                progress_cb=None,
                auto_log=False,  # CRITICAL: never touch live_predictions.json
            )
    finally:
        pf.extract_pedigree_features = pf_original

    # Strip any forbidden fields from the prediction result (defense in depth)
    FORBIDDEN = frozenset([
        "finishing_order", "payouts", "result", "final_odds",
        "actual_rank", "win_time",
    ])
    def _strip(obj):
        if isinstance(obj, dict):
            return {k: _strip(v) for k, v in obj.items() if k not in FORBIDDEN}
        if isinstance(obj, list):
            return [_strip(v) for v in obj]
        return obj

    result = _strip(result)

    # Add backtest metadata
    result["_backtest_meta"] = {
        "snapshot_version":  snap.get("snapshot_version"),
        "snapshot_built_at": snap.get("snapshot_built_at"),
        "pedigree_mode":     pedigree,
        "leak_clear":        snap["leak_audit"]["leak_clear"],
        "recent_races_dropped": snap["leak_audit"]["recent_races_dropped"],
        "predicted_at":      dt.datetime.now().isoformat(timespec="seconds"),
    }

    if save:
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        out = PRED_DIR / f"{race_id}_{pedigree}.json"
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        os.replace(tmp, out)

    return result


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-id", help="Run a single race_id")
    parser.add_argument("--all", action="store_true",
                        help="Run all snapshots in data/snapshot/")
    parser.add_argument("--pedigree", choices=["on", "off", "both"],
                        default="both",
                        help="Run with pedigree on, off, or both (default: both)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit to first N snapshots (0 = no limit)")
    args = parser.parse_args()

    modes = ["on", "off"] if args.pedigree == "both" else [args.pedigree]

    if args.all:
        snap_files = sorted(SNAPSHOT_DIR.glob("*.json"))
        if args.limit > 0:
            snap_files = snap_files[:args.limit]
        print(f"[info] Predicting {len(snap_files)} snapshots × {len(modes)} mode(s)")
        ok, fail = 0, 0
        for p in snap_files:
            rid = p.stem
            for mode in modes:
                try:
                    r = run_predict_from_snapshot(rid, pedigree=mode, save=True)
                    top1 = (r.get("ranked") or [{}])[0].get("name", "?")
                    print(f"  OK  {rid} [{mode}] top1={top1}")
                    ok += 1
                except LeakError as e:
                    print(f"  LEAK {rid} [{mode}]: {e}")
                    fail += 1
                except Exception as e:
                    print(f"  FAIL {rid} [{mode}]: {e}")
                    fail += 1
        print(f"\n[summary] ok={ok} fail={fail}")
        return 0 if fail == 0 else 1

    if not args.race_id:
        parser.error("--race-id required (or use --all)")

    for mode in modes:
        result = run_predict_from_snapshot(args.race_id, pedigree=mode, save=True)
        ranked = result.get("ranked") or []
        print(f"\n=== {args.race_id} [pedigree={mode}] ===")
        print(f"  prediction_stage: {result.get('prediction_stage')}")
        print(f"  odds_status:      {result.get('odds_status')}")
        print(f"  leak_clear:       {result['_backtest_meta']['leak_clear']}")
        print(f"  recent_races_dropped: "
              f"{result['_backtest_meta']['recent_races_dropped']}")
        print(f"  Top 5:")
        for i, h in enumerate(ranked[:5], 1):
            print(f"    {i}. {h.get('name','?'):20} odds={h.get('odds',0):6.1f} "
                  f"win_prob={h.get('win_prob',0):.4f} "
                  f"ped={h.get('pedigree_composite',0.5):.3f} "
                  f"camp={h.get('camp_composite',0.5):.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
