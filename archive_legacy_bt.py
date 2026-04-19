"""One-shot: move legacy bt_ rows (no `_ranking_meta`) out of the live
predictions/results files so the new evaluator only sees clean-ranker data.

Legacy rows are preserved in data/archive/legacy_bt_predictions.json and
legacy_bt_results.json for auditing.

Usage:
    python archive_legacy_bt.py           # dry-run summary
    python archive_legacy_bt.py --apply   # actually move the rows
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ARCHIVE = DATA / "archive"
PRED_FILE = DATA / "predictions.json"
RES_FILE = DATA / "results.json"
ARCH_PRED = ARCHIVE / "legacy_bt_predictions.json"
ARCH_RES = ARCHIVE / "legacy_bt_results.json"


def load_json(p: Path) -> dict:
    if not p.exists(): return {}
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write changes to disk (default: dry run)")
    args = ap.parse_args()

    preds = load_json(PRED_FILE)
    res = load_json(RES_FILE)

    legacy_keys = [
        k for k, v in preds.items()
        if k.startswith("bt_") and not v.get("_ranking_meta")
    ]
    clean_keys = [k for k in preds if k.startswith("bt_") and preds[k].get("_ranking_meta")]
    non_bt_keys = [k for k in preds if not k.startswith("bt_")]

    print(f"Total predictions     : {len(preds)}")
    print(f"  bt_ legacy (no meta): {len(legacy_keys)}")
    print(f"  bt_ clean           : {len(clean_keys)}")
    print(f"  non-bt              : {len(non_bt_keys)}")
    print(f"Will archive          : {len(legacy_keys)} legacy bt_ rows")

    if not args.apply:
        print("\nDry run — pass --apply to move.")
        return

    arch_preds = load_json(ARCH_PRED)
    arch_res = load_json(ARCH_RES)

    moved = 0
    for k in legacy_keys:
        arch_preds[k] = preds.pop(k)
        if k in res:
            arch_res[k] = res.pop(k)
        moved += 1

    save_json(PRED_FILE, preds)
    save_json(RES_FILE, res)
    save_json(ARCH_PRED, arch_preds)
    save_json(ARCH_RES, arch_res)

    print(f"\nMoved {moved} rows to {ARCH_PRED.relative_to(ROOT)} / {ARCH_RES.relative_to(ROOT)}")
    print(f"Live predictions.json now has {len(preds)} rows.")


if __name__ == "__main__":
    sys.exit(main())
