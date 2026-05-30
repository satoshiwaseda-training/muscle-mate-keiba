"""Backfill per-combo payouts (payouts_detail) into data/results.json.

WHY: results.json was scraped with the old parser that kept only the first
(winner's) value for 複勝/ワイド. The fixed scraper now returns
`payouts_detail` with per-horse 複勝 and per-pair ワイド payouts, which is
required for investment-grade 複勝/ワイド backtesting.

WHAT: for each G1/G2 race in data/backtest_predictions/*_on.json, fetch the
result page again and MERGE `payouts_detail` into results.json without
disturbing the existing `finishing_order` / `payouts` (those are validated
backtest inputs). Non-destructive, resumable, atomic write, polite delay.

Run:
  python3 tools/backfill_payouts_detail.py            # G1/G2 only (default)
  python3 tools/backfill_payouts_detail.py --all      # every race in results.json
  python3 tools/backfill_payouts_detail.py --limit 5  # smoke test
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import scraper  # noqa: E402

RESULTS = ROOT / "data" / "results.json"
PRED_DIR = ROOT / "data" / "backtest_predictions"
DELAY_SEC = 1.5


def bucket(g):
    g = (g or "").upper()
    return "G1" if "G1" in g else ("G2" if "G2" in g else "OTHER")


def g1g2_race_ids():
    ids = []
    for f in sorted(glob.glob(str(PRED_DIR / "*_on.json"))):
        try:
            p = json.loads(Path(f).read_text(encoding="utf-8"))
        except Exception:
            continue
        if bucket(p.get("grade")) in ("G1", "G2"):
            rid = str(p.get("race_id"))
            if rid and rid not in ids:
                ids.append(rid)
    return ids


def atomic_write(path: Path, data: dict):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="backfill every race in results.json")
    ap.add_argument("--limit", type=int, default=0, help="cap number of fetches (smoke test)")
    ap.add_argument("--force", action="store_true", help="re-fetch even if payouts_detail present")
    args = ap.parse_args()

    results = json.loads(RESULTS.read_text(encoding="utf-8"))

    if args.all:
        targets = [k[3:] if k.startswith("bt_") else k for k in results.keys()]
    else:
        targets = g1g2_race_ids()
    print(f"targets: {len(targets)} races ({'ALL' if args.all else 'G1/G2'})")

    done = skipped = failed = added = 0
    for i, rid in enumerate(targets, 1):
        key = f"bt_{rid}"
        existing = results.get(key) or results.get(rid) or {}
        if existing.get("payouts_detail") and not args.force:
            skipped += 1
            continue
        if args.limit and done >= args.limit:
            break
        try:
            res = scraper.fetch_result_netkeiba(rid)
        except Exception as e:
            print(f"  [{i}/{len(targets)}] {rid} ERROR {type(e).__name__}: {e}")
            failed += 1
            time.sleep(DELAY_SEC)
            continue
        if not res or not res.get("payouts_detail"):
            print(f"  [{i}/{len(targets)}] {rid} no payouts_detail (skip)")
            failed += 1
            time.sleep(DELAY_SEC)
            continue
        # Non-destructive merge: keep existing finishing_order/payouts if present,
        # only add payouts_detail. If race is entirely new, store the full record.
        target_key = key if key in results else (rid if rid in results else key)
        if target_key in results:
            results[target_key]["payouts_detail"] = res["payouts_detail"]
            # also refresh the flat payouts if any key was missing before
            flat = results[target_key].get("payouts") or {}
            for k, v in (res.get("payouts") or {}).items():
                flat.setdefault(k, v)
            results[target_key]["payouts"] = flat
        else:
            results[target_key] = res
            added += 1
        done += 1
        # checkpoint every 10 fetches so a kill mid-run keeps progress
        if done % 10 == 0:
            atomic_write(RESULTS, results)
            print(f"  [{i}/{len(targets)}] {rid} OK (checkpoint, done={done})")
        else:
            print(f"  [{i}/{len(targets)}] {rid} OK 複勝={len(res['payouts_detail'].get('複勝',[]))}頭 "
                  f"ワイド={len(res['payouts_detail'].get('ワイド',[]))}組")
        time.sleep(DELAY_SEC)

    atomic_write(RESULTS, results)
    print(f"\nDONE  fetched={done}  added_new={added}  skipped(had detail)={skipped}  failed={failed}")
    print(f"results.json now has {len(results)} races")


if __name__ == "__main__":
    main()
