"""
Spring 2026 G1/G2 レース発見・予測生成・結果取得スクリプト。

1. 2026-04-06 〜 2026-07-04 の全土日を fetch_race_list_netkeiba でスキャン
2. G1/G2 で未予測のレースを特定
3. predict_live() を呼んで backtest_predictions/*_on.json を生成
4. fetch_result_netkeiba() で結果取得 → results.json に追記
5. payouts_detail バックフィル

READ/WRITE: backtest_predictions/, results.json
NEVER touches: evaluator.py, train.py, probability_engine.py
"""
from __future__ import annotations
import json, os, sys, time
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "data" / "backtest_predictions"
RES_PATH = ROOT / "data" / "results.json"

import scraper

# ── 日付範囲 ──────────────────────────────────────────────
START = date(2026, 4, 6)
END   = date(2026, 7, 4)

def iter_weekends(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() in (5, 6):  # Sat=5 Sun=6
            yield d
        d += timedelta(days=1)


def load_existing_race_ids():
    ids = set()
    for f in PRED_DIR.glob("*_on.json"):
        ids.add(f.stem.replace("_on", ""))
    return ids


def load_results():
    if RES_PATH.exists():
        return json.loads(RES_PATH.read_text(encoding="utf-8"))
    return {}


def save_results(results: dict):
    tmp = RES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, RES_PATH)


# ── メイン ──────────────────────────────────────────────
def main():
    existing = load_existing_race_ids()
    print(f"[info] existing predictions: {len(existing)}")

    new_races = []
    for d in iter_weekends(START, END):
        try:
            races = scraper.fetch_race_list_netkeiba(d, graded_only=True, grades=("G1", "G2"))
            time.sleep(0.5)
        except Exception as e:
            print(f"  {d} fetch_race_list failed: {e}")
            continue
        for r in races:
            rid = str(r.get("race_id") or "")
            if not rid:
                continue
            if rid in existing:
                continue
            new_races.append({
                "race_id": rid,
                "race_name": r.get("race_name", ""),
                "race_date": d.isoformat(),
                "venue": r.get("venue", ""),
                "grade": r.get("grade", ""),
            })
            print(f"  [NEW] {d} {r.get('grade')} {r.get('race_name')} id={rid}")
        time.sleep(0.5)

    print(f"\n[info] found {len(new_races)} new G1/G2 races")

    if not new_races:
        print("[done] no new races to process")
        return

    # ── 予測生成 ──────────────────────────────────────────
    import live_pipeline
    import prediction_log

    results = load_results()
    success_pred, fail_pred = 0, 0
    success_res, fail_res = 0, 0

    for r in new_races:
        rid = r["race_id"]
        print(f"\n[predict] {rid} {r['race_name']} ({r['race_date']})")
        try:
            pred = live_pipeline.predict_live(
                race_id=rid,
                venue=r["venue"],
                race_name=r["race_name"],
                race_date=r["race_date"],
                progress_cb=None,
                auto_log=False,
            )
            pred["_backtest_meta"] = {
                "predicted_at": "2026-07-04T00:00:00",
                "note": "retroactive spring2026 prediction",
            }
            out = PRED_DIR / f"{rid}_on.json"
            tmp = out.with_name(out.name + ".tmp")
            tmp.write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, out)
            success_pred += 1
            print(f"  -> prediction saved ({len(pred.get('ranked') or [])} horses)")
        except Exception as e:
            fail_pred += 1
            print(f"  -> predict failed: {e}")

        # ── 結果取得 ──────────────────────────────────────
        key = f"bt_{rid}"
        if key not in results:
            try:
                res = scraper.fetch_result_netkeiba(rid)
                if res and res.get("finishing_order"):
                    results[key] = {
                        "race_name": r["race_name"],
                        "timestamp": r["race_date"] + "T12:00:00",
                        **res,
                    }
                    save_results(results)
                    success_res += 1
                    fo = res.get("finishing_order") or []
                    pd = res.get("payouts_detail") or {}
                    print(f"  -> result saved (fo={len(fo)}, ワイド={len(pd.get('ワイド',[]))})")
                else:
                    fail_res += 1
                    print(f"  -> result empty")
                time.sleep(1.5)
            except Exception as e:
                fail_res += 1
                print(f"  -> result failed: {e}")
        else:
            print(f"  -> result already in results.json")

    print(f"\n[summary] predictions: {success_pred} ok / {fail_pred} fail")
    print(f"[summary] results:     {success_res} ok / {fail_res} fail")


if __name__ == "__main__":
    main()
