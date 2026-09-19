"""期間指定で G1/G2(/G3) レースを発見し、予測ファイルと結果を冪等に取得する。

前身 tools/fetch_spring2026_races.py (日付ハードコード) の汎用版。

WHAT
  1. --start〜--end の土日祝(JRA開催日は土日+一部月曜) を fetch_race_list_netkeiba でスキャン
  2. 未予測レース: live_pipeline.predict_live() を後追い実行 → data/backtest_predictions/<id>_on.json
  3. 未取得結果:   scraper.fetch_result_netkeiba() → data/results.json["bt_<id>"] (payouts_detail 含む)
     既存エントリに payouts_detail が無ければ補完。
  4. レース日が未来 (結果未確定) のものは結果取得をスキップ。

READ/WRITE: data/backtest_predictions/, data/results.json  (アトミック書込)
NEVER touches: evaluator.py / train.py / probability_engine.py / LOOSE 条件

USAGE
  python3 tools/fetch_recent_races.py --start 2026-06-15 --end 2026-09-19
  python3 tools/fetch_recent_races.py --start 2026-06-15 --end 2026-09-19 --dry-run
  python3 tools/fetch_recent_races.py --start 2026-09-19 --end 2026-09-21 --predict-only
  python3 tools/fetch_recent_races.py ... --grades G1,G2,G3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PRED_DIR = ROOT / "data" / "backtest_predictions"
RES_PATH = ROOT / "data" / "results.json"

_GRADE_RE = re.compile(r"\((G[123]|Jpn(?:I{1,3}))\)")


def _grade_from_name(name: str) -> str:
    m = _GRADE_RE.search(name or "")
    return m.group(1) if m else ""


def iter_race_days(start: date, end: date):
    """JRA 開催日候補: 土日 + 月曜 (祝日開催の可能性)。存在しなければ空リストが返るだけ。"""
    d = start
    while d <= end:
        if d.weekday() in (5, 6, 0):
            yield d
        d += timedelta(days=1)


def _atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def discover(start: date, end: date, grades: tuple[str, ...], sleep_s: float) -> list[dict]:
    import scraper
    found: list[dict] = []
    for d in iter_race_days(start, end):
        try:
            races = scraper.fetch_race_list_netkeiba(d, graded_only=True, grades=grades)
        except Exception as e:  # noqa: BLE001
            print(f"  {d} fetch_race_list failed: {e}")
            races = []
        for r in races:
            rid = str(r.get("race_id") or "")
            if not rid:
                continue
            found.append({
                "race_id": rid,
                "race_name": r.get("race_name", ""),
                "race_date": d.isoformat(),
                "venue": r.get("venue", ""),
                "grade": r.get("grade") or _grade_from_name(r.get("race_name", "")),
                "time": r.get("time", ""),
            })
        time.sleep(sleep_s)
    found.sort(key=lambda r: (r["race_date"], r["race_id"]))
    return found


def ensure_prediction(r: dict, force: bool = False) -> str:
    import live_pipeline
    out = PRED_DIR / f"{r['race_id']}_on.json"
    if out.exists() and not force:
        p = json.loads(out.read_text(encoding="utf-8"))
        if len(p.get("ranked") or []) < 3:
            # 過去の失敗実行が残した空ファイル → 再生成対象にする
            print(f"  pred:   既存ファイルが空 (ranked={len(p.get('ranked') or [])}) → 再生成")
            out.unlink()
            return ensure_prediction(r, force=True)
        # 旧ファイルの grade 空欄を補完 (predict_live v5.11 以前の出力)
        if not p.get("grade") and r.get("grade"):
            p["grade"] = r["grade"]
            _atomic_write_json(out, p)
            return "exists (grade patched)"
        return "exists"
    pred = live_pipeline.predict_live(
        race_id=r["race_id"], venue=r["venue"], race_name=r["race_name"],
        race_date=r["race_date"], progress_cb=None, auto_log=False,
    )
    ranked = pred.get("ranked") or []
    if len(ranked) < 3:
        # ネットワーク断などで出馬表が取れなかった場合。空ファイルを残すと
        # 次回 "exists" 扱いで永久にスキップされるので保存しない。
        raise RuntimeError(
            f"ranked={len(ranked)}頭 (出馬表取得失敗?) odds_status={pred.get('odds_status')} — 保存せず")
    if not pred.get("grade") and r.get("grade"):
        pred["grade"] = r["grade"]
    pred["_backtest_meta"] = {
        "retroactive": True,
        "predicted_at": datetime.now().isoformat(timespec="seconds"),
        "tool": "tools/fetch_recent_races.py",
    }
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out, pred)
    ranked = pred.get("ranked") or []
    top = ranked[0].get("name") if ranked else "?"
    return f"new ({len(ranked)}頭, top={top}, odds_status={pred.get('odds_status')})"


def ensure_result(r: dict, results: dict) -> str:
    import scraper
    key = f"bt_{r['race_id']}"
    cur = results.get(key)
    if cur and cur.get("finishing_order") and cur.get("payouts_detail"):
        return "exists"
    res = scraper.fetch_result_netkeiba(r["race_id"])
    if not res or not res.get("finishing_order"):
        return "EMPTY (結果未確定 or 取得失敗 — 未保存)"
    if cur:
        cur.update(res)
        status = "payouts_detail patched"
    else:
        results[key] = {"race_name": r["race_name"],
                        "timestamp": r["race_date"] + "T12:00:00", **res}
        status = "new"
    _atomic_write_json(RES_PATH, results)
    fo = res.get("finishing_order") or []
    winner = next((h.get("name") for h in fo if str(h.get("rank")) == "1"), "?")
    tansho = (res.get("payouts") or {}).get("単勝", "?")
    wide = (res.get("payouts_detail") or {}).get("ワイド", [])
    return f"{status} 1着={winner} 単勝={tansho} ワイド={len(wide)}点"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--grades", default="G1,G2")
    ap.add_argument("--dry-run", action="store_true", help="発見のみ、書込なし")
    ap.add_argument("--predict-only", action="store_true", help="結果取得をしない")
    ap.add_argument("--force-predict", action="store_true", help="既存予測を再生成")
    ap.add_argument("--sleep", type=float, default=0.7)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    grades = tuple(g.strip() for g in args.grades.split(",") if g.strip())
    today = date.today()

    print(f"[discover] {start}..{end} grades={grades}")
    races = discover(start, end, grades, args.sleep)
    for r in races:
        print(f"  {r['race_date']} [{r['grade']}] {r['race_name']} id={r['race_id']} {r['time']}")
    print(f"[discover] {len(races)} races")
    if args.dry_run or not races:
        return 0

    results = json.loads(RES_PATH.read_text(encoding="utf-8")) if RES_PATH.exists() else {}
    for r in races:
        print(f"\n[{r['grade']}] {r['race_id']} {r['race_name']} {r['race_date']}", flush=True)
        try:
            print("  pred:  ", ensure_prediction(r, force=args.force_predict), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  pred:   FAIL {e}", flush=True)
        if args.predict_only:
            continue
        if date.fromisoformat(r["race_date"]) > today:
            print("  result: skip (未来のレース)", flush=True)
            continue
        try:
            print("  result:", ensure_result(r, results), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  result: FAIL {e}", flush=True)
        time.sleep(args.sleep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
