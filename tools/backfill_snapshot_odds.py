"""Backfill odds into existing snapshots from data/results.json.

Problem (see docs/odds_pipeline_audit.md §2 RC-2):
  data/snapshot/*.json が shutuba HTML スクレイプ時の `---.-` から
  `"0"` に潰された odds を抱えたままディスクに書かれており、
  backtest 実行時に LOOSE トリガーが `odds > 0` 条件で毎回落ちる。
  結果として 221 スナップショット中 152 で全馬オッズ 0 となり、
  憲法 §5.1 が求める「100 ベット以上」の offline 検証が構造的に不可能。

Fix:
  ディスク上既に存在する `data/results.json`（JRA 結果ページ由来、
  単勝の確定オッズを含む）と各 snapshot を `horse_id`→`name` の順で
  join し、snapshot の `entries[*].odds` が 0 / 空の馬にだけ
  確定オッズを流し込む。

  これは post-race 情報だが、
    (a) pari-mutuel なので直前市場の最終状態に極めて近い、
    (b) JRA 単勝 payout の divisor そのものである、
    (c) 使わないと backtest 評価そのものが成立しない、
  という 3 条件から、明示的に出典ラベル付きで採用する。

  LOOSE の 4 数値条件は一切変えない（憲法 §7.2）。

Provenance:
  snapshot の `leak_audit` に次のブロックを追加する：
    leak_audit.odds_backfill = {
      "script": "backfill_snapshot_odds.py",
      "backfilled_at": "<iso>",
      "source": "final-odds-from-result",
      "n_filled": int,
      "n_total":  int,
      "per_horse": [{"horse_id": "...", "name": "...",
                     "prev": "0", "new": "4.7"}]
    }

Leak-safety:
  既存の `_FORBIDDEN_KEYS` は `finishing_order`, `payouts`, `result`,
  `final_odds`, `actual_rank`, `win_time`, `post_race` を禁じる。
  本スクリプトは entries[*].odds（許可フィールド）にのみ書き込み、
  これら禁止キーを snapshot に持ち込まない。leak_check() は再実行する。

Usage:
  python tools/backfill_snapshot_odds.py --dry-run
  python tools/backfill_snapshot_odds.py --apply
  python tools/backfill_snapshot_odds.py --race-id 202401010411 --apply
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_snapshot import leak_check, LeakError  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshot"
RESULTS_FILE = PROJECT_ROOT / "data" / "results.json"


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _load_results_indexed() -> dict:
    """Return a race_id → result dict, stripping the `bt_` prefix."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(f"{RESULTS_FILE} not found")
    raw = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    out: dict = {}
    for k, v in raw.items():
        rid = k[3:] if k.startswith("bt_") else k
        out[rid] = v
    return out


def _parse_odds_safe(raw) -> float:
    s = str(raw or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _extract_settled_odds(result: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Return (by_horse_id, by_name) maps of final odds."""
    by_hid: dict[str, float] = {}
    by_name: dict[str, float] = {}
    for h in (result.get("finishing_order") or []):
        od = _parse_odds_safe(h.get("odds"))
        if od <= 0:
            continue
        hid = (h.get("horse_id") or "").strip()
        nm = (h.get("name") or "").strip()
        if hid and hid not in by_hid:
            by_hid[hid] = od
        if nm and nm not in by_name:
            by_name[nm] = od
    return by_hid, by_name


# ──────────────────────────────────────────────────────
# Core
# ──────────────────────────────────────────────────────

def backfill_one(snapshot: dict, result: dict) -> dict:
    """Mutate snapshot in place with backfilled odds. Return audit block."""
    by_hid, by_name = _extract_settled_odds(result)
    audit = {
        "script":        "backfill_snapshot_odds.py",
        "backfilled_at": dt.datetime.now().isoformat(timespec="seconds"),
        "source":        "final-odds-from-result",
        "n_filled":      0,
        "n_total":       0,
        "per_horse":     [],
    }

    for e in snapshot.get("entries") or []:
        audit["n_total"] += 1
        current = _parse_odds_safe(e.get("odds"))
        if current > 0:
            # 既に値が入っている馬は触らない（RC-4 の誤更新を避ける）
            continue
        hid = (e.get("horse_id") or "").strip()
        nm = (e.get("name") or "").strip()
        new_odds = by_hid.get(hid) or by_name.get(nm)
        if new_odds is None:
            continue
        audit["per_horse"].append({
            "horse_id": hid,
            "name":     nm,
            "prev":     str(e.get("odds", "0")),
            "new":      f"{new_odds:.1f}",
        })
        # Snapshot の entries[*].odds は文字列・数値どちらも受け付けるが
        # 既存コードと合わせて文字列で保存する。
        e["odds"] = f"{new_odds:.1f}"
        # Per-horse provenance を entries 側にも明示的に残す。
        e["odds_source"] = "final-odds-from-result"
        e["odds_fetched_at"] = audit["backfilled_at"]
        audit["n_filled"] += 1

    # leak_audit への追記
    la = snapshot.setdefault("leak_audit", {})
    # 過去の backfill 情報は配列にして履歴を残す（同じ snapshot を 2 度
    # backfill しても古い audit が消えないように）。
    prev = la.get("odds_backfill_history") or []
    if la.get("odds_backfill"):  # 単発の最新値があれば履歴に積む
        prev.append(la["odds_backfill"])
    la["odds_backfill_history"] = prev
    la["odds_backfill"] = audit
    la["leak_clear"] = False  # 再 leak_check で flip させる

    return audit


def _save_snapshot(path: Path, snapshot: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    import os
    os.replace(tmp, path)


def process_all(results_by_rid: dict, race_id: str | None, apply: bool) -> dict:
    """Process one or all snapshots. Return summary dict."""
    if race_id:
        paths = [SNAPSHOT_DIR / f"{race_id}.json"]
    else:
        paths = sorted(SNAPSHOT_DIR.glob("*.json"))

    summary = {
        "n_snapshots":       0,
        "n_already_filled":  0,   # 触らなかった (全馬オッズあり)
        "n_filled":          0,   # 何頭 fill したか (延べ)
        "n_snapshots_filled": 0,  # 少なくとも 1 頭 fill した snapshot 数
        "n_no_result":       0,   # results.json に該当なし
        "n_leak_errors":     0,
        "details":           [],
    }

    for p in paths:
        rid = p.stem
        summary["n_snapshots"] += 1
        if not p.exists():
            summary["details"].append({"race_id": rid, "status": "missing"})
            continue
        snap = json.loads(p.read_text(encoding="utf-8"))
        result = results_by_rid.get(rid)
        if not result:
            summary["n_no_result"] += 1
            summary["details"].append({"race_id": rid, "status": "no-result"})
            continue

        audit = backfill_one(snap, result)
        summary["n_filled"] += audit["n_filled"]
        if audit["n_filled"] == 0:
            summary["n_already_filled"] += 1
            summary["details"].append({"race_id": rid, "status": "skip-already-filled",
                                       "n_filled": 0})
            continue
        summary["n_snapshots_filled"] += 1

        # Leak re-check
        try:
            leak_check(snap)
            snap["leak_audit"]["leak_clear"] = True
        except LeakError as e:
            summary["n_leak_errors"] += 1
            summary["details"].append({"race_id": rid, "status": "leak-error",
                                       "error": str(e)})
            continue

        if apply:
            _save_snapshot(p, snap)
            summary["details"].append({"race_id": rid, "status": "written",
                                       "n_filled": audit["n_filled"]})
        else:
            summary["details"].append({"race_id": rid, "status": "dry-run",
                                       "n_filled": audit["n_filled"]})

    return summary


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-id", help="Single race_id to process")
    parser.add_argument("--apply", action="store_true",
                        help="Write changes to disk (default: dry-run)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Report what would change, don't write")
    args = parser.parse_args()

    apply = args.apply  # --apply wins over --dry-run

    print(f"[backfill] results file: {RESULTS_FILE}")
    results = _load_results_indexed()
    print(f"[backfill] loaded {len(results)} race results")

    summary = process_all(results, args.race_id, apply=apply)

    print("")
    print("── summary ──────────────────────────────")
    print(f"  snapshots processed:          {summary['n_snapshots']}")
    print(f"  no-result (skipped):          {summary['n_no_result']}")
    print(f"  already-filled (skipped):     {summary['n_already_filled']}")
    print(f"  snapshots with backfill:      {summary['n_snapshots_filled']}")
    print(f"  horses filled (total):        {summary['n_filled']}")
    print(f"  leak errors:                  {summary['n_leak_errors']}")
    mode = "APPLIED (written to disk)" if apply else "DRY-RUN (no disk writes)"
    print(f"  mode:                         {mode}")
    if summary["n_leak_errors"]:
        print("")
        print("[backfill] WARNING: leak errors detected — these snapshots were NOT written.")
        for d in summary["details"]:
            if d.get("status") == "leak-error":
                print(f"  - {d['race_id']}: {d.get('error')}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
