"""Recent performance review for TOP-3 betting strategies.

Default window: trailing 90 days from 2026-05-08, matching the current
operational review request. The tool only uses already-known predictions
and attached results, so it does not leak future data into a live race.

Outputs:
  - console summary
  - data/recent_performance_review.json

This is an analysis/reporting tool. It does not change the model, loose
trigger rule, or probability engine.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import grade_strategy as gs

RESULTS_FILE = ROOT / "data" / "results.json"
PREDICTION_DIR = ROOT / "data" / "backtest_predictions"
OUT_FILE = ROOT / "data" / "recent_performance_review.json"


def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    raw = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return {(k[3:] if k.startswith("bt_") else k): v for k, v in raw.items()}


def _parse_date(value: object) -> dt.date | None:
    raw = str(value or "")[:10]
    try:
        return dt.date.fromisoformat(raw)
    except ValueError:
        return None


def _result_date(result: dict) -> dt.date | None:
    return _parse_date(result.get("race_date") or result.get("timestamp"))


def _grade_bucket(grade: str) -> str:
    g = (grade or "").upper()
    if "G1" in g or g == "GI":
        return "G1"
    if "G2" in g or g == "GII":
        return "G2"
    if "G3" in g or g == "GIII":
        return "G3"
    return "OTHER"


def _rank_map(result: dict) -> dict[str, int]:
    out = {}
    for h in result.get("finishing_order") or []:
        try:
            rank = int(h.get("rank", 0) or 0)
        except (TypeError, ValueError):
            continue
        name = (h.get("name") or "").strip()
        if name and rank > 0:
            out[name] = rank
    return out


def _payout(result: dict, bet_type: str) -> float:
    try:
        return float((result.get("payouts") or {}).get(bet_type, 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _pick_top3(prediction: dict, strategy: str) -> list[str]:
    ranked = prediction.get("ranked") or []
    if len(ranked) < 3:
        return []
    if strategy == "win_prob":
        return [(h.get("name") or "").strip() for h in ranked[:3]]
    market_rank = gs.build_market_rank_map(ranked)
    picked = gs.pick_diversified_top3(ranked, market_rank, strategy=strategy)
    return [(h.get("name") or "").strip() for h in picked[:3]]


def _simulate_race(prediction: dict, result: dict, strategy: str, mode: str) -> dict | None:
    top3 = _pick_top3(prediction, strategy)
    if len([n for n in top3 if n]) < 3:
        return None

    ranks = _rank_map(result)
    winner = next((name for name, rank in ranks.items() if rank == 1), None)
    second = next((name for name, rank in ranks.items() if rank == 2), None)
    third = next((name for name, rank in ranks.items() if rank == 3), None)
    if not winner:
        return None

    top3_set = set(top3)
    selected_pairs = {frozenset(pair) for pair in combinations(top3, 2)}
    actual_top3 = [name for name in (winner, second, third) if name]
    actual_wide_pairs = {frozenset(pair) for pair in combinations(actual_top3, 2)}

    if mode == "umaren":
        cost = 300.0
        hit = bool(second and frozenset((winner, second)) in selected_pairs)
        payout = _payout(result, "馬連") if hit else 0.0
    elif mode == "tansho_umaren":
        cost = 600.0
        win_hit = winner in top3_set
        umaren_hit = bool(second and frozenset((winner, second)) in selected_pairs)
        payout = (_payout(result, "単勝") if win_hit else 0.0)
        payout += _payout(result, "馬連") if umaren_hit else 0.0
        hit = win_hit or umaren_hit
    elif mode == "umaren_wide":
        cost = 600.0
        umaren_hit = bool(second and frozenset((winner, second)) in selected_pairs)
        wide_hit = bool(selected_pairs & actual_wide_pairs)
        payout = (_payout(result, "馬連") if umaren_hit else 0.0)
        payout += _payout(result, "ワイド") if wide_hit else 0.0
        hit = umaren_hit or wide_hit
    else:
        raise ValueError(f"unknown mode: {mode}")

    top2_pair_hit = bool(second and frozenset((winner, second)) in selected_pairs)
    return {
        "race_id": prediction.get("race_id", ""),
        "race_name": prediction.get("race_name", ""),
        "grade": prediction.get("grade", ""),
        "top3": top3,
        "actual_top3": [winner, second, third],
        "cost": cost,
        "payout": payout,
        "pnl": payout - cost,
        "hit": hit,
        "top3_winner_hit": winner in top3_set,
        "top2_pair_hit": top2_pair_hit,
    }


def _aggregate(records: list[dict]) -> dict:
    if not records:
        return {
            "n": 0, "cost": 0.0, "payout": 0.0, "pnl": 0.0,
            "roi": 0.0, "hit_rate": 0.0, "top3_winner_rate": 0.0,
            "top2_pair_rate": 0.0,
        }
    n = len(records)
    cost = sum(r["cost"] for r in records)
    payout = sum(r["payout"] for r in records)
    pnl = payout - cost
    return {
        "n": n,
        "cost": round(cost, 1),
        "payout": round(payout, 1),
        "pnl": round(pnl, 1),
        "roi": round(pnl / cost if cost else 0.0, 4),
        "hit_rate": round(sum(1 for r in records if r["hit"]) / n, 4),
        "top3_winner_rate": round(sum(1 for r in records if r["top3_winner_hit"]) / n, 4),
        "top2_pair_rate": round(sum(1 for r in records if r["top2_pair_hit"]) / n, 4),
    }


def _load_window(start: dt.date, end: dt.date) -> list[tuple[dict, dict, dt.date]]:
    results = _load_results()
    rows = []
    for path in sorted(PREDICTION_DIR.glob("*_on.json")):
        try:
            prediction = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result = results.get(prediction.get("race_id", ""))
        if not result:
            continue
        race_date = _result_date(result)
        if race_date and start <= race_date <= end:
            rows.append((prediction, result, race_date))
    return rows


def build_review(start: dt.date, end: dt.date) -> dict:
    races = _load_window(start, end)
    strategies = ["win_prob"] + list(gs.STRATEGIES.keys())
    modes = ["umaren", "tansho_umaren", "umaren_wide"]
    grades = ["ALL", "G1", "G2", "G3"]

    review = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "race_count": len(races),
        "races": [
            {
                "race_date": d.isoformat(),
                "race_id": p.get("race_id", ""),
                "race_name": p.get("race_name", ""),
                "grade": p.get("grade", ""),
                "actual_top3": [
                    h.get("name", "")
                    for h in (r.get("finishing_order") or [])[:3]
                ],
                "umaren_payout": _payout(r, "馬連"),
            }
            for p, r, d in races
        ],
        "metrics": {},
    }

    for mode in modes:
        review["metrics"][mode] = {}
        for grade in grades:
            scoped = races if grade == "ALL" else [
                row for row in races if _grade_bucket(row[0].get("grade", "")) == grade
            ]
            table = {}
            for strategy in strategies:
                records = []
                for prediction, result, _ in scoped:
                    rec = _simulate_race(prediction, result, strategy, mode)
                    if rec:
                        records.append(rec)
                table[strategy] = _aggregate(records)
            ranked = sorted(
                table.items(),
                key=lambda item: (item[1]["roi"], item[1]["top2_pair_rate"]),
                reverse=True,
            )
            review["metrics"][mode][grade] = {
                "best_strategy": ranked[0][0] if ranked else "",
                "strategies": table,
            }

    return review


def print_review(review: dict) -> None:
    print(f"window: {review['window']['start']} to {review['window']['end']}")
    print(f"races with result: {review['race_count']}")
    for mode, by_grade in review["metrics"].items():
        print(f"\n== {mode} ==")
        for grade, payload in by_grade.items():
            print(f"  {grade}")
            rows = sorted(
                payload["strategies"].items(),
                key=lambda item: item[1]["roi"],
                reverse=True,
            )
            for strategy, agg in rows[:4]:
                print(
                    f"    {strategy:28} "
                    f"n={agg['n']:>2} ROI={agg['roi']*100:+7.1f}% "
                    f"hit={agg['hit_rate']*100:5.1f}% "
                    f"top2={agg['top2_pair_rate']*100:5.1f}% "
                    f"pnl={agg['pnl']:+.0f}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2026-02-08")
    parser.add_argument("--end", default="2026-05-08")
    parser.add_argument("--out", default=str(OUT_FILE))
    args = parser.parse_args()

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    review = build_review(start, end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8")
    print_review(review)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
