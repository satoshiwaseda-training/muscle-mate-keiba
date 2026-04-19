"""TOP 3 予測の的中率・リグレット集計 (v5.5 — 2026-04-19).

皐月賞 2026 で TOP 3 が全滅した事態を受けて、過去レース全体で
モデル TOP 3 予測の的中率を集計し、基準値 (市場 TOP 3 / ランダム) と
比較する。憲法 §5.1「100 bets 以上で評価」の精神に従い、
1 レースの結果ではなく累積統計で判断するためのツール。

集計対象:
  - data/live_predictions.json                          (live の最新ログ)
  - data/backtest_predictions_v4_n221/*_on.json         (v4 backtest 221 本)
  - data/results.json                                    (答え合わせ)

評価指標:
  TOP1_hit_rate        model top-1 が 1 着だった割合
  TOP3_winner_rate     model top-3 に 1 着馬が含まれた割合
  TOP3_all_in_rate     model top-3 の 3 頭全てが実結果 1-3 着に入った割合
  TOP3_overlap_avg     model top-3 と 実結果 1-3 着の集合重複数の平均 (0-3)

基準:
  MARKET_TOP1_hit      市場 1 番人気 (odds 最小) が 1 着だった割合
  MARKET_TOP3_winner   市場 TOP 3 (odds 昇順) に 1 着馬が含まれた割合

リグレット:
  regret_win_prob      モデルが 1 着馬に割り当てた win_prob の平均。
                       0 に近いほど「1 着馬を低評価していた」

使い方:
  python tools/evaluate_top3.py                       # 両方集計
  python tools/evaluate_top3.py --source live         # ライブログのみ
  python tools/evaluate_top3.py --source backtest     # バックテストのみ
  python tools/evaluate_top3.py --grade G1            # G1 レースのみ

注: backtest_predictions_v4_n221 は snapshot odds が 0 だった時期に
生成されたもので、rankings は fact 層のみに依存していた可能性あり
(2026-04-18 の snapshot backfill 以前)。レポートに caveat 表示する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LIVE_FILE = ROOT / "data" / "live_predictions.json"
BACKTEST_DIR_V4 = ROOT / "data" / "backtest_predictions_v4_n221"
BACKTEST_DIR_V5 = ROOT / "data" / "backtest_predictions"        # v5 re-run output
BACKTEST_DIR = BACKTEST_DIR_V5 if BACKTEST_DIR_V5.exists() else BACKTEST_DIR_V4
RESULTS_FILE = ROOT / "data" / "results.json"


def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    raw = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    out: dict = {}
    for k, v in raw.items():
        rid = k[3:] if k.startswith("bt_") else k
        out[rid] = v
    return out


def _winner_name(result: dict) -> Optional[str]:
    for h in (result.get("finishing_order") or []):
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return (h.get("name") or "").strip()
        except (TypeError, ValueError):
            continue
    return None


def _top3_names(result: dict) -> list[str]:
    out = []
    for h in (result.get("finishing_order") or []):
        try:
            r = int(h.get("rank", 0) or 0)
        except (TypeError, ValueError):
            continue
        if 1 <= r <= 3:
            out.append((h.get("name") or "").strip())
    return out


def _parse_odds(raw) -> float:
    try:
        return float(str(raw).replace("---", "0").replace("--", "0").replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _market_top3_names(result: dict) -> list[str]:
    """Sort finishing_order by odds ascending, return top 3 (= 市場人気 TOP 3)."""
    fo = result.get("finishing_order") or []
    decorated = []
    for h in fo:
        od = _parse_odds(h.get("odds"))
        if od > 0:
            decorated.append((od, (h.get("name") or "").strip()))
    decorated.sort(key=lambda x: x[0])
    return [n for _, n in decorated[:3]]


# ──────────────────────────────────────────────────────
# Per-prediction evaluation
# ──────────────────────────────────────────────────────

def evaluate_one(pred: dict, result: dict) -> Optional[dict]:
    """Return evaluation record for a single prediction, or None if unusable."""
    ranked = pred.get("ranked") or []
    if len(ranked) < 3:
        return None
    winner = _winner_name(result)
    top3_actual = set(_top3_names(result))
    if not winner or len(top3_actual) < 1:
        return None

    model_top1 = (ranked[0].get("name") or "").strip()
    model_top3 = [(r.get("name") or "").strip() for r in ranked[:3]]
    model_top3_set = set(model_top3)

    overlap = len(model_top3_set & top3_actual)
    market_top3 = set(_market_top3_names(result))
    market_winner_in_top3 = (winner in market_top3) if market_top3 else None
    market_top1 = None
    if market_top3:
        fo = result.get("finishing_order") or []
        decorated = [((_parse_odds(h.get("odds")), (h.get("name") or "").strip()))
                     for h in fo]
        decorated = [(o, n) for o, n in decorated if o > 0]
        if decorated:
            decorated.sort(key=lambda x: x[0])
            market_top1 = decorated[0][1]

    # Winner's win_prob in model → regret signal
    winner_prob = 0.0
    for r in ranked:
        if (r.get("name") or "").strip() == winner:
            winner_prob = float(r.get("win_prob", 0) or 0)
            break

    return {
        "race_id": pred.get("race_id", ""),
        "race_name": pred.get("race_name", ""),
        "grade": pred.get("grade", ""),
        "race_date": pred.get("race_date", ""),
        "winner": winner,
        "model_top1": model_top1,
        "model_top3": model_top3,
        "actual_top3": sorted(top3_actual),
        "market_top1": market_top1,
        "market_top3": sorted(market_top3),
        "winner_prob_in_model": winner_prob,
        "top1_hit": model_top1 == winner,
        "top3_winner_in_model": winner in model_top3_set,
        "top3_all_in": overlap == 3,
        "top3_overlap": overlap,
        "market_top1_hit": (market_top1 == winner) if market_top1 else None,
        "market_top3_winner": market_winner_in_top3,
    }


# ──────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    n = len(records)
    top1_hits = sum(1 for r in records if r["top1_hit"])
    top3_winner = sum(1 for r in records if r["top3_winner_in_model"])
    top3_all_in = sum(1 for r in records if r["top3_all_in"])
    overlap_sum = sum(r["top3_overlap"] for r in records)
    winner_prob_sum = sum(r["winner_prob_in_model"] for r in records)

    market_records = [r for r in records if r["market_top1_hit"] is not None]
    market_top1_hits = sum(1 for r in market_records if r["market_top1_hit"])
    market_top3_winner = sum(1 for r in market_records if r["market_top3_winner"])
    m_n = len(market_records)

    grade_counter: Counter = Counter(
        (r.get("grade", "") or "(no-grade)") for r in records
    )

    return {
        "n": n,
        "top1_hit_rate": round(top1_hits / n, 4),
        "top3_winner_rate": round(top3_winner / n, 4),
        "top3_all_in_rate": round(top3_all_in / n, 4),
        "top3_overlap_avg": round(overlap_sum / n, 4),
        "winner_prob_avg": round(winner_prob_sum / n, 4),
        "market_top1_hit_rate": round(market_top1_hits / m_n, 4) if m_n else None,
        "market_top3_winner_rate": round(market_top3_winner / m_n, 4) if m_n else None,
        "market_n": m_n,
        "grade_counts": dict(grade_counter),
    }


# ──────────────────────────────────────────────────────
# Data loaders
# ──────────────────────────────────────────────────────

def load_live_predictions() -> list[dict]:
    if not LIVE_FILE.exists():
        return []
    data = json.loads(LIVE_FILE.read_text(encoding="utf-8"))
    return list(data.values()) if isinstance(data, dict) else []


def load_backtest_predictions() -> list[dict]:
    if not BACKTEST_DIR.exists():
        return []
    out = []
    for p in sorted(BACKTEST_DIR.glob("*_on.json")):
        try:
            pred = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append(pred)
    return out


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def _print_summary(label: str, agg: dict) -> None:
    print(f"\n── {label} ──")
    if agg.get("n", 0) == 0:
        print("  no usable records")
        return
    print(f"  races evaluated:           {agg['n']}")
    print(f"  grade distribution:        {agg['grade_counts']}")
    print("")
    print(f"  MODEL top-1 hit rate:      {agg['top1_hit_rate']*100:.1f}% "
          f"(1 着馬 = モデル 本命)")
    print(f"  MODEL top-3 winner rate:   {agg['top3_winner_rate']*100:.1f}% "
          f"(1 着馬 ∈ モデル TOP3)")
    print(f"  MODEL top-3 all-in rate:   {agg['top3_all_in_rate']*100:.1f}% "
          f"(モデル TOP3 = 実結果 TOP3 完全一致)")
    print(f"  MODEL top-3 overlap avg:   {agg['top3_overlap_avg']:.2f} / 3 頭")
    print(f"  winner win_prob (model):   {agg['winner_prob_avg']*100:.1f}%")
    print("")
    if agg.get("market_top1_hit_rate") is not None:
        print(f"  MARKET top-1 hit rate:     {agg['market_top1_hit_rate']*100:.1f}% "
              f"(市場 1 番人気 = 1 着)")
        print(f"  MARKET top-3 winner rate:  {agg['market_top3_winner_rate']*100:.1f}%")
        print(f"  market_n:                  {agg['market_n']}")
        # 差分
        dt1 = agg['top1_hit_rate'] - agg['market_top1_hit_rate']
        dt3 = agg['top3_winner_rate'] - agg['market_top3_winner_rate']
        print("")
        print(f"  MODEL vs MARKET top-1:     {dt1*100:+.1f} pp")
        print(f"  MODEL vs MARKET top-3:     {dt3*100:+.1f} pp")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("live", "backtest", "both"),
                        default="both")
    parser.add_argument("--grade", help="grade フィルタ (例: G1 / G2 / 部分一致)")
    parser.add_argument("--verbose", action="store_true",
                        help="per-race rows をダンプ")
    args = parser.parse_args()

    results = _load_results()
    print(f"[eval] results loaded: {len(results)} races")

    all_live = load_live_predictions() if args.source in ("live", "both") else []
    all_bt = load_backtest_predictions() if args.source in ("backtest", "both") else []

    def _filter_grade(preds):
        if not args.grade:
            return preds
        return [p for p in preds if args.grade in (p.get("grade", "") or "")]

    live_preds = _filter_grade(all_live)
    bt_preds = _filter_grade(all_bt)

    # Evaluate each
    def _eval_list(preds, label):
        records = []
        for p in preds:
            rid = p.get("race_id", "")
            res = results.get(rid)
            if not res:
                continue
            rec = evaluate_one(p, res)
            if rec:
                records.append(rec)
        print(f"[eval] {label}: {len(records)}/{len(preds)} usable (有 result & TOP3)")
        return records

    live_records = _eval_list(live_preds, "live")
    bt_records = _eval_list(bt_preds, "backtest")

    if args.verbose:
        print("\n── per-race rows (live) ──")
        for r in live_records:
            print(f"  {r['race_id']} {r['race_name']:<30s} "
                  f"winner={r['winner']:<12s} model_top1={r['model_top1']:<12s} "
                  f"overlap={r['top3_overlap']}")
        print("\n── per-race rows (backtest) [first 20] ──")
        for r in bt_records[:20]:
            print(f"  {r['race_id']} {r['race_name']:<30s} "
                  f"winner={r['winner']:<12s} model_top1={r['model_top1']:<12s} "
                  f"overlap={r['top3_overlap']}")

    if live_records:
        _print_summary("LIVE predictions", aggregate(live_records))
    if bt_records:
        _print_summary("BACKTEST predictions (v4 n=221)", aggregate(bt_records))
        print("")
        print("  ⚠️ CAVEAT: backtest_predictions_v4_n221 は snapshot odds が")
        print("     backfill される前に生成された予測。rankings が market_prob の")
        print("     uniform 1/N に引きずられる可能性あり。")
        print("     再 backtest 実行後の再集計を強く推奨。")

    # Global
    combined = live_records + bt_records
    if combined:
        _print_summary("COMBINED (live + backtest)", aggregate(combined))

    return 0


if __name__ == "__main__":
    sys.exit(main())
