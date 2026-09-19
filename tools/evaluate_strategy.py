"""Evaluate user's specific betting strategy:

  レースごとに以下を買う:
    - 単勝: モデル TOP 3 の 3 頭 → 100 円 × 3 = 300 円
    - 馬連: モデル TOP 3 の 3 頭 C(3,2)=3 組 → 100 円 × 3 = 300 円
  合計 600 円/レース

このスクリプトは過去レース (live_predictions + backtest_predictions) を
走査して、このポリシーでの ROI / 的中率を集計する。憲法 §5.1 の
「100 bet 以上で評価」に合わせ、累積で判断する。

使い方:
  python tools/evaluate_strategy.py                 # すべて
  python tools/evaluate_strategy.py --grade G1      # G1 のみ
  python tools/evaluate_strategy.py --source live   # live のみ
  python tools/evaluate_strategy.py --verbose       # per-race 明細
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
LIVE_FILE = ROOT / "data" / "live_predictions.json"
BACKTEST_DIR_V4 = ROOT / "data" / "backtest_predictions_v4_n221"
BACKTEST_DIR_V5 = ROOT / "data" / "backtest_predictions"
BACKTEST_DIR = BACKTEST_DIR_V5 if BACKTEST_DIR_V5.exists() else BACKTEST_DIR_V4
RESULTS_FILE = ROOT / "data" / "results.json"


# ─── Strategy parameters ────────────────────────────────
BET_AMOUNT_PER_TICKET = 100.0     # 円/券
N_TOP_HORSES         = 3
N_TANSHO_TICKETS     = 3          # モデル TOP 3 の 3 頭
N_UMAREN_TICKETS     = 3          # 3 頭から 3 組
COST_PER_RACE        = (N_TANSHO_TICKETS + N_UMAREN_TICKETS) * BET_AMOUNT_PER_TICKET
# = 600 円/レース


# ──────────────────────────────────────────────────────
# Loaders
# ──────────────────────────────────────────────────────

def _load_results() -> dict:
    if not RESULTS_FILE.exists():
        return {}
    raw = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    out: dict = {}
    for k, v in raw.items():
        rid = k[3:] if k.startswith("bt_") else k
        out[rid] = v
    return out


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
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _rank_map(result: dict) -> dict:
    """Return {horse_name: rank_int} for finishing_order."""
    m: dict[str, int] = {}
    for h in (result.get("finishing_order") or []):
        try:
            r = int(h.get("rank", 0) or 0)
        except (TypeError, ValueError):
            continue
        nm = (h.get("name") or "").strip()
        if nm and r > 0:
            m[nm] = r
    return m


def _payout_tansho(result: dict) -> float:
    """Return the 単勝 payout (単勝 配当). 0 if not available."""
    try:
        return float((result.get("payouts") or {}).get("単勝", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


def _payout_umaren(result: dict) -> float:
    """Return the 馬連 payout (for the actual 1-2 pair)."""
    try:
        return float((result.get("payouts") or {}).get("馬連", 0) or 0)
    except (TypeError, ValueError):
        return 0.0


# ──────────────────────────────────────────────────────
# Per-race evaluation
# ──────────────────────────────────────────────────────

def evaluate_one(pred: dict, result: dict) -> Optional[dict]:
    ranked = pred.get("ranked") or []
    if len(ranked) < N_TOP_HORSES:
        return None

    rank_map = _rank_map(result)
    if not rank_map:
        return None

    top3_model = [(r.get("name") or "").strip() for r in ranked[:N_TOP_HORSES]]
    if any(not n for n in top3_model):
        return None
    top3_set = set(top3_model)

    # ── 単勝: 3 枚買う (1 枚 100 円), 勝ち馬が TOP3 にいれば 1 枚当たり ──
    winner = None
    for nm, r in rank_map.items():
        if r == 1:
            winner = nm
            break
    if winner is None:
        return None

    tansho_payout = 0.0
    tansho_hit = False
    if winner in top3_set:
        tansho_payout = _payout_tansho(result)
        tansho_hit = True

    # ── 馬連: 3 組買う, 実際の 1-2 着ペアが買い目に含まれれば 1 枚当たり ──
    second = None
    for nm, r in rank_map.items():
        if r == 2:
            second = nm
            break

    umaren_payout = 0.0
    umaren_hit = False
    if winner and second:
        actual_pair = frozenset([winner, second])
        # Our 3 combinations:
        my_combos = {frozenset(c) for c in combinations(top3_model, 2)}
        if actual_pair in my_combos:
            umaren_payout = _payout_umaren(result)
            umaren_hit = True

    cost = COST_PER_RACE
    payout = tansho_payout + umaren_payout
    pnl = payout - cost

    return {
        "race_id":    pred.get("race_id", ""),
        "race_name":  pred.get("race_name", ""),
        "grade":      pred.get("grade", "") or "",
        "race_date":  pred.get("race_date", ""),
        "model_top3": top3_model,
        "winner":     winner,
        "second":     second,
        "tansho_hit": tansho_hit,
        "tansho_payout": tansho_payout,
        "umaren_hit": umaren_hit,
        "umaren_payout": umaren_payout,
        "cost":       cost,
        "payout":     payout,
        "pnl":        pnl,
    }


# ──────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────

def aggregate(records: list[dict]) -> dict:
    if not records:
        return {"n": 0}
    n = len(records)
    tansho_hits = sum(1 for r in records if r["tansho_hit"])
    umaren_hits = sum(1 for r in records if r["umaren_hit"])
    any_hit     = sum(1 for r in records if r["tansho_hit"] or r["umaren_hit"])
    cost_total   = sum(r["cost"] for r in records)
    payout_total = sum(r["payout"] for r in records)
    pnl_total    = payout_total - cost_total

    # Top hits in PNL
    hits = [r for r in records if r["payout"] > 0]
    hits.sort(key=lambda r: r["pnl"], reverse=True)

    # Losses (no pay)
    losses = [r for r in records if r["payout"] == 0]

    return {
        "n":                  n,
        "any_hit_rate":       round(any_hit / n, 4),
        "tansho_hit_rate":    round(tansho_hits / n, 4),
        "umaren_hit_rate":    round(umaren_hits / n, 4),
        "cost_total":         cost_total,
        "payout_total":       payout_total,
        "pnl_total":          pnl_total,
        "roi":                round(pnl_total / cost_total, 4) if cost_total else 0.0,
        "avg_payout_hit":     round(sum(h["payout"] for h in hits) / len(hits), 1)
                                if hits else 0.0,
        "big_wins":           [{"race": h["race_name"],
                                "grade": h["grade"],
                                "top3": h["model_top3"],
                                "payout": h["payout"],
                                "pnl": h["pnl"]} for h in hits[:5]],
        "n_zero_return_races": len(losses),
    }


def aggregate_by_grade(records: list[dict]) -> dict:
    by_grade: dict[str, list] = defaultdict(list)
    for r in records:
        g = r["grade"] or "(no-grade)"
        # Normalize common variants
        g_norm = g.upper().replace("JPN", "JPN")
        if "G1" in g_norm or "GI" == g_norm or "(G1)" in g:
            key = "G1"
        elif "G2" in g_norm or "GII" == g_norm:
            key = "G2"
        elif "G3" in g_norm or "GIII" == g_norm:
            key = "G3"
        else:
            key = "OTHER"
        by_grade[key].append(r)

    out = {}
    for g, rs in by_grade.items():
        out[g] = aggregate(rs)
    return out


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def _print_agg(label: str, agg: dict) -> None:
    print(f"\n── {label} ──")
    if agg.get("n", 0) == 0:
        print("  no records")
        return
    n = agg["n"]
    print(f"  races:                     {n}")
    print(f"  any_hit (単勝 or 馬連):      {agg['any_hit_rate']*100:.1f}%")
    print(f"  単勝 hit rate:              {agg['tansho_hit_rate']*100:.1f}%  "
          f"(勝ち馬が TOP3)")
    print(f"  馬連 hit rate:              {agg['umaren_hit_rate']*100:.1f}%  "
          f"(1-2 着が両方 TOP3)")
    print(f"  cost total:                {agg['cost_total']:,.0f} 円")
    print(f"  payout total:              {agg['payout_total']:,.0f} 円")
    print(f"  pnl total:                 {agg['pnl_total']:+,.0f} 円")
    print(f"  ROI:                       {agg['roi']*100:+.1f}%")
    print(f"  平均 payout (当たり時):     {agg['avg_payout_hit']:,.0f} 円")
    if agg.get("big_wins"):
        print(f"  big wins (top 5 by pnl):")
        for w in agg["big_wins"]:
            top3 = " / ".join(w["top3"])
            print(f"    {w['grade']:<4s} {w['race'][:30]:<30s} top3=[{top3}] "
                  f"payout={w['payout']:>8,.0f}円 pnl={w['pnl']:>+7,.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("live", "backtest", "both"),
                        default="both")
    parser.add_argument("--grade")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    results = _load_results()
    print(f"[strategy] results loaded: {len(results)} races")
    print(f"[strategy] bet policy: 単勝×3 ({N_TANSHO_TICKETS*100}円) + 馬連×3 "
          f"({N_UMAREN_TICKETS*100}円) = {COST_PER_RACE:,.0f}円/race")

    all_preds = []
    if args.source in ("live", "both"):
        all_preds.extend(load_live_predictions())
    if args.source in ("backtest", "both"):
        all_preds.extend(load_backtest_predictions())

    if args.grade:
        all_preds = [p for p in all_preds if args.grade in (p.get("grade", "") or "")]

    records = []
    for p in all_preds:
        res = results.get(p.get("race_id", ""))
        if not res:
            continue
        rec = evaluate_one(p, res)
        if rec:
            records.append(rec)
    print(f"[strategy] usable records: {len(records)}")

    if args.verbose:
        print("\n── per-race (sample) ──")
        for r in records[:20]:
            print(f"  {r['race_id']} {r['race_name'][:25]:<25s} "
                  f"winner={r['winner'][:10]:<10s} "
                  f"tansho={'✓' if r['tansho_hit'] else '·'} "
                  f"({r['tansho_payout']:>5,.0f}) "
                  f"umaren={'✓' if r['umaren_hit'] else '·'} "
                  f"({r['umaren_payout']:>6,.0f}) "
                  f"pnl={r['pnl']:>+6,.0f}")

    _print_agg("ALL (single pool)", aggregate(records))

    by_g = aggregate_by_grade(records)
    for g in ("G1", "G2", "G3", "OTHER"):
        if g in by_g:
            _print_agg(f"by grade: {g}", by_g[g])

    return 0


if __name__ == "__main__":
    sys.exit(main())
