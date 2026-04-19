"""Top-3 / probability-aware evaluation metrics.

Separate from evaluator.py (which must stay untouched). Reads the same
data/predictions.json and data/results.json, but uses the new `theoretical`
block written by backtest.py / prediction_pipeline.py.

Metrics:
  - Top-1 accuracy (model top)
  - Top-3 winner coverage  (did the winner appear in selected_top3?)
  - Top-3 top2-pair coverage (did both 1st and 2nd come from selected_top3?)
  - ROI vs odds baseline (1-unit flat on model top)

Usage:
    python evaluation_extras.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent
PRED = ROOT / "data" / "predictions.json"
RES = ROOT / "data" / "results.json"


def _po(s) -> float:
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def evaluate() -> dict:
    preds = json.loads(PRED.read_text(encoding="utf-8")) if PRED.exists() else {}
    results = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {}

    bt_keys = sorted(k for k in preds if k.startswith("bt_") and k in results)

    n = m_top1 = o_top1 = top3_winner = top3_pair = 0
    m_pnl = o_pnl = 0.0
    theo_keys = 0

    for k in bt_keys:
        p, r = preds[k], results[k]
        horses = p.get("horses") or []
        fo = r.get("finishing_order") or []
        if not horses or not fo:
            continue
        winner = None
        second = None
        for h in fo:
            try:
                rk = int(h.get("rank", 0) or 0)
            except ValueError:
                continue
            name = (h.get("name") or "").strip()
            if rk == 1: winner = name
            elif rk == 2: second = name
            if winner and second: break
        if not winner:
            continue
        odds_map = {(h.get("name") or "").strip(): _po(h.get("odds"))
                    for h in fo if _po(h.get("odds")) > 0}
        if not odds_map:
            continue
        of = min(odds_map, key=lambda nm: odds_map[nm])

        model_top = (horses[0].get("name") or "").strip()
        theo = p.get("theoretical") or {}
        sel = theo.get("selected_top3") or [h.get("name") for h in horses[:3]]

        n += 1
        if theo: theo_keys += 1
        if model_top == winner: m_top1 += 1
        if of == winner: o_top1 += 1
        if winner in sel: top3_winner += 1
        if winner in sel and second and second in sel: top3_pair += 1

        m_pnl += (odds_map.get(model_top, 0) - 1) if model_top == winner else -1
        o_pnl += (odds_map.get(of, 0) - 1) if of == winner else -1

    if not n:
        return {"n": 0}

    return {
        "n": n,
        "rows_with_theoretical_block": theo_keys,
        "model_top1_acc": m_top1 / n,
        "odds_top1_acc": o_top1 / n,
        "top3_winner_coverage": top3_winner / n,
        "top3_pair_coverage": top3_pair / n,
        "model_roi": m_pnl / n,
        "odds_roi": o_pnl / n,
    }


def main():
    m = evaluate()
    if not m.get("n"):
        print("No paired backtest rows found.")
        return
    print(f"N                        : {m['n']}")
    print(f"Rows with theoretical{{}} : {m['rows_with_theoretical_block']}")
    print(f"Model  Top-1             : {m['model_top1_acc']*100:.2f}%")
    print(f"Odds   Top-1             : {m['odds_top1_acc']*100:.2f}%")
    print(f"Top-3  winner coverage   : {m['top3_winner_coverage']*100:.2f}%")
    print(f"Top-3  1-2 pair coverage : {m['top3_pair_coverage']*100:.2f}%")
    print(f"Model  ROI (flat 1-unit) : {m['model_roi']*100:+.2f}%")
    print(f"Odds   ROI (flat 1-unit) : {m['odds_roi']*100:+.2f}%")


if __name__ == "__main__":
    main()
