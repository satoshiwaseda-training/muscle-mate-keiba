"""Offline tuning harness — re-ranks every saved bt_ row from
structured_features, sweeps α/β, the softmax temperature, and the
diversity constraint, and reports Top-3 / ROI / override metrics.

No network calls. Works directly on data/predictions.json +
data/results.json.

Usage:
    python tune_probability.py                  # full sweep
    python tune_probability.py --subset clean   # only NEW ranker rows
    python tune_probability.py --subset all
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import probability_engine as pe
from train import score_runner
from data_store import load_weights
import feature_store as fs

ROOT = Path(__file__).parent
PRED = ROOT / "data" / "predictions.json"
RES = ROOT / "data" / "results.json"


def _po(s) -> float:
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _norm(n: str) -> str:
    return (n or "").strip()


def load_dataset(subset: str) -> list[dict]:
    """Return a list of (race_id, entries_with_features, race_info, result)
    rows, already scratch-filtered."""
    preds = json.loads(PRED.read_text(encoding="utf-8")) if PRED.exists() else {}
    results = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {}

    bt = [k for k in preds if k.startswith("bt_") and k in results]
    if subset == "clean":
        bt = [k for k in bt if preds[k].get("_ranking_meta")]
    elif subset == "legacy":
        bt = [k for k in bt if not preds[k].get("_ranking_meta")]

    rows = []
    for k in bt:
        p, r = preds[k], results[k]
        sf = p.get("structured_features") or {}
        horses_sf = sf.get("horses") or {}
        fo = r.get("finishing_order") or []
        if not horses_sf or not fo:
            continue
        # Scratch filter: drop horses not in finishing_order
        fo_names = {_norm(h.get("name")) for h in fo}
        entries = []
        for name, h in horses_sf.items():
            if _norm(name) not in fo_names:
                continue
            entries.append({
                "name": _norm(name),
                "odds": (h.get("odds") or 0),
                **h,
            })
        if not entries:
            continue
        winner = next((_norm(h.get("name")) for h in fo if int(h.get("rank", 0) or 0) == 1), None)
        second = next((_norm(h.get("name")) for h in fo if int(h.get("rank", 0) or 0) == 2), None)
        if not winner:
            continue
        odds_map = {_norm(h.get("name")): _po(h.get("odds"))
                    for h in fo if _po(h.get("odds")) > 0}
        if not odds_map:
            continue
        rows.append({
            "race_id": k,
            "entries": entries,
            "race": sf.get("race") or {},
            "grade": p.get("grade", ""),
            "winner": winner,
            "second": second,
            "odds_map": odds_map,
            "has_meta": bool(p.get("_ranking_meta")),
        })
    return rows


def rescore_row(row: dict) -> list[dict]:
    """Call score_runner once per horse, rotating into slot 0. Returns
    scored list with name/odds/score."""
    ctx = {"weights": load_weights()}
    entries = row["entries"]
    sf = {"race": row["race"], "horses": {e["name"]: e for e in entries}, "version": 1}
    grade = row["grade"]
    scored = []
    for horse in entries:
        this_first = [{
            "name": horse["name"], "rank": 1, "odds": horse["odds"],
            "confidence": 0, "ev_gap": 0, "bet": "",
        }]
        for other in entries:
            if other is horse:
                continue
            this_first.append({
                "name": other["name"], "rank": 2, "odds": other["odds"],
                "confidence": 0, "ev_gap": 0, "bet": "",
            })
        feat = {
            "grade": grade,
            "num_horses": len(entries),
            "horse_features": this_first,
            "structured_features": sf,
        }
        s = score_runner(feat, ctx).get("top_confidence", 50.0)
        scored.append({
            "name": horse["name"],
            "odds": horse["odds"],
            "score": float(s),
        })
    return scored


def eval_config(
    rows: list[dict],
    alpha: float,
    beta: float,
    temperature: float,
    enforce_diversity: bool,
    edge_threshold: float,
    coverage_threshold: float,
    feature_coverage_override: float,
    label: str,
) -> dict:
    """Run the full pipeline on every row under one parameter set."""
    N = 0
    m_top1 = o_top1 = top3_winner = top3_pair = 0
    m_pnl = o_pnl = 0.0
    deviations = deviations_won = 0
    override_fires = override_hits = 0
    override_pnl = 0.0
    diversity_fallbacks = 0

    for row in rows:
        scored = rescore_row(row)
        ranked = pe.assign_win_probs(scored, temperature=temperature)
        if not ranked:
            continue
        sel = pe.select_top3(
            ranked, alpha=alpha, beta=beta,
            enforce_diversity=enforce_diversity,
        )
        if sel.get("diversity_fallback"):
            diversity_fallbacks += 1

        # Identify odds favorite and model top
        running = [r for r in ranked if r["odds"] > 0]
        if not running:
            continue
        odds_fav = min(running, key=lambda r: r["odds"])
        model_top = sel["selected"][0] if sel["selected"] else running[0]

        N += 1
        winner = row["winner"]
        second = row["second"]

        if model_top["name"] == winner: m_top1 += 1
        if odds_fav["name"] == winner: o_top1 += 1
        sel_names = {h["name"] for h in sel["selected"]}
        if winner in sel_names: top3_winner += 1
        if winner in sel_names and second and second in sel_names: top3_pair += 1

        # ROI — bet 1 unit on model_top vs odds_fav
        m_pnl += (row["odds_map"].get(model_top["name"], 0) - 1) if model_top["name"] == winner else -1
        o_pnl += (row["odds_map"].get(odds_fav["name"], 0) - 1) if odds_fav["name"] == winner else -1

        # Deviations (model top != odds favorite by name)
        if model_top["name"] != odds_fav["name"]:
            deviations += 1
            if model_top["name"] == winner:
                deviations_won += 1

            # Override gate
            allowed, _ = pe.should_override_market(
                selected_top=model_top,
                odds_favorite=odds_fav,
                feature_coverage=feature_coverage_override,
                coverage_threshold=coverage_threshold,
                edge_threshold=edge_threshold,
            )
            if allowed:
                override_fires += 1
                if model_top["name"] == winner:
                    override_hits += 1
                override_pnl += (row["odds_map"].get(model_top["name"], 0) - 1) if model_top["name"] == winner else -1

    if N == 0:
        return {"label": label, "N": 0}
    return {
        "label": label,
        "N": N,
        "model_top1": m_top1 / N,
        "odds_top1": o_top1 / N,
        "top3_winner": top3_winner / N,
        "top3_pair": top3_pair / N,
        "model_roi": m_pnl / N,
        "odds_roi": o_pnl / N,
        "deviation_rate": deviations / N,
        "deviation_win_rate": (deviations_won / deviations) if deviations else 0.0,
        "override_fires": override_fires,
        "override_win_rate": (override_hits / override_fires) if override_fires else 0.0,
        "override_roi_mean": (override_pnl / override_fires) if override_fires else 0.0,
        "diversity_fallbacks": diversity_fallbacks,
    }


def fmt_row(r: dict) -> str:
    if not r.get("N"):
        return f'{r["label"]:32}: (empty)'
    return (f'{r["label"]:32}: '
            f'n={r["N"]:3d} '
            f'T1 m={r["model_top1"]*100:5.2f}%/o={r["odds_top1"]*100:5.2f}% '
            f'T3W={r["top3_winner"]*100:5.2f}% '
            f'T3P={r["top3_pair"]*100:5.2f}% '
            f'ROI m={r["model_roi"]*100:+6.2f}%/o={r["odds_roi"]*100:+6.2f}% '
            f'dev={r["deviation_rate"]*100:4.1f}%'
            f'(w {r["deviation_win_rate"]*100:4.1f}%) '
            f'ovr={r["override_fires"]:2d}'
            f'(w {r["override_win_rate"]*100:4.1f}%,roi {r["override_roi_mean"]*100:+5.1f})')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["all", "clean", "legacy"], default="all")
    ap.add_argument("--coverage-probe", type=float, default=0.0,
                    help="Pretend feature coverage is this (0..1) to allow the "
                         "override gate to fire on current data (the saved bt rows "
                         "have real coverage ≈ 0).")
    args = ap.parse_args()

    print(f"Loading dataset (subset={args.subset})...")
    rows = load_dataset(args.subset)
    print(f"Loaded {len(rows)} scratch-filtered races\n")
    if not rows:
        return

    baseline = eval_config(
        rows, alpha=0.5, beta=0.5, temperature=12.0,
        enforce_diversity=False, edge_threshold=0.04,
        coverage_threshold=0.5,
        feature_coverage_override=args.coverage_probe,
        label="baseline α=.5 β=.5 nodiv",
    )
    print(fmt_row(baseline))

    print("\n--- α/β sweep (no diversity constraint) ---")
    sweep = []
    for a, b in [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]:
        r = eval_config(rows, alpha=a, beta=b, temperature=12.0,
                        enforce_diversity=False, edge_threshold=0.04,
                        coverage_threshold=0.5,
                        feature_coverage_override=args.coverage_probe,
                        label=f'α={a} β={b}')
        sweep.append(r)
        print(fmt_row(r))

    print("\n--- Diversity constraint ON (no-extreme-longshot, must-deviate) ---")
    for a, b in [(0.6, 0.4), (0.5, 0.5), (0.4, 0.6)]:
        r = eval_config(rows, alpha=a, beta=b, temperature=12.0,
                        enforce_diversity=True, edge_threshold=0.04,
                        coverage_threshold=0.5,
                        feature_coverage_override=args.coverage_probe,
                        label=f'α={a} β={b} +diversity')
        print(fmt_row(r))

    print("\n--- Override threshold: 0.04 vs 0.02 (with coverage probe) ---")
    for thr in (0.04, 0.02):
        r = eval_config(rows, alpha=0.5, beta=0.5, temperature=12.0,
                        enforce_diversity=False, edge_threshold=thr,
                        coverage_threshold=0.5,
                        feature_coverage_override=max(args.coverage_probe, 0.6),
                        label=f'edge_thr={thr} (cov probe=60%)')
        print(fmt_row(r))

    print("\n--- Softmax temperature sweep (α=0.5 β=0.5) ---")
    for T in (4, 8, 12, 18, 25):
        r = eval_config(rows, alpha=0.5, beta=0.5, temperature=T,
                        enforce_diversity=False, edge_threshold=0.04,
                        coverage_threshold=0.5,
                        feature_coverage_override=args.coverage_probe,
                        label=f'T={T}')
        print(fmt_row(r))


if __name__ == "__main__":
    main()
