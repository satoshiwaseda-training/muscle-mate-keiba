"""Apply composite_condition_score to v2 data, re-rank, and report.

Reads:  data/enriched_backtest_results_v2.json
Writes: data/enriched_backtest_results_v3.json

Per horse:
  1. Compute composite_condition_score from v2 structured_features
  2. Store composite + per-dim breakdown in a new `composite` block
  3. Inject the composite into score_runner's paddock bio keys
     (only where paddock coverage was previously empty)
  4. Re-run score_runner + softmax + top-3 selection

Compare v2 vs v3 on:
  - Top-1, Top-3 winner, Top-3 pair
  - Deviation rate
  - Changed-race win rate
  - ROI
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import composite_features as cf
import probability_engine as pe
from train import score_runner
from data_store import load_weights

ROOT = Path(__file__).parent
V2 = ROOT / "data" / "enriched_backtest_results_v2.json"
V3 = ROOT / "data" / "enriched_backtest_results_v3.json"
RES = ROOT / "data" / "results.json"


def _norm(n): return (n or "").strip()
def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try: return float(s)
    except ValueError: return 0.0


def main():
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    results = json.loads(RES.read_text(encoding="utf-8"))
    ctx = {"weights": load_weights()}

    out = []
    composite_stats = {
        "horses_with_composite_signal": 0,
        "horses_with_3_dims": 0,
        "horses_with_2_dims": 0,
        "horses_with_1_dim": 0,
        "horses_with_0_dims": 0,
        "composite_mean": 0.0,
        "composite_count": 0,
        "bio_injected": 0,
    }

    for row in v2:
        sf = dict(row.get("structured_features") or {})
        race_sf = sf.get("race") or {}
        sf_horses = dict(sf.get("horses") or {})

        for name, h in list(sf_horses.items()):
            c = cf.composite_condition_score(h, race_sf)
            new_h = dict(h)
            new_h["composite_condition_score"] = round(c["composite"], 3)
            new_h["composite_n_dims"] = c["n_dims_used"]
            new_h["composite_has_signal"] = c["has_signal"]

            if c["has_signal"]:
                composite_stats["horses_with_composite_signal"] += 1
                composite_stats["composite_mean"] += c["composite"]
                composite_stats["composite_count"] += 1
            n_dims = c["n_dims_used"]
            if n_dims >= 3: composite_stats["horses_with_3_dims"] += 1
            elif n_dims == 2: composite_stats["horses_with_2_dims"] += 1
            elif n_dims == 1: composite_stats["horses_with_1_dim"] += 1
            else: composite_stats["horses_with_0_dims"] += 1

            # Inject into bio pathway only where paddock keys are empty
            before_v = new_h.get("paddock_vascularity", 0)
            before_h = new_h.get("paddock_hindquarter", 0)
            before_g = new_h.get("paddock_gait", 0)
            new_h = cf.inject_composite_into_bio(new_h, c["composite"], only_when_empty=True)
            if (new_h.get("paddock_vascularity") != before_v
                or new_h.get("paddock_hindquarter") != before_h
                or new_h.get("paddock_gait") != before_g):
                composite_stats["bio_injected"] += 1

            sf_horses[name] = new_h
        sf["horses"] = sf_horses

        # Re-run score_runner for all horses with the updated structured_features
        entries = list(sf_horses.keys())
        grade = row.get("grade", "")
        scored = []
        for this_name in entries:
            rotated = [{"name": this_name, "rank": 1,
                        "odds": sf_horses[this_name].get("odds", 0),
                        "confidence": 0, "ev_gap": 0, "bet": ""}]
            for other in entries:
                if other == this_name:
                    continue
                rotated.append({"name": other, "rank": 2,
                                "odds": sf_horses[other].get("odds", 0),
                                "confidence": 0, "ev_gap": 0, "bet": ""})
            feat = {"grade": grade, "num_horses": len(entries),
                    "horse_features": rotated, "structured_features": sf}
            s = score_runner(feat, ctx).get("top_confidence", 50.0)
            scored.append({
                "name": this_name,
                "odds": sf_horses[this_name].get("odds", 0),
                "score": float(s),
            })
        ranked = pe.assign_win_probs(scored, temperature=pe.DEFAULT_TEMPERATURE)
        sel = pe.select_top3(ranked, alpha=pe.DEFAULT_ALPHA, beta=pe.DEFAULT_BETA)

        out.append({
            "race_id": row["race_id"],
            "race_name": row.get("race_name", ""),
            "grade": grade,
            "ranked": ranked,
            "selected_top3": sel["selected"],
            "p1": sel["p1"], "p2": sel["p2"],
            "structured_features": sf,
        })

    V3.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out)} races to {V3.relative_to(ROOT)}")

    total = sum((composite_stats["horses_with_3_dims"],
                 composite_stats["horses_with_2_dims"],
                 composite_stats["horses_with_1_dim"],
                 composite_stats["horses_with_0_dims"]))
    mean = (composite_stats["composite_mean"] / composite_stats["composite_count"]
            if composite_stats["composite_count"] else 0.0)
    print("\n" + "=" * 72)
    print("COMPOSITE COVERAGE")
    print("=" * 72)
    print(f"  Total horses              : {total}")
    print(f"  With any composite signal : {composite_stats['horses_with_composite_signal']} "
          f"({composite_stats['horses_with_composite_signal']/total*100:.1f}%)")
    print(f"  With ≥3 dimensions used   : {composite_stats['horses_with_3_dims']}")
    print(f"  With 2 dimensions used    : {composite_stats['horses_with_2_dims']}")
    print(f"  With 1 dimension used     : {composite_stats['horses_with_1_dim']}")
    print(f"  With 0 dimensions (empty) : {composite_stats['horses_with_0_dims']}")
    print(f"  Mean composite value      : {mean:.3f}")
    print(f"  Horses where bio injected : {composite_stats['bio_injected']}")

    # Now compare v2 vs v3 metrics
    def metrics(rows_obj):
        by_rid = {r["race_id"]: r for r in rows_obj}
        N = m_hit = o_hit = t3w = t3p = dev = m_chg = 0
        m_pnl = o_pnl = 0.0
        for rid, r in by_rid.items():
            res = results.get(f"bt_{rid}")
            if not res:
                continue
            fo = res.get("finishing_order") or []
            winner = next((_norm(h.get("name")) for h in fo
                           if int(h.get("rank", 0) or 0) == 1), None)
            second = next((_norm(h.get("name")) for h in fo
                           if int(h.get("rank", 0) or 0) == 2), None)
            if not winner:
                continue
            odds_map = {}
            for h in fo:
                nm = _norm(h.get("name")); od = _po(h.get("odds"))
                if nm and od > 0 and nm not in odds_map:
                    odds_map[nm] = od
            if not odds_map:
                continue
            of = min(odds_map, key=lambda n: odds_map[n])
            mt = r["ranked"][0]["name"] if r["ranked"] else ""
            sel_names = {h["name"] for h in r["selected_top3"]}
            N += 1
            if mt == winner: m_hit += 1
            if of == winner: o_hit += 1
            if winner in sel_names: t3w += 1
            if winner in sel_names and second and second in sel_names: t3p += 1
            if mt != of:
                dev += 1
                if mt == winner: m_chg += 1
            m_pnl += (odds_map.get(mt, 0) - 1) if mt == winner else -1
            o_pnl += (odds_map.get(of, 0) - 1) if of == winner else -1
        return {
            "N": N, "model_top1": m_hit, "odds_top1": o_hit,
            "top3_w": t3w, "top3_p": t3p, "dev": dev, "chg_w": m_chg,
            "m_roi": m_pnl / N * 100 if N else 0,
            "o_roi": o_pnl / N * 100 if N else 0,
        }

    m_v2 = metrics(v2)
    m_v3 = metrics(out)
    print("\n" + "=" * 72)
    print("v2 vs v3 METRIC COMPARISON")
    print("=" * 72)
    print(f'{"metric":22} {"v2":>12} {"v3":>12} {"Δ":>10}')
    print("-" * 60)
    for key, label in [("model_top1", "Model Top-1"),
                       ("odds_top1", "Odds Top-1"),
                       ("top3_w", "Top-3 winner"),
                       ("top3_p", "Top-3 pair"),
                       ("dev", "Deviations"),
                       ("chg_w", "Changed-race wins"),
                       ("m_roi", "Model ROI (%)"),
                       ("o_roi", "Odds ROI (%)")]:
        v2v = m_v2[key]; v3v = m_v3[key]
        if isinstance(v2v, float):
            print(f'{label:22} {v2v:+12.2f} {v3v:+12.2f} {v3v-v2v:+10.2f}')
        else:
            print(f'{label:22} {v2v:>12} {v3v:>12} {v3v-v2v:>+10}')

    # Per-race diff in rankings
    changed_top1 = 0
    changed_sel3 = 0
    for v2r, v3r in zip(v2, out):
        t1_v2 = v2r["ranked"][0]["name"] if v2r["ranked"] else ""
        t1_v3 = v3r["ranked"][0]["name"] if v3r["ranked"] else ""
        if t1_v2 != t1_v3: changed_top1 += 1
        s2 = {h["name"] for h in v2r["selected_top3"]}
        s3 = {h["name"] for h in v3r["selected_top3"]}
        if s2 != s3: changed_sel3 += 1
    print(f"\nRanking motion:")
    print(f"  Model top-1 changed in {changed_top1}/{len(out)} races")
    print(f"  Selected top-3 set changed in {changed_sel3}/{len(out)} races")


if __name__ == "__main__":
    main()
