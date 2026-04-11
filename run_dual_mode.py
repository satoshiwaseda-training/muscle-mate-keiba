"""Offline dual-mode evaluation on the 50-race saved dataset.

For each horse:
  1. Extract facts from cached paddock_comment text (source="news")
  2. Extract per-horse-slice paddock_features scores (source="news")
  3. Derive JRA numeric facts from structured_features (weight, carried, track)
  4. Merge with consensus bonus
  5. Compute composite_condition + consensus_count + negative_facts list
  6. Run the already-computed odds_score (from v3 ranked)
  7. Apply dual_mode_score decision
  8. Re-rank races using the chosen scores + softmax
  9. Compare metrics vs v3 baseline

No network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import probability_engine as pe
import fact_extractor as fe
import fact_schema
import fact_validator as fv
import paddock_features as pf
import dual_mode_scoring as dm

ROOT = Path(__file__).parent
V3 = ROOT / "data" / "enriched_backtest_results_v3.json"
RES = ROOT / "data" / "results.json"
CACHE = ROOT / "data" / "scraper_cache" / "enrich_race"


def _norm(n): return (n or "").strip()
def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try: return float(s)
    except ValueError: return 0.0


def horse_facts(
    horse_name: str,
    h_sf: dict,
    cached_horse: dict | None,
    page_text: str | None,
    race_sf: dict,
) -> list:
    """Produce a fact list for a horse, combining text extraction and
    JRA numeric derivation."""
    facts = []

    # 1. JRA numeric: weight delta, carried weight
    wd = h_sf.get("horse_weight_delta")
    facts.extend(fe.fact_from_weight_delta(horse_name, wd))

    carried = h_sf.get("carried_weight") or 0
    if isinstance(carried, (int, float)) and carried >= 58:
        facts.append(fact_schema.Fact(
            type="high_carried_weight", horse=horse_name, polarity=-1,
            confidence=0.80, source="jra",
            raw_text=f"斤量 {carried}kg", category="weight",
        ))

    # 2. Per-horse paddock text extraction (if we can find this horse's
    # sentence segment in the full page text)
    if page_text:
        # Build segments across the full field
        names_in_page = [horse_name]   # single-target slicing
        segs = pf.extract_per_horse_comments(
            page_text,
            [h_sf_name for h_sf_name in race_sf.get("__all_names__", [horse_name])],
        )
        seg = segs.get(horse_name)
        if seg and seg.get("comment"):
            facts.extend(fe.extract_canonical_facts(
                seg["comment"], source="news", horse=horse_name,
            ))

    # 3. Raw cached paddock_comment text (less clean but broader)
    if cached_horse:
        raw = cached_horse.get("paddock_comment") or ""
        if raw:
            # Only grab text in a ~200-char window around the horse's name
            idx = raw.find(horse_name)
            if idx >= 0:
                window = raw[idx: idx + 200]
                facts.extend(fe.extract_canonical_facts(
                    window, source="keibalab", horse=horse_name,
                ))

        # 4. netkeiba oikiri editorial (training_eval field) — this is a
        # DIFFERENT author than the paddock-comment pipeline. Published
        # pre-race during training week. Serves as a third source for
        # category/type consensus.
        training_eval = cached_horse.get("training_eval") or ""
        if training_eval and len(training_eval) >= 3:
            facts.extend(fe.extract_canonical_facts(
                training_eval, source="netkeiba_oikiri", horse=horse_name,
            ))

    return facts


def main():
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    results = json.loads(RES.read_text(encoding="utf-8"))

    rows_out = []
    mode_stats = {"fact": 0, "odds": 0}
    horses_total = 0
    races_with_fact_mode_horse = 0
    validation_drops = {"total": 0, "by_reason": {}}
    state_totals = {"condition": [], "fatigue": [], "stress": [], "pain": []}

    for row in v3:
        rid = row["race_id"]
        sf = row["structured_features"]
        sf_horses = sf.get("horses") or {}
        all_names = list(sf_horses.keys())

        # Load cached paddock text
        cache_file = CACHE / f"{rid}.json"
        cached = []
        page_text = ""
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            for h in cached:
                pt = h.get("paddock_comment") or ""
                if len(pt) > len(page_text):
                    page_text = pt

        # For horse-name slicing in paddock_features we need the full list
        race_sf_for_slice = dict(sf)
        race_sf_for_slice["__all_names__"] = all_names

        # Build per-horse fact tables
        per_horse_facts = {}
        per_horse_consensus_count = {}
        per_horse_composite = {}
        per_horse_negatives = {}
        per_horse_states_cache = {}

        for name in all_names:
            h_sf = sf_horses[name]
            cached_h = next((c for c in cached if _norm(c.get("name")) == name), None)
            all_facts = horse_facts(name, h_sf, cached_h, page_text, race_sf_for_slice)
            # Validate + contradiction-detect BEFORE merge
            drop: list = []
            validated = fv.validate_and_transform(all_facts, drop_report=drop)
            validation_drops["total"] += len(drop)
            for d in drop:
                r = d["reason"].split(":")[0].split("(")[0].strip()
                validation_drops["by_reason"][r] = validation_drops["by_reason"].get(r, 0) + 1
            # Merge so consensus counts appear
            merged = fe.merge_fact_layers(validated)
            per_horse_facts[name] = merged
            agg = fe.aggregate_horse_score(merged)
            per_horse_composite[name] = agg["composite_condition"]
            per_horse_consensus_count[name] = agg["consensed_fact_count"]
            per_horse_negatives[name] = [f for f in merged if f.polarity < 0]
            # Capture state scores for diagnostic AND score penalty
            st = fv.compute_state_scores(merged)
            for k in ("condition", "fatigue", "stress", "pain"):
                key = ("condition_score" if k == "condition"
                       else "fatigue_score" if k == "fatigue"
                       else "stress_score" if k == "stress"
                       else "pain_risk")
                state_totals[k].append(st.get(key, 0))
            per_horse_states_cache[name] = st

        # Compute dual-mode scores per horse, with negative-state penalty
        STATE_PENALTY_WEIGHT = {
            "fatigue_score": 0.30,
            "stress_score":  0.30,
            "pain_risk":     0.40,
        }
        # Penalty is applied as a score-point deduction so horses with
        # strong fatigue/stress/pain signals drop in the ranking even
        # when their odds_score is high. The magnitude (15 points at
        # full negative state) is chosen so it affects ordering in
        # genuinely concerning cases without overwhelming the odds term.

        scored = []
        race_has_fact_mode = False
        for entry in row["ranked"]:
            name = entry["name"]
            odds_score = float(entry["score"])
            st = per_horse_states_cache.get(name, {})
            state_penalty_units = sum(
                float(st.get(k, 0.0)) * w
                for k, w in STATE_PENALTY_WEIGHT.items()
            )
            score_deduction = state_penalty_units * 15.0

            decision = dm.dual_mode_score(
                h_sf=sf_horses.get(name, {}),
                odds_score=odds_score,
                consensus_count=per_horse_consensus_count.get(name, 0),
                composite_condition=per_horse_composite.get(name, 0.5),
                negative_facts=per_horse_negatives.get(name, []),
            )
            adjusted_score = max(2.0, decision["score"] - score_deduction)
            mode_stats[decision["mode"]] += 1
            horses_total += 1
            if decision["mode"] == "fact":
                race_has_fact_mode = True
            scored.append({
                "name": name,
                "odds": entry["odds"],
                "score": adjusted_score,
                "mode": decision["mode"],
                "odds_score": decision["odds_score"],
                "fact_score": decision["fact_score"],
                "state_penalty": round(score_deduction, 2),
                "consensus_count": per_horse_consensus_count.get(name, 0),
                "composite": per_horse_composite.get(name, 0.5),
                "fatigue_score": st.get("fatigue_score", 0),
                "stress_score":  st.get("stress_score", 0),
                "pain_risk":     st.get("pain_risk", 0),
            })

        if race_has_fact_mode:
            races_with_fact_mode_horse += 1

        # Re-rank + softmax + select top-3
        ranked = pe.assign_win_probs(scored, temperature=pe.DEFAULT_TEMPERATURE)
        sel = pe.select_top3(ranked, alpha=pe.DEFAULT_ALPHA, beta=pe.DEFAULT_BETA)
        rows_out.append({
            "race_id": rid,
            "race_name": row.get("race_name", ""),
            "grade": row.get("grade", ""),
            "ranked": ranked,
            "selected_top3": sel["selected"],
            "p1": sel["p1"], "p2": sel["p2"],
            "structured_features": sf,
        })

    # Save
    OUT = ROOT / "data" / "enriched_backtest_results_dual.json"
    OUT.write_text(json.dumps(rows_out, ensure_ascii=False, indent=2),
                    encoding="utf-8")

    print(f"Wrote {len(rows_out)} races to {OUT.relative_to(ROOT)}")
    print(f"\nMode distribution (horses): fact={mode_stats['fact']}  "
          f"odds={mode_stats['odds']}  (total={horses_total})")
    print(f"Races with ≥1 fact-mode horse: {races_with_fact_mode_horse} / {len(rows_out)}")

    # Metrics vs v3 baseline
    def metrics(rows_list):
        N = m_hit = o_hit = t3w = t3p = dev = m_chg = 0
        m_pnl = o_pnl = 0.0
        for r in rows_list:
            rid = r["race_id"]
            res = results.get(f"bt_{rid}")
            if not res: continue
            fo = res.get("finishing_order") or []
            winner = next((_norm(h.get("name")) for h in fo if int(h.get("rank", 0) or 0) == 1), None)
            second = next((_norm(h.get("name")) for h in fo if int(h.get("rank", 0) or 0) == 2), None)
            if not winner: continue
            odds_map = {}
            for h in fo:
                nm = _norm(h.get("name")); od = _po(h.get("odds"))
                if nm and od > 0 and nm not in odds_map:
                    odds_map[nm] = od
            if not odds_map: continue
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

    m_v3 = metrics(v3)
    m_dual = metrics(rows_out)
    print("\n" + "=" * 72)
    print("v3 (odds-weighted) vs DUAL (fact-weighted when strong) — 50 races")
    print("=" * 72)
    print(f'{"metric":22} {"v3":>12} {"dual":>12} {"Δ":>10}')
    print("-" * 60)
    for key, label in [("model_top1", "Model Top-1"),
                       ("odds_top1", "Odds Top-1"),
                       ("top3_w", "Top-3 winner"),
                       ("top3_p", "Top-3 pair"),
                       ("dev", "Deviations"),
                       ("chg_w", "Changed-race wins"),
                       ("m_roi", "Model ROI (%)"),
                       ("o_roi", "Odds ROI (%)")]:
        v3v = m_v3[key]; dv = m_dual[key]
        if isinstance(v3v, float):
            print(f'{label:22} {v3v:+12.2f} {dv:+12.2f} {dv-v3v:+10.2f}')
        else:
            print(f'{label:22} {v3v:>12} {dv:>12} {dv-v3v:>+10}')

    # Show which horses triggered fact mode
    print("\n── Horses triggered by fact-mode ──")
    shown = 0
    for row in rows_out:
        for h in row["ranked"]:
            if h.get("mode") == "fact" and shown < 20:
                print(f"  {row['race_id']} {row['race_name'][:12]:14} {h['name'][:14]:16} "
                      f"consensus={h['consensus_count']}  composite={h['composite']:.2f}  "
                      f"odds={h['odds']:.1f}")
                shown += 1

    # Validator diagnostics
    print("\n── Validation layer diagnostics ──")
    print(f"  Raw facts dropped by validator : {validation_drops['total']}")
    for reason, count in sorted(validation_drops["by_reason"].items(),
                                 key=lambda kv: -kv[1]):
        print(f"    {reason:30}: {count}")
    # State score summary
    import statistics as _st
    print("\n── State-score distribution (all horses) ──")
    for key, label in [("condition", "condition_score (↑ = ready)"),
                        ("fatigue",   "fatigue_score   (↑ = tired)"),
                        ("stress",    "stress_score    (↑ = tense)"),
                        ("pain",      "pain_risk       (↑ = discomfort)")]:
        vals = state_totals[key] or [0.0]
        print(f"  {label:36}: mean={_st.mean(vals):.3f}  "
              f"max={max(vals):.3f}  >0.5 count={sum(1 for v in vals if v > 0.5)}")

    # Diagnostic: distribution of fact-counts and composites across all horses
    all_horses = [h for row in rows_out for h in row["ranked"]]
    print("\n── Distribution across 758 horses ──")
    consensus_dist = {}
    for h in all_horses:
        c = h.get("consensus_count", 0)
        consensus_dist[c] = consensus_dist.get(c, 0) + 1
    print(f"  consensus_count histogram: {dict(sorted(consensus_dist.items()))}")
    # composite distribution
    buckets = {"0.0-0.4": 0, "0.4-0.5": 0, "0.5-0.6": 0, "0.6-0.7": 0, "0.7-0.8": 0, "0.8-1.0": 0}
    for h in all_horses:
        c = h.get("composite", 0.5)
        if c < 0.4: buckets["0.0-0.4"] += 1
        elif c < 0.5: buckets["0.4-0.5"] += 1
        elif c < 0.6: buckets["0.5-0.6"] += 1
        elif c < 0.7: buckets["0.6-0.7"] += 1
        elif c < 0.8: buckets["0.7-0.8"] += 1
        else: buckets["0.8-1.0"] += 1
    print(f"  composite histogram      : {buckets}")
    ever_passed = sum(
        1 for h in all_horses
        if h.get("consensus_count", 0) >= 3
        and h.get("composite", 0.5) >= 0.7
    )
    print(f"  horses with consensus≥3 AND composite≥0.7: {ever_passed}")


def run_relaxed():
    """Run with a RELAXED strong_fact_signal criterion to characterize
    what the dual-mode would do if data were denser. Uses
      consensus_count ≥ 1 AND composite ≥ 0.65 AND no strong negatives.
    """
    import dual_mode_scoring as dm_mod

    # Monkey-patch thresholds
    orig_cons = dm_mod.STRONG_FACT_MIN_CONSENSUS
    orig_comp = dm_mod.STRONG_FACT_MIN_COMPOSITE
    dm_mod.STRONG_FACT_MIN_CONSENSUS = 1
    dm_mod.STRONG_FACT_MIN_COMPOSITE = 0.65
    try:
        main()
    finally:
        dm_mod.STRONG_FACT_MIN_CONSENSUS = orig_cons
        dm_mod.STRONG_FACT_MIN_COMPOSITE = orig_comp


if __name__ == "__main__":
    import sys as _sys
    if "--relaxed" in _sys.argv:
        print(">>> RELAXED MODE (consensus≥1, composite≥0.65) <<<\n")
        run_relaxed()
    else:
        main()
