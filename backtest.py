"""Historical backtest pipeline.

Fetches real past G1/G2/G3 races from netkeiba, builds structured
features from pre-race data, runs score_runner(), fetches results,
and evaluates through the production evaluator.

Usage:
    python backtest.py [--weeks N] [--enrich] [--resume]

    --weeks N    : how many weeks back to scan (default: 52)
    --enrich     : run full enrichment (slow: ~40s/race)
    --resume     : skip races already in backtest cache

Data flow:
    1. fetch_race_list_netkeiba(date) → race_ids
    2. fetch_entries(race_id)         → basic horse data
    3. [optional] enrich_entries()    → jockey/training/paddock
    4. extract_structured_features()  → structured_features dict
    5. score_runner()                 → prediction score
    6. fetch_result(race_id)          → actual finishing order
    7. evaluator.evaluate_walk_forward() → metrics
"""

import json
import sys
import time as _time
import argparse
from datetime import date, timedelta
from pathlib import Path

# Force line-buffered stdout for background execution
sys.stdout.reconfigure(line_buffering=True)

import scraper
import feature_store
from data_store import save_prediction, save_result, load_predictions, load_results

CACHE_DIR = Path(__file__).parent / "data" / "backtest_cache"
CACHE_FILE = CACHE_DIR / "backtest_races.json"


def _load_cache() -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def fetch_historical_races(n_weeks: int = 52) -> list[dict]:
    """Fetch G1/G2/G3 race list for past n_weeks."""
    print(f"Fetching race list for past {n_weeks} weeks...")
    today = date.today()
    weekday = today.weekday()
    days_since_sat = (weekday - 5) % 7
    last_sat = today - timedelta(days=days_since_sat)

    races = []
    seen = set()
    for w in range(n_weeks):
        for day_delta in (0, 1):  # Sat, Sun
            target = last_sat - timedelta(weeks=w) + timedelta(days=day_delta)
            if target >= today:
                continue
            try:
                day_races = scraper.fetch_race_list_netkeiba(target)
            except Exception as e:
                print(f"  Error fetching {target}: {e}")
                day_races = []
            for r in day_races:
                if r["race_id"] not in seen:
                    seen.add(r["race_id"])
                    r["race_date"] = target.isoformat()
                    races.append(r)
            _time.sleep(0.5)

        if (w + 1) % 10 == 0:
            print(f"  Scanned {w+1}/{n_weeks} weeks, found {len(races)} races so far")

    races.sort(key=lambda x: x.get("race_date", ""))
    print(f"Found {len(races)} G-races total")
    return races


def build_prediction_and_result(
    race: dict,
    do_enrich: bool = False,
) -> tuple[dict | None, dict | None]:
    """Build prediction + result for one race.

    For historical races, the entries page no longer shows pre-race odds.
    We fetch the result page to get odds (which were fixed pre-race by JRA
    at final calculation, typically T-2 minutes), then inject them into
    entries ONLY as the odds field. No finishing positions or post-race
    data enter the prediction features.

    Returns (prediction_dict, result_dict) or (None, None).
    """
    race_id = race["race_id"]
    venue = race.get("venue", "")

    # Step 1: Fetch result first (to get odds for entries)
    result = scraper.fetch_result_netkeiba(race_id)
    if not result or not result.get("finishing_order"):
        return None, None

    # Step 2: Fetch entries (basic pre-race data)
    entries = scraper.fetch_entries_netkeiba(race_id, venue)
    if not entries:
        return None, None

    # Check if entries are mock data
    is_mock = all(e.get("horse_id", "").startswith("horse0") for e in entries)
    if is_mock:
        return None, None

    # Step 3: Inject odds from result page into entries
    # ONLY the odds field — no finishing positions or times
    result_odds = {}
    for h in result.get("finishing_order", []):
        name = h.get("name", "")
        odds_val = h.get("odds", "")
        if name and odds_val:
            result_odds[name] = odds_val

    for e in entries:
        name = e.get("name", "")
        current_odds = e.get("odds", "0").replace("---", "0")
        # Only inject if entry has no odds (historical page shows 0)
        if feature_store._parse_odds(current_odds) <= 0 and name in result_odds:
            e["odds"] = result_odds[name]

    # Step 3b: DATA HYGIENE — drop scratched/excluded horses.
    # A horse that was in the shutuba page but never appears in finishing_order
    # was pulled from the race. Keeping it would produce auto-losing "model picks".
    finishers = set(result_odds.keys())
    pre_scratch_n = len(entries)
    entries = [e for e in entries if e.get("name", "") in finishers]
    scratched_count = pre_scratch_n - len(entries)
    if not entries:
        return None, None

    # Step 4: Fetch race info (track, distance, condition)
    race_info = scraper.fetch_race_info_netkeiba(race_id)

    # Step 5: Optional enrichment (slow: ~40s/race)
    if do_enrich:
        try:
            entries = scraper.enrich_entries(entries, race_id)
        except Exception as e:
            print(f"    Enrichment failed: {e}")

    # Step 6: Extract structured features (T-15 safe only)
    sf = feature_store.extract_structured_features(
        entries=entries,
        race_info=race_info,
        track_condition=race_info.get("track_condition", ""),
        weather=race_info.get("weather", ""),
        temperature=race_info.get("temperature", ""),
        cushion_value=race_info.get("cushion_value", ""),
        venue=venue,
    )

    # Step 7: Score ALL horses via score_runner, rank by score
    from train import score_runner as _score_fn
    from data_store import load_weights
    _ctx = {"weights": load_weights()}
    grade = race.get("grade", "")

    scored = []
    for e in entries:
        e_name = e.get("name", "")
        e_odds = feature_store._parse_odds(e.get("odds", "0"))
        # Build a single-horse feature dict to score this horse
        single_hf = [
            {"name": e_name, "rank": 1, "odds": e_odds,
             "confidence": 0, "ev_gap": 0, "bet": ""},
        ]
        # Include other horses for consensus calculation
        for other in entries:
            if other.get("name") != e_name:
                o_odds = feature_store._parse_odds(other.get("odds", "0"))
                single_hf.append({"name": other.get("name", ""), "rank": 2,
                                  "odds": o_odds, "confidence": 0, "ev_gap": 0, "bet": ""})
        feat = {
            "grade": grade,
            "num_horses": len(entries),
            "horse_features": single_hf,
            "structured_features": sf,
        }
        sc = _score_fn(feat, _ctx)["top_confidence"]
        scored.append({"name": e_name, "odds": e.get("odds", "0"), "score": sc})

    # Sort by score descending (highest score = best pick)
    scored.sort(key=lambda x: x["score"], reverse=True)

    # Also compute odds-only ranking for comparison metadata
    odds_ranked = sorted(scored, key=lambda x: feature_store._parse_odds(x["odds"]))
    odds_top = odds_ranked[0]["name"] if odds_ranked else ""
    score_top = scored[0]["name"] if scored else ""

    # Probability layer: softmax → per-horse win prob → Top-3 set selection
    import probability_engine as _pe
    _cfg = _pe.load_config()
    _T = float(_cfg.get("temperature", _pe.DEFAULT_TEMPERATURE))
    _alpha = float(_cfg.get("alpha", _pe.DEFAULT_ALPHA))
    _beta = float(_cfg.get("beta", _pe.DEFAULT_BETA))
    # Ensure numeric odds on each scored row
    for _s in scored:
        _s["odds"] = feature_store._parse_odds(_s.get("odds", "0"))
    ranked = _pe.assign_win_probs(scored, temperature=_T)
    sel = _pe.select_top3(ranked, alpha=_alpha, beta=_beta)

    horses = []
    for i, s in enumerate(sel["selected"]):
        horses.append({
            "rank": i + 1,
            "name": s["name"],
            "odds": f'{s["odds"]:.1f}' if s["odds"] else "0",
            "win_prob": s["win_prob"],
            "confidence": round(s["win_prob"] * 100, 1),
            "ev_gap": "0", "bet": "",
        })

    if not horses:
        return None, None

    prediction = {
        "race_name": race.get("race_name", ""),
        "grade": grade,
        "horses": horses,
        "timestamp": race.get("race_date", "") + "T10:00:00",
        "structured_features": sf,
        "backtest": True,
        "_ranking_meta": {
            "odds_top": odds_top,
            "score_top": score_top,
            "ranking_changed": odds_top != score_top,
            "top_score": scored[0]["score"] if scored else 0,
            "scratched_count": scratched_count,
        },
        "theoretical": {
            "selected_top3": [h["name"] for h in sel["selected"]],
            "p1": sel["p1"],
            "p2": sel["p2"],
            "objective": sel["objective"],
            "temperature": _T,
            "alpha": _alpha,
            "beta": _beta,
            "win_probs": {h["name"]: h["win_prob"] for h in ranked},
        },
    }

    result["timestamp"] = race.get("race_date", "") + "T12:00:00"
    result["race_name"] = race.get("race_name", "")

    return prediction, result


def run_backtest(n_weeks: int = 52, do_enrich: bool = False, resume: bool = True):
    """Main backtest pipeline."""
    print("=" * 60)
    print("HISTORICAL BACKTEST PIPELINE")
    print(f"Weeks: {n_weeks}, Enrich: {do_enrich}, Resume: {resume}")
    print("=" * 60)

    cache = _load_cache() if resume else {}

    # Phase 1: Fetch race list
    races = fetch_historical_races(n_weeks)
    if not races:
        print("ERROR: No races found")
        return

    # Phase 2: Build predictions + fetch results
    print(f"\nPhase 2: Building predictions and fetching results...")
    predictions = load_predictions()
    results = load_results()

    processed = 0
    skipped = 0
    errors = 0
    enrichment_stats = {"full": 0, "basic": 0}
    enrich_horse_totals = {"total": 0, "ok": 0, "partial": 0, "failed": 0}

    for i, race in enumerate(races):
        race_id = race["race_id"]
        bt_key = f"bt_{race_id}"

        # Skip if already processed
        if resume and bt_key in predictions and race_id in cache:
            skipped += 1
            continue

        print(f"  [{i+1}/{len(races)}] {race.get('race_name', race_id)} "
              f"({race.get('race_date', '?')})...", end=" ", flush=True)

        # Build prediction + result together (odds come from result page)
        pred, result = build_prediction_and_result(race, do_enrich=do_enrich)
        if pred is None or result is None:
            print("SKIP (no data)")
            errors += 1
            continue

        # Save both
        predictions[bt_key] = pred
        results[bt_key] = result

        # Track enrichment quality
        sf_horses = pred.get("structured_features", {}).get("horses", {})
        if sf_horses:
            first_horse = next(iter(sf_horses.values()), {})
            if first_horse.get("jockey_win_rate", 0) > 0:
                enrichment_stats["full"] += 1
            else:
                enrichment_stats["basic"] += 1

        # Roll up per-horse enrichment stats from scraper cache (if --enrich was on)
        if do_enrich:
            rstats = scraper.get_enrich_stats(race_id)
            if rstats:
                for k in enrich_horse_totals:
                    enrich_horse_totals[k] += rstats.get(k, 0)

        cache[race_id] = {
            "race_name": race.get("race_name", ""),
            "race_date": race.get("race_date", ""),
            "grade": race.get("grade", ""),
            "num_horses": len(pred.get("structured_features", {}).get("horses", {})),
        }

        processed += 1
        meta = pred.get("_ranking_meta", {})
        changed = "CHANGED" if meta.get("ranking_changed") else "same"
        print(f"OK ({changed}, top={meta.get('score_top','?')[:6]})")

        # Save frequently so a crash never loses more than 5 races of work.
        if processed % 5 == 0:
            _save_predictions_results(predictions, results)
            _save_cache(cache)
            total_h = enrich_horse_totals["total"] or 1
            ok_rate = (enrich_horse_totals["ok"] + enrich_horse_totals["partial"]) / total_h * 100
            print(f"    [Checkpoint: {processed} processed, {skipped} skipped, "
                  f"{errors} errors, enrich success {ok_rate:.1f}%]")

    # Final save
    _save_predictions_results(predictions, results)
    _save_cache(cache)

    # Count backtest pairs and ranking changes
    bt_keys = [k for k in predictions if k.startswith("bt_")]
    paired = [k for k in bt_keys if k in results]

    ranking_changed = 0
    for k in bt_keys:
        meta = predictions[k].get("_ranking_meta", {})
        if meta.get("ranking_changed"):
            ranking_changed += 1

    print(f"\nPhase 2 complete:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (cached): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Total backtest predictions: {len(bt_keys)}")
    print(f"  Paired (pred+result): {len(paired)}")
    print(f"  Enrichment: {enrichment_stats}")
    if do_enrich and enrich_horse_totals["total"]:
        t = enrich_horse_totals["total"]
        ok = enrich_horse_totals["ok"]
        partial = enrich_horse_totals["partial"]
        failed = enrich_horse_totals["failed"]
        print(f"  Horse-level enrichment: {ok}/{t} ok, {partial} partial, {failed} failed "
              f"(success rate {(ok + partial) / t * 100:.1f}%)")
    print(f"  Rankings changed vs odds: {ranking_changed}/{len(bt_keys)} "
          f"({ranking_changed/len(bt_keys)*100:.1f}%)" if bt_keys else "")

    if len(paired) < 10:
        print("\nERROR: Too few paired races for evaluation")
        return

    # Phase 3: Run evaluation
    print(f"\n{'='*60}")
    print("Phase 3: Walk-forward evaluation")
    print("=" * 60)
    _run_evaluation(predictions, results)


def _save_predictions_results(predictions: dict, results: dict):
    """Save to production data files."""
    from data_store import PREDICTIONS_FILE, RESULTS_FILE, _save_json
    _save_json(PREDICTIONS_FILE, predictions)
    _save_json(RESULTS_FILE, results)


def _run_evaluation(predictions: dict, results: dict):
    """Run full evaluation using the production evaluator."""
    # We need to use evaluator as-is, which reads from data_store
    # Data is already saved, so evaluator will pick it up
    from evaluator import evaluate_walk_forward, baseline_score_runner, check_adoption
    from train import score_runner

    print("\n--- Baseline (Gemini confidence passthrough) ---")
    baseline = evaluate_walk_forward(baseline_score_runner, mask_odds=False)
    if baseline.get("error"):
        print(f"ERROR: {baseline['error']}")
        return
    _print_metrics("Baseline", baseline)

    print("\n--- Candidate (structured formula) ---")
    candidate = evaluate_walk_forward(score_runner, mask_odds=False)
    if candidate.get("error"):
        print(f"ERROR: {candidate['error']}")
        return
    _print_metrics("Candidate", candidate)

    print("\n--- Candidate (odds masked) ---")
    masked = evaluate_walk_forward(score_runner, mask_odds=True)
    if not masked.get("error"):
        _print_metrics("Masked", masked)

    # Adoption gate
    print("\n--- Adoption Gate ---")
    adoption = check_adoption(candidate, baseline, masked, min_races=300)
    for name, check in adoption["checks"].items():
        status = "PASS" if check["passed"] else "FAIL"
        print(f"  [{status}] {name}: {check['details']}")

    print(f"\nOverall: {'ADOPTED' if adoption['adopted'] else 'DISCARDED'}")
    if adoption.get("reason"):
        print(f"Reason: {adoption['reason']}")

    # Generate report
    _generate_backtest_report(baseline, candidate, masked, adoption)


def _print_metrics(label: str, m: dict):
    print(f"  {label} ROI:     {m['roi']:.4f}")
    print(f"  {label} Brier:   {m['brier']:.4f}")
    print(f"  {label} MaxDD:   {m['max_drawdown']:.4f}")
    print(f"  {label} Races:   {m['num_races']}")
    yr = m.get("yearly_results", {})
    if yr:
        print(f"  {label} Years:   {sorted(yr.keys())}")


def _generate_backtest_report(baseline, candidate, masked, adoption):
    """Generate backtest_report.md."""
    lines = []
    lines.append("# Historical Backtest Report")
    lines.append(f"\nGenerated: {date.today().isoformat()}")

    # 1. Dataset Summary
    lines.append("\n## 1. Dataset Summary")
    lines.append(f"\n- Total races evaluated: {candidate['num_races']}")
    yr = candidate.get("yearly_results", {})
    years = sorted(yr.keys())
    if years:
        lines.append(f"- Time range: {min(years)} to {max(years)}")
    for y in years:
        lines.append(f"  - {y}: {yr[y]['num_races']} races, "
                     f"hit rate {yr[y].get('hit_rate', 0):.1%}")

    # 2. Metrics
    lines.append("\n## 2. Core Metrics")
    lines.append("\n| Metric | Baseline (Gemini conf) | Candidate (formula) | Status |")
    lines.append("|--------|----------------------|---------------------|--------|")
    lines.append(f"| ROI | {baseline['roi']:.4f} | {candidate['roi']:.4f} | {'PASS' if candidate['roi'] >= baseline['roi'] else 'FAIL'} |")
    lines.append(f"| Brier | {baseline['brier']:.4f} | {candidate['brier']:.4f} | {'PASS' if candidate['brier'] <= baseline['brier'] else 'FAIL'} |")
    lines.append(f"| MaxDrawdown | {baseline['max_drawdown']:.4f} | {candidate['max_drawdown']:.4f} | - |")
    b_wins = sum(r["actual_win"] for r in baseline.get("race_results", []))
    c_wins = sum(r["actual_win"] for r in candidate.get("race_results", []))
    lines.append(f"| Win count | {b_wins}/{baseline['num_races']} | {c_wins}/{candidate['num_races']} | - |")

    # 3. Fold Results
    lines.append("\n## 3. Fold Consistency")
    lines.append("\n| Fold | Races | B-ROI | C-ROI | B-Brier | C-Brier |")
    lines.append("|------|-------|-------|-------|---------|---------|")
    b_folds = baseline.get("fold_results", [])
    c_folds = candidate.get("fold_results", [])
    improved_folds = 0
    for i in range(max(len(b_folds), len(c_folds))):
        bf = b_folds[i] if i < len(b_folds) else {}
        cf = c_folds[i] if i < len(c_folds) else {}
        lines.append(f"| {i} | {cf.get('num_races', '?')} | "
                     f"{bf.get('roi', 0):.4f} | {cf.get('roi', 0):.4f} | "
                     f"{bf.get('brier', 1):.4f} | {cf.get('brier', 1):.4f} |")
        if i < len(b_folds) and i < len(c_folds):
            if cf["brier"] <= bf["brier"]:
                improved_folds += 1
    total_folds = min(len(b_folds), len(c_folds))
    if total_folds:
        lines.append(f"\nFolds with improved Brier: {improved_folds}/{total_folds} ({improved_folds/total_folds:.0%})")

    # 4. Odds Mask Test
    lines.append("\n## 4. Odds Mask Test")
    if not masked.get("error"):
        lines.append(f"\n| Metric | Normal | Masked | Ratio |")
        lines.append("|--------|--------|--------|-------|")
        lines.append(f"| ROI | {candidate['roi']:.4f} | {masked['roi']:.4f} | - |")
        lines.append(f"| Brier | {candidate['brier']:.4f} | {masked['brier']:.4f} | - |")
        comp_n = candidate['roi'] - candidate['brier']
        comp_m = masked['roi'] - masked['brier']
        ratio = comp_m / comp_n if comp_n != 0 else 0
        lines.append(f"\nComposite ratio (masked/normal): {ratio:.4f}")
        lines.append(f"Threshold: >= 0.80 → {'PASS' if ratio >= 0.8 else 'FAIL'}")
    else:
        lines.append(f"\nMasked evaluation error: {masked.get('error', 'unknown')}")

    # 5. Annual Breakdown
    lines.append("\n## 5. Annual Matrix")
    lines.append("\n| Year | Races | B-ROI | C-ROI | B-Brier | C-Brier | Hit Rate |")
    lines.append("|------|-------|-------|-------|---------|---------|----------|")
    b_yr = baseline.get("yearly_results", {})
    c_yr = candidate.get("yearly_results", {})
    for y in sorted(set(list(b_yr.keys()) + list(c_yr.keys()))):
        b = b_yr.get(y, {})
        c = c_yr.get(y, {})
        lines.append(f"| {y} | {c.get('num_races', '?')} | "
                     f"{b.get('roi', 0):.4f} | {c.get('roi', 0):.4f} | "
                     f"{b.get('brier', 1):.4f} | {c.get('brier', 1):.4f} | "
                     f"{c.get('hit_rate', 0):.1%} |")

    # 6. Adoption Gate
    lines.append("\n## 6. Adoption Gate")
    for name, check in adoption["checks"].items():
        lines.append(f"\n- **[{'PASS' if check['passed'] else 'FAIL'}] {name}**: {check['details']}")
    lines.append(f"\nOverall: **{'ADOPTED' if adoption['adopted'] else 'DISCARDED'}**")
    if adoption.get("reason"):
        lines.append(f"\nReason: {adoption['reason']}")

    # 7. Ranking Analysis
    lines.append("\n## 7. Ranking Shift Analysis")
    preds = load_predictions()
    bt_preds = {k: v for k, v in preds.items() if k.startswith("bt_")}
    results_data = load_results()

    total_bt = len(bt_preds)
    changed = 0
    changed_won = 0
    odds_would_won = 0
    for k, pred in bt_preds.items():
        meta = pred.get("_ranking_meta", {})
        if meta.get("ranking_changed"):
            changed += 1
            # Did the model's pick win?
            score_top = meta.get("score_top", "")
            odds_top = meta.get("odds_top", "")
            res = results_data.get(k, {})
            finishing = {h["name"]: h["rank"] for h in res.get("finishing_order", [])}
            if finishing.get(score_top) == 1:
                changed_won += 1
            if finishing.get(odds_top) == 1:
                odds_would_won += 1

    lines.append(f"\n- Total races: {total_bt}")
    lines.append(f"- Rankings changed (model != odds favorite): **{changed}/{total_bt}** "
                 f"({changed/total_bt*100:.1f}%)" if total_bt else "")
    if changed > 0:
        lines.append(f"- When model overrode odds: model's pick won **{changed_won}/{changed}** "
                     f"({changed_won/changed*100:.1f}%)")
        lines.append(f"- Odds favorite would have won: **{odds_would_won}/{changed}** "
                     f"({odds_would_won/changed*100:.1f}%)")

    # 8. Data Quality
    lines.append("\n## 8. Data Quality Notes")
    lines.append("\n- Top picks selected by model score (not by odds alone)")
    lines.append("- Gemini confidence is set to 0 for all backtest records")
    lines.append("- structured_features availability depends on enrichment flag")
    lines.append("- Missing features default to 0.0 (graceful degradation)")

    lines.append("\n---")
    lines.append("**This is a historical backtest, not a live trading result.**")

    report_path = Path(__file__).parent / "backtest_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Historical backtest")
    parser.add_argument("--weeks", type=int, default=52, help="Weeks back to scan")
    parser.add_argument("--enrich", action="store_true", help="Run full enrichment (slow)")
    parser.add_argument("--resume", action="store_true", default=True, help="Skip cached races")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args()

    run_backtest(n_weeks=args.weeks, do_enrich=args.enrich, resume=args.resume)


if __name__ == "__main__":
    main()
