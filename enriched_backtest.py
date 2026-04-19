"""Focused enriched backtest — no race_list scraping, just a list of race_ids.

Picks N race_ids from the clean NEW cohort (rows with `_ranking_meta`),
runs the full enrich → features → score → probability pipeline on each,
and reports:
  - per-feature coverage (populate rate across horses)
  - Top-1 / Top-3 / ROI
  - deviation rate
  - comparison to the non-enriched baseline from saved data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import scraper
import feature_store as fs
import probability_engine as pe
from train import score_runner
from data_store import load_weights

ROOT = Path(__file__).parent
PRED_FILE = ROOT / "data" / "predictions.json"
RES_FILE = ROOT / "data" / "results.json"
OUT_FILE = ROOT / "data" / "enriched_backtest_results.json"


def _po(s) -> float:
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def _norm(n: str) -> str:
    return (n or "").strip()


def load_clean_race_ids(limit: int) -> list[dict]:
    preds = json.loads(PRED_FILE.read_text(encoding="utf-8"))
    results = json.loads(RES_FILE.read_text(encoding="utf-8"))
    clean = [k for k in preds
             if k.startswith("bt_")
             and preds[k].get("_ranking_meta")
             and k in results]
    # Sort by timestamp desc (most recent first) for a useful sample
    clean.sort(key=lambda k: preds[k].get("timestamp", ""), reverse=True)
    out = []
    for k in clean[:limit]:
        rid = k.replace("bt_", "")
        out.append({
            "race_id": rid,
            "key": k,
            "venue": preds[k].get("structured_features", {}).get("race", {}).get("venue", ""),
            "grade": preds[k].get("grade", ""),
            "race_name": preds[k].get("race_name", ""),
            "saved_pred": preds[k],
            "result": results[k],
        })
    return out


def run_enriched_prediction(race: dict) -> dict | None:
    rid = race["race_id"]
    venue = race["venue"]
    grade = race["grade"]
    print(f"  fetching {rid}...", flush=True)

    entries = scraper.fetch_entries_netkeiba(rid, venue)
    if not entries or all(e.get("horse_id", "").startswith("horse0") for e in entries):
        print(f"    (no entries or mock data)")
        return None

    # Scratch filter based on result finishing_order
    fo = race["result"].get("finishing_order") or []
    finishers = {_norm(h.get("name")) for h in fo if _norm(h.get("name"))}
    entries = [e for e in entries if _norm(e.get("name")) in finishers]
    if not entries:
        return None

    # Build result_odds defensively: the result file can contain a duplicate
    # rank=1 row with empty odds; skip zero-odds entries so they can't clobber
    # the real value and keep the first non-zero hit per name.
    result_odds: dict[str, float] = {}
    for h in fo:
        name = _norm(h.get("name"))
        od = _po(h.get("odds"))
        if name and od > 0 and name not in result_odds:
            result_odds[name] = od

    # Enrich (uses scraper cache automatically)
    t0 = time.time()
    try:
        entries = scraper.enrich_entries(entries, rid, race_name=race.get("race_name", ""))
    except Exception as e:
        print(f"    enrichment failed: {e}")
        return None
    dt = time.time() - t0

    # Inject odds from result page AFTER enrichment so that
    # (a) a stale enrich_race cache can't return wrong-odds entries, and
    # (b) the injection always wins over whatever enrich_entries put there.
    for e in entries:
        name = _norm(e.get("name"))
        if name in result_odds:
            e["odds"] = str(result_odds[name])

    race_info = scraper.fetch_race_info_netkeiba(rid) or {}
    sf = fs.extract_structured_features(
        entries=entries,
        race_info=race_info,
        track_condition=race_info.get("track_condition", ""),
        weather=race_info.get("weather", ""),
        temperature=race_info.get("temperature", ""),
        cushion_value=race_info.get("cushion_value", ""),
        venue=venue,
    )

    # Score every horse
    ctx = {"weights": load_weights()}
    scored = []
    for horse in entries:
        rotated = [{"name": _norm(horse.get("name")), "rank": 1,
                    "odds": _po(horse.get("odds", "0")),
                    "confidence": 0, "ev_gap": 0, "bet": ""}]
        for other in entries:
            if other is horse:
                continue
            rotated.append({"name": _norm(other.get("name")), "rank": 2,
                            "odds": _po(other.get("odds", "0")),
                            "confidence": 0, "ev_gap": 0, "bet": ""})
        feat = {
            "grade": grade, "num_horses": len(entries),
            "horse_features": rotated, "structured_features": sf,
        }
        s = score_runner(feat, ctx).get("top_confidence", 50.0)
        scored.append({
            "name": _norm(horse.get("name")),
            "odds": _po(horse.get("odds", "0")),
            "score": float(s),
        })

    ranked = pe.assign_win_probs(scored, temperature=pe.DEFAULT_TEMPERATURE)
    sel = pe.select_top3(ranked, alpha=pe.DEFAULT_ALPHA, beta=pe.DEFAULT_BETA)

    return {
        "race_id": rid,
        "race_name": race["race_name"],
        "grade": grade,
        "enrich_time_s": round(dt, 1),
        "structured_features": sf,
        "ranked": ranked,
        "selected_top3": sel["selected"],
        "p1": sel["p1"], "p2": sel["p2"], "objective": sel["objective"],
    }


def coverage(structured: dict, fields: list[str]) -> dict:
    horses = (structured.get("horses") or {}).values()
    cov = {f: {"pop": 0, "n": 0} for f in fields}
    for h in horses:
        for f in fields:
            cov[f]["n"] += 1
            v = h.get(f)
            if isinstance(v, (int, float)) and v != 0:
                cov[f]["pop"] += 1
    return cov


def metrics(rows: list[tuple[dict, dict]]) -> dict:
    """rows: list of (enriched_result, race_info from loader)."""
    N = 0
    m_hit = o_hit = t3w = t3p = 0
    dev = m_chg_win = o_chg_win = 0
    m_pnl = o_pnl = 0.0
    for er, race in rows:
        if not er:
            continue
        fo = race["result"].get("finishing_order") or []
        winner = next((_norm(h.get("name")) for h in fo if int(h.get("rank", 0) or 0) == 1), None)
        second = next((_norm(h.get("name")) for h in fo if int(h.get("rank", 0) or 0) == 2), None)
        if not winner:
            continue
        odds_map = {_norm(h.get("name")): _po(h.get("odds")) for h in fo if _po(h.get("odds")) > 0}
        if not odds_map:
            continue
        of = min(odds_map, key=lambda n: odds_map[n])
        mt = er["ranked"][0]["name"] if er["ranked"] else ""
        sel_names = {h["name"] for h in er["selected_top3"]}

        N += 1
        if mt == winner: m_hit += 1
        if of == winner: o_hit += 1
        if winner in sel_names: t3w += 1
        if winner in sel_names and second in sel_names: t3p += 1
        if mt != of:
            dev += 1
            if mt == winner: m_chg_win += 1
            if of == winner: o_chg_win += 1
        m_pnl += (odds_map.get(mt, 0) - 1) if mt == winner else -1
        o_pnl += (odds_map.get(of, 0) - 1) if of == winner else -1

    if not N:
        return {}
    return {
        "n": N,
        "model_top1": m_hit / N,
        "odds_top1": o_hit / N,
        "top3_winner": t3w / N,
        "top3_pair": t3p / N,
        "model_roi": m_pnl / N,
        "odds_roi": o_pnl / N,
        "deviation_rate": dev / N,
        "changed_n": dev,
        "changed_model_win": m_chg_win,
        "changed_odds_win": o_chg_win,
        "changed_model_win_rate": (m_chg_win / dev) if dev else 0.0,
        "changed_odds_win_rate": (o_chg_win / dev) if dev else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num-races", type=int, default=20)
    args = ap.parse_args()

    races = load_clean_race_ids(args.num_races)
    print(f"Running enriched backtest on {len(races)} race(s)\n")

    COVERAGE_FIELDS = [
        "jockey_win_rate", "jockey_g1_wins",
        "training_acceleration", "training_cardio_index",
        "training_final_split",
        "paddock_vascularity", "paddock_hindquarter", "paddock_gait",
        "horse_weight_delta", "horse_weight_kg",
    ]

    results = []
    tot_cov = {f: {"pop": 0, "n": 0} for f in COVERAGE_FIELDS}

    for i, race in enumerate(races):
        print(f"[{i+1}/{len(races)}] {race['race_name']} ({race['grade']})")
        er = run_enriched_prediction(race)
        results.append((er, race))
        if er:
            c = coverage(er["structured_features"], COVERAGE_FIELDS)
            for f, d in c.items():
                tot_cov[f]["pop"] += d["pop"]
                tot_cov[f]["n"] += d["n"]

    # Save detailed results
    try:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for er, race in results:
            if not er:
                continue
            serializable.append({
                "race_id": er["race_id"],
                "race_name": er["race_name"],
                "grade": er["grade"],
                "ranked": er["ranked"],
                "selected_top3": er["selected_top3"],
                "p1": er["p1"], "p2": er["p2"],
                "structured_features": er["structured_features"],
            })
        OUT_FILE.write_text(json.dumps(serializable, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        print(f"\nSaved to {OUT_FILE.relative_to(ROOT)}")
    except Exception as e:
        print(f"save failed: {e}")

    # Coverage report
    print(f"\n{'='*60}")
    print("FEATURE COVERAGE (populated across all horses)")
    print('='*60)
    for f, d in tot_cov.items():
        n = d["n"] or 1
        pct = d["pop"] / n * 100
        print(f"  {f:28}: {d['pop']:4d}/{n:4d} ({pct:5.1f}%)")

    # Metrics
    m = metrics(results)
    if m:
        print(f"\n{'='*60}")
        print("ENRICHED BACKTEST METRICS")
        print('='*60)
        print(f"  N                    : {m['n']}")
        print(f"  Model Top-1          : {m['model_top1']*100:.2f}%")
        print(f"  Odds  Top-1          : {m['odds_top1']*100:.2f}%")
        print(f"  Top-3 winner cover   : {m['top3_winner']*100:.2f}%")
        print(f"  Top-3 pair   cover   : {m['top3_pair']*100:.2f}%")
        print(f"  Model ROI            : {m['model_roi']*100:+.2f}%")
        print(f"  Odds  ROI            : {m['odds_roi']*100:+.2f}%")
        print(f"  Deviation rate       : {m['deviation_rate']*100:.1f}% ({m['changed_n']}/{m['n']})")
        print(f"  Changed-race M wins  : {m['changed_model_win']}/{m['changed_n']} = {m['changed_model_win_rate']*100:.1f}%")
        print(f"  Changed-race O wins  : {m['changed_odds_win']}/{m['changed_n']} = {m['changed_odds_win_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
