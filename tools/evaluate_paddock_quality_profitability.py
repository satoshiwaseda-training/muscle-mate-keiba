"""Evaluate whether paddock observation text improves betting profitability.

This is an offline diagnostic, not a production strategy change.  It
compares three fact-input variants on cached races:

  - no_paddock: objective/training facts only
  - legacy_paddock: historical cached paddock text, including old full-page
    extraction noise
  - quality_gated: same historical paddock cache after paddock_quality gate

The key question: does adding the newly quality-gated paddock information
create positive ROI, or at least improve ROI versus the old noisy cache?
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import fact_extractor as fe
import fact_validator as fv
import paddock_quality as pq


V3 = ROOT / "data" / "enriched_backtest_results_v3.json"
RESULTS = ROOT / "data" / "results.json"
ENRICH_CACHE = ROOT / "data" / "scraper_cache" / "enrich_race"
PADDOCK_CACHE = ROOT / "data" / "scraper_cache" / "paddock"
TICKET = 100.0

Variant = Literal["no_paddock", "legacy_paddock", "quality_gated"]


def _norm(name: str) -> str:
    return (name or "").strip()


def _float(raw) -> float:
    try:
        return float(str(raw).replace(",", "").replace("--", "0").replace("---", "0"))
    except (TypeError, ValueError):
        return 0.0


def _load_results() -> dict:
    raw = json.loads(RESULTS.read_text(encoding="utf-8"))
    return {(k[3:] if k.startswith("bt_") else k): v for k, v in raw.items()}


def _winner(result: dict) -> str:
    for horse in result.get("finishing_order") or []:
        try:
            if int(horse.get("rank", 0) or 0) == 1:
                return _norm(horse.get("name"))
        except (TypeError, ValueError):
            continue
    return ""


def _odds_map(result: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for horse in result.get("finishing_order") or []:
        name = _norm(horse.get("name"))
        odds = _float(horse.get("odds"))
        if name and odds > 0:
            out[name] = odds
    return out


def _grade_bucket(grade: str) -> str:
    g = (grade or "").upper()
    if "G1" in g or g == "GI":
        return "G1"
    if "G2" in g or g == "GII":
        return "G2"
    if "G3" in g or g == "GIII":
        return "G3"
    return "OTHER"


def _load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _paddock_report(race_id: str, horse: str) -> dict:
    reports = _load_json(PADDOCK_CACHE / f"{race_id}.json", {})
    report = reports.get(horse) or {}
    if isinstance(report, dict):
        return report
    return {}


def _cached_horse(race_id: str, horse: str) -> dict:
    rows = _load_json(ENRICH_CACHE / f"{race_id}.json", [])
    for row in rows if isinstance(rows, list) else []:
        if _norm(row.get("name")) == horse:
            return row
    return {}


def _facts_for_horse(race_id: str, horse: str, structured: dict, variant: Variant):
    facts = []
    facts.extend(fe.fact_from_weight_delta(horse, structured.get("horse_weight_delta")))

    cached = _cached_horse(race_id, horse)
    training_eval = cached.get("training_eval") or ""
    if training_eval and len(training_eval) >= 3:
        facts.extend(fe.extract_canonical_facts(training_eval, "netkeiba_oikiri", horse=horse))

    if variant == "no_paddock":
        return facts

    report = _paddock_report(race_id, horse)
    text = report.get("text") or cached.get("paddock_comment") or ""
    source = report.get("source") or cached.get("paddock_source") or ""

    if variant == "quality_gated":
        quality = pq.assess_paddock_report(text, source)
        if not quality.get("usable"):
            return facts

    if text:
        facts.extend(fe.extract_canonical_facts(text, "news", horse=horse))
    return facts


def _horse_metrics(race_id: str, horse: str, structured: dict, variant: Variant) -> dict:
    raw = _facts_for_horse(race_id, horse, structured, variant)
    valid = fv.validate_and_transform(raw)
    merged = fe.merge_fact_layers(valid)
    agg = fe.aggregate_horse_score(merged)
    negatives = [f for f in merged if f.polarity < 0]
    return {
        "n_facts": agg["n_facts"],
        "consensus_count": agg["consensed_fact_count"],
        "composite_condition": agg["composite_condition"],
        "strong_negative_present": any(float(f.confidence) > 0.6 for f in negatives),
    }


def trigger_loose_capped(horse: dict) -> bool:
    return (
        horse["consensus_count"] >= 1
        and horse["composite_condition"] >= 0.60
        and horse["odds"] <= 15.0
        and not horse["strong_negative_present"]
    )


def build_rows(variant: Variant) -> list[dict]:
    results = _load_results()
    rows = []
    for race in json.loads(V3.read_text(encoding="utf-8")):
        race_id = race.get("race_id", "")
        result = results.get(race_id)
        if not result:
            continue
        winner = _winner(result)
        odds = _odds_map(result)
        if not winner or not odds:
            continue
        win_payout = _float((result.get("payouts") or {}).get("単勝"))
        structured = (race.get("structured_features") or {}).get("horses") or {}

        horses = []
        for name, features in structured.items():
            name = _norm(name)
            if not name:
                continue
            metrics = _horse_metrics(race_id, name, features, variant)
            horses.append({
                **metrics,
                "name": name,
                "odds": _float(features.get("odds") or odds.get(name)),
                "is_winner": name == winner,
                "win_payout": win_payout if name == winner else 0.0,
            })
        rows.append({
            "race_id": race_id,
            "race_name": race.get("race_name", ""),
            "grade": race.get("grade", ""),
            "grade_bucket": _grade_bucket(race.get("grade", "")),
            "horses": horses,
        })
    return rows


def evaluate(rows: list[dict], grades: set[str] | None = None) -> dict:
    n_bets = wins = 0
    cost = payout = 0.0
    races = 0
    for row in rows:
        if grades and row["grade_bucket"] not in grades:
            continue
        races += 1
        for horse in row["horses"]:
            if trigger_loose_capped(horse):
                n_bets += 1
                cost += TICKET
                if horse["is_winner"]:
                    wins += 1
                    payout += horse["win_payout"]
    pnl = payout - cost
    return {
        "races": races,
        "n_bets": n_bets,
        "wins": wins,
        "cost": cost,
        "payout": payout,
        "pnl": pnl,
        "roi": pnl / cost if cost else 0.0,
        "win_rate": wins / n_bets if n_bets else 0.0,
    }


def coverage(rows: list[dict]) -> dict:
    out = defaultdict(lambda: {"races": 0, "horses": 0, "bets": 0})
    for row in rows:
        bucket = row["grade_bucket"]
        out[bucket]["races"] += 1
        for horse in row["horses"]:
            out[bucket]["horses"] += 1
            if trigger_loose_capped(horse):
                out[bucket]["bets"] += 1
    return dict(out)


def _fmt(label: str, metrics: dict) -> str:
    return (
        f"{label:<16s}"
        f" races={metrics['races']:>3d}"
        f" bets={metrics['n_bets']:>4d}"
        f" wins={metrics['wins']:>3d}"
        f" ROI={metrics['roi']*100:>+7.2f}%"
        f" PnL={metrics['pnl']:>+8.0f}"
        f" win%={metrics['win_rate']*100:>5.1f}%"
    )


def main() -> int:
    variants: list[Variant] = ["no_paddock", "legacy_paddock", "quality_gated"]
    rows_by_variant = {variant: build_rows(variant) for variant in variants}

    print("PADDOCK PROFITABILITY DIAGNOSTIC")
    print("trigger = LOOSE frozen rule with odds <= 15")
    print("bet = 100 yen win flat per triggered horse")
    print()

    for scope, grades in [
        ("ALL grades", None),
        ("G1/G2 only", {"G1", "G2"}),
        ("G2 only", {"G2"}),
    ]:
        print("=" * 84)
        print(scope)
        print("=" * 84)
        base = None
        for variant in variants:
            metrics = evaluate(rows_by_variant[variant], grades)
            if base is None:
                base = metrics
                delta = ""
            else:
                delta = f"  delta_vs_no_paddock={((metrics['roi'] - base['roi']) * 100):+.2f}pp"
            print(_fmt(variant, metrics) + delta)
        print()

    print("=" * 84)
    print("QUALITY GATE COVERAGE")
    print("=" * 84)
    legacy_reports = 0
    usable_reports = 0
    rejected_by_reason = defaultdict(int)
    for path in PADDOCK_CACHE.glob("*.json"):
        reports = _load_json(path, {})
        for report in (reports or {}).values():
            if not isinstance(report, dict) or not report.get("text"):
                continue
            legacy_reports += 1
            quality = pq.assess_paddock_report(report.get("text", ""), report.get("source", ""))
            if quality.get("usable"):
                usable_reports += 1
            else:
                for reason in quality.get("reasons", []):
                    rejected_by_reason[reason] += 1
    print(f"legacy_reports={legacy_reports} usable_after_gate={usable_reports}")
    for reason, count in sorted(rejected_by_reason.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  rejected {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
