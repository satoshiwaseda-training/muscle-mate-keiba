"""Weekly performance report.

Aggregates predictions from the most recent Sat-Sun pair and writes a
JSON summary plus a human-readable CLI printout.

Scope rules:
  - Run on Monday (normal) → last weekend (yesterday Sun + day-before Sat)
  - Run on Sunday          → today's Sat-Sun window (partial, Sat complete)
  - Run on Saturday        → previous weekend (fully complete)
  - Run mid-week (forced)  → most recent completed weekend

Output:
  - CLI printout
  - data/weekly_reports/<sat_date>_to_<sun_date>.json

Usage:
  python tools/weekly_report.py
  python tools/weekly_report.py --date 2026-04-11
  python tools/weekly_report.py --force

────────────────────────────────────────────────────────────────────
EVALUATION RULES (frozen 2026-04 — do NOT change without approval)
────────────────────────────────────────────────────────────────────

A prediction record can exist in one of two stages:
  early  — prediction made when netkeiba had not yet published odds;
           odds_status is "not-published-yet" or "all-zero-no-source";
           ranking is effectively fact-layer-only and is NOT
           representative of what would actually be bet.
  final  — prediction made after odds became available (shutuba-ok,
           partial-kept, or successfully injected from live-odds API
           / result page); this is the prediction that a human would
           actually place a bet on.

This report applies three independent rules to these stages:

  RULE 1 — AUDIT RETENTION (監査用保持)
    BOTH early and final records are persisted forever in
    live_predictions.json, and BOTH are surfaced by this report in the
    `volume.by_stage_counts` field. We never delete early records,
    because their divergence from final is itself a signal.

  RULE 2 — ROI EVALUATION SOURCE (ROI評価)
    For the time being, the headline block (`headline.win_roi`,
    `headline.win_hit_rate`, etc.) is computed from EVERY record with a
    result attached, regardless of stage. This is a known approximation
    and WILL BE SPLIT once the `by_stage.final` placeholder is filled.
    The operational truth is:

      → ROI trust level is proportional to the `final` share of the
        record set. If by_stage_counts.final >> by_stage_counts.early,
        headline numbers are trustworthy. Otherwise, treat them as
        contaminated and look at by_stage.final when it is implemented.

  RULE 3 — EARLY AS REFERENCE ONLY (参考値扱い)
    Early predictions must NEVER be used to tune thresholds or promote
    a strategy to production. They exist only to answer:
      "Did giving the model odds change what it would have picked?"
    The comparison is by inspecting a race's `history[]` block in
    live_predictions.json, not by aggregation here.

  INVARIANT:
    No code path in this report may silently drop early records, and
    no code path may mix early+final into a single ROI number that is
    presented as "the" ROI without a disclaimer.

The `by_stage` block in the JSON output is reserved for the future
split (`by_stage.early`, `by_stage.final`). It exists today as an
empty placeholder so downstream consumers can `.get("by_stage", {})`
safely before the aggregation is implemented.
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._autolog_utils import (
    WEEKLY_REPORT_DIR, banner, build_parser, ensure_project_on_path,
    last_weekend, log, parse_date,
)

ensure_project_on_path()

import prediction_log as plog  # noqa: E402


def _parse_int(s) -> int:
    try:
        return int(str(s).replace(",", "").replace("円", "").strip())
    except Exception:
        return 0


def _norm(s) -> str:
    return (s or "").strip()


def _winner_of(result: dict) -> str | None:
    fo = (result or {}).get("finishing_order") or []
    for h in fo:
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return _norm(h.get("name"))
        except ValueError:
            pass
    return None


def _odds_favorite_of(result: dict) -> str | None:
    fo = (result or {}).get("finishing_order") or []
    best = None
    for h in fo:
        try:
            od = float(str(h.get("odds", 0)).replace("---", "0"))
        except Exception:
            od = 0.0
        if od > 1.0 and (best is None or od < best[1]):
            best = (_norm(h.get("name")), od)
    return best[0] if best else None


def _conf_bucket(p: float) -> str:
    if p >= 0.70: return "≥70%"
    if p >= 0.60: return "60-69%"
    if p >= 0.50: return "50-59%"
    return "<50%"


def _pop_bucket(pop: int) -> str:
    if pop == 1: return "1番人気"
    if pop in (2, 3): return "2-3番人気"
    return "4番人気以下"


def _odds_bucket(o: float) -> str:
    if o < 2.0: return "<2.0"
    if o < 3.5: return "2.0-3.5"
    if o < 6.0: return "3.5-6"
    if o < 12.0: return "6-12"
    if o < 30.0: return "12-30"
    return "30+"


def _roi(cost: float, payout: float) -> float:
    return (payout - cost) / cost if cost > 0 else 0.0


def build_report(sat: dt.date, sun: dt.date) -> dict:
    """Compute all weekly KPIs from persisted predictions in [sat, sun].

    Output schema reserves a `by_stage` block with two empty sub-blocks
    (`early` / `final`) so that a future aggregation step can split ROI
    by prediction_stage without breaking existing consumers. Today we
    only populate the top-level headline metrics; the stage blocks are
    intentionally left empty but present so the JSON schema is stable.
    """
    sat_iso, sun_iso = sat.isoformat(), sun.isoformat()
    all_preds = plog.list_predictions(only_live=True)
    scoped = [
        e for e in all_preds
        if sat_iso <= (e.get("race_date") or "") <= sun_iso
    ]
    with_result = [e for e in scoped if (e.get("result") or {}).get("finishing_order")]

    total_preds = len(scoped)
    total_with_result = len(with_result)

    # Count by stage so the headline block can show the split even before
    # stage-scoped ROI is implemented. Missing stage is treated as "final"
    # (legacy records from before the stage field existed).
    stage_counts = {"early": 0, "final": 0}
    for e in scoped:
        s = e.get("prediction_stage") or "final"
        stage_counts[s] = stage_counts.get(s, 0) + 1

    # Accumulators
    win_hits = 0
    place_hits = 0
    odds_fav_hits = 0
    model_follows_fav = 0
    win_cost = 0.0
    win_payout = 0.0

    by_conf = defaultdict(lambda: [0, 0, 0.0, 0.0])  # [n, wins, cost, payout]
    by_pop = defaultdict(lambda: [0, 0, 0.0, 0.0])
    by_grade = defaultdict(lambda: [0, 0, 0.0, 0.0])
    by_odds = defaultdict(lambda: [0, 0, 0.0, 0.0])

    per_race_pnl: list[tuple[str, str, float, float, bool]] = []

    for e in with_result:
        ranked = e.get("ranked") or []
        if not ranked:
            continue
        top1 = ranked[0]
        top1_name = _norm(top1.get("name"))
        top1_prob = float(top1.get("win_prob", 0) or 0)
        top1_odds = float(top1.get("odds", 0) or 0)

        result = e.get("result") or {}
        winner = _winner_of(result)
        if not winner:
            continue
        odds_fav = _odds_favorite_of(result)
        pay = result.get("payouts") or {}
        win_pay = float(_parse_int(pay.get("単勝", 0)))

        # Place (top3) detection
        fo = result.get("finishing_order") or []
        top3_names = set()
        for h in fo:
            try:
                r = int(h.get("rank", 0) or 0)
            except ValueError:
                r = 0
            if 1 <= r <= 3:
                top3_names.add(_norm(h.get("name")))

        # Popularity of model top1 among running horses
        running = [
            (_norm(h.get("name")), float(str(h.get("odds", 0)).replace("---", "0") or 0))
            for h in fo
        ]
        running = [(n, o) for n, o in running if o > 1.0]
        running.sort(key=lambda x: x[1])
        pop_map = {n: i + 1 for i, (n, _) in enumerate(running)}
        top1_pop = pop_map.get(top1_name, 99)

        is_win = (top1_name == winner)
        is_place = (top1_name in top3_names)
        if is_win:
            win_hits += 1
            win_payout += win_pay
        if is_place:
            place_hits += 1
        if odds_fav == winner:
            odds_fav_hits += 1
        if odds_fav and top1_name == odds_fav:
            model_follows_fav += 1
        win_cost += 100.0

        # Bucket aggregation
        cb = _conf_bucket(top1_prob)
        by_conf[cb][0] += 1; by_conf[cb][2] += 100
        if is_win:
            by_conf[cb][1] += 1; by_conf[cb][3] += win_pay

        pb = _pop_bucket(top1_pop)
        by_pop[pb][0] += 1; by_pop[pb][2] += 100
        if is_win:
            by_pop[pb][1] += 1; by_pop[pb][3] += win_pay

        g = e.get("grade") or "その他"
        g_key = g if g in ("G1", "G2", "G3") else "その他"
        by_grade[g_key][0] += 1; by_grade[g_key][2] += 100
        if is_win:
            by_grade[g_key][1] += 1; by_grade[g_key][3] += win_pay

        ob = _odds_bucket(top1_odds)
        by_odds[ob][0] += 1; by_odds[ob][2] += 100
        if is_win:
            by_odds[ob][1] += 1; by_odds[ob][3] += win_pay

        # Per-race PnL for best/worst
        pnl = (win_pay - 100.0) if is_win else -100.0
        per_race_pnl.append((
            e.get("race_id", ""),
            e.get("race_name", "") or "",
            top1_odds,
            pnl,
            is_win,
        ))

    # Derived
    def _rate(num, den):
        return (num / den) if den else 0.0

    win_rate = _rate(win_hits, total_with_result)
    place_rate = _rate(place_hits, total_with_result)
    odds_fav_rate = _rate(odds_fav_hits, total_with_result)
    follow_fav_rate = _rate(model_follows_fav, total_with_result)
    win_roi = _roi(win_cost, win_payout)

    # Best/worst 3 races by PnL
    sorted_by_pnl = sorted(per_race_pnl, key=lambda t: t[3])
    worst3 = sorted_by_pnl[:3]
    best3 = list(reversed(sorted_by_pnl[-3:]))

    def _to_bucket(d):
        return {
            k: {
                "n": v[0],
                "wins": v[1],
                "win_rate": round(_rate(v[1], v[0]), 4),
                "cost": v[2],
                "payout": v[3],
                "roi": round(_roi(v[2], v[3]), 4),
            }
            for k, v in sorted(d.items())
        }

    # Empty placeholders for the stage-scoped aggregations.
    # Schema is frozen so a later PR can fill these in without breaking
    # downstream consumers. Do NOT delete the keys even when empty.
    EMPTY_STAGE_BLOCK = {
        "n_preds": 0,
        "n_with_result": 0,
        "win_hit_rate": None,
        "place_hit_rate": None,
        "win_roi": None,
        "win_cost_yen": 0,
        "win_payout_yen": 0,
        "win_pnl_yen": 0,
        "note": "aggregation not implemented yet (schema placeholder)",
    }

    return {
        "period": {
            "sat": sat_iso,
            "sun": sun_iso,
            "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        },
        "volume": {
            "total_predictions": total_preds,
            "total_with_result": total_with_result,
            "attach_rate": round(_rate(total_with_result, total_preds), 4),
            "by_stage_counts": stage_counts,
        },
        "headline": {
            "win_hit_rate": round(win_rate, 4),
            "place_hit_rate": round(place_rate, 4),
            "win_roi": round(win_roi, 4),
            "win_cost_yen": win_cost,
            "win_payout_yen": win_payout,
            "win_pnl_yen": win_payout - win_cost,
        },
        # ── Reserved schema for future stage-split aggregation ──
        # Consumers can safely .get("by_stage", {}).get("final", {}) without
        # KeyErrors. Populated by a later implementation once enough early
        # predictions exist to make the split statistically meaningful.
        "by_stage": {
            "early": dict(EMPTY_STAGE_BLOCK),
            "final": dict(EMPTY_STAGE_BLOCK),
        },
        "market_comparison": {
            "odds_favorite_win_rate": round(odds_fav_rate, 4),
            "model_follow_fav_rate": round(follow_fav_rate, 4),
            "model_minus_market_win_rate": round(win_rate - odds_fav_rate, 4),
        },
        "by_popularity": _to_bucket(by_pop),
        "by_confidence": _to_bucket(by_conf),
        "by_grade": _to_bucket(by_grade),
        "by_odds": _to_bucket(by_odds),
        "worst_3": [
            {"race_id": r, "race_name": n, "odds": o, "pnl": p, "hit": h}
            for r, n, o, p, h in worst3
        ],
        "best_3": [
            {"race_id": r, "race_name": n, "odds": o, "pnl": p, "hit": h}
            for r, n, o, p, h in best3
        ],
    }


def print_report(rep: dict) -> None:
    p = rep["period"]
    v = rep["volume"]
    h = rep["headline"]
    mc = rep["market_comparison"]

    banner(f"Weekly Report  {p['sat']} ～ {p['sun']}")
    print(f"  generated_at     : {p['generated_at']}")
    print()
    print("── volume ──")
    print(f"  total_predictions: {v['total_predictions']}")
    print(f"  total_with_result: {v['total_with_result']}")
    print(f"  attach_rate      : {v['attach_rate']*100:.1f}%")
    stage_counts = v.get("by_stage_counts") or {}
    if stage_counts:
        print(f"  stage split      : early={stage_counts.get('early', 0)}  "
              f"final={stage_counts.get('final', 0)}")
    print()
    print("── evaluation rule (frozen) ──")
    print("  audit retention  : BOTH early and final are kept in live_predictions.json")
    print("  ROI source       : headline metrics use ALL staged records (interim)")
    print("                     → trust only when by_stage_counts.final dominates")
    print("  early usage      : reference only — never drives tuning")
    if v["total_with_result"] == 0:
        print()
        print("  [!] No predictions with results. Report ends here.")
        return
    print()
    print("── headline ──")
    print(f"  win hit rate     : {h['win_hit_rate']*100:.2f}%")
    print(f"  place hit rate   : {h['place_hit_rate']*100:.2f}%")
    print(f"  win ROI (100¥)   : {h['win_roi']*100:+.2f}%  "
          f"(cost={h['win_cost_yen']:,.0f}  payout={h['win_payout_yen']:,.0f}  "
          f"pnl={h['win_pnl_yen']:+,.0f})")
    print()
    print("── market comparison ──")
    print(f"  odds-favorite win rate : {mc['odds_favorite_win_rate']*100:.2f}%")
    print(f"  model follows market   : {mc['model_follow_fav_rate']*100:.2f}%  "
          f"(1番人気をそのまま選んだ割合)")
    print(f"  model − market (win%)  : {mc['model_minus_market_win_rate']*100:+.2f} pt")
    print()

    def _print_bucket(title: str, d: dict, order: list[str]) -> None:
        print(f"── {title} ──")
        for k in order:
            if k in d:
                b = d[k]
                print(f"  {k:12} n={b['n']:>3}  "
                      f"win%={b['win_rate']*100:>5.1f}  "
                      f"ROI={b['roi']*100:+6.1f}%")
        print()

    _print_bucket("by popularity", rep["by_popularity"],
                  ["1番人気", "2-3番人気", "4番人気以下"])
    _print_bucket("by confidence", rep["by_confidence"],
                  ["≥70%", "60-69%", "50-59%", "<50%"])
    _print_bucket("by grade", rep["by_grade"],
                  ["G1", "G2", "G3", "その他"])
    _print_bucket("by odds band", rep["by_odds"],
                  ["<2.0", "2.0-3.5", "3.5-6", "6-12", "12-30", "30+"])

    print("── best 3 races (PnL) ──")
    for r in rep["best_3"]:
        mark = "WIN" if r["hit"] else "."
        print(f"  {r['race_id']:>14}  {r['race_name'][:24]:<24}  "
              f"odds={r['odds']:>5.1f}  pnl={r['pnl']:+,.0f}  {mark}")
    print()
    print("── worst 3 races (PnL) ──")
    for r in rep["worst_3"]:
        mark = "WIN" if r["hit"] else "."
        print(f"  {r['race_id']:>14}  {r['race_name'][:24]:<24}  "
              f"odds={r['odds']:>5.1f}  pnl={r['pnl']:+,.0f}  {mark}")


def main() -> int:
    parser = build_parser("Weekly weekend performance report.")
    args = parser.parse_args()

    today = dt.date.today()
    anchor = parse_date(args.date) or today
    sat, sun = last_weekend(anchor)

    rep = build_report(sat, sun)
    print_report(rep)

    # Persist JSON atomically so a kill mid-write cannot leave the file
    # as a truncated half-JSON (consistent with prediction_log._save).
    import os as _os
    WEEKLY_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = WEEKLY_REPORT_DIR / f"{sat.isoformat()}_to_{sun.isoformat()}.json"
    tmp = out.with_name(out.name + ".tmp")
    payload = json.dumps(rep, ensure_ascii=False, indent=2)
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
        f.flush()
        try:
            _os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    _os.replace(tmp, out)
    log(f"report saved: {out}")

    if rep["volume"]["total_predictions"] == 0:
        log(
            f"CRITICAL: zero predictions in week {sat.isoformat()}-{sun.isoformat()}. "
            f"weekend_autolog may have failed. Check data/autolog/.",
            level="ERROR",
        )
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
