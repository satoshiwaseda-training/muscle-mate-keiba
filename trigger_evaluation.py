"""Trigger-only betting evaluation.

Runs the full fact pipeline offline across all 121 cached races and
applies several trigger-severity definitions to answer:

  "Does betting ONLY on trigger horses produce positive expectation
   vs flat-betting the odds favorite?"

Trigger levels tested:

  STRICT      : consensus ≥ 3 AND composite ≥ 0.7 AND no strong negative
                (the production `dual_mode_scoring` criterion)
  MEDIUM      : consensus ≥ 2 AND composite ≥ 0.65 AND no strong negative
  LOOSE       : consensus ≥ 1 AND composite ≥ 0.60 AND no strong negative
  FACT_TOP    : any horse whose fact_score beats its odds_score by ≥ 10%
  FAV_DEV     : any race where model top ≠ odds favorite (deviation-only)

Baselines:
  FLAT_FAV    : bet 1 unit on the lowest-odds horse every race
  FLAT_MODEL  : bet 1 unit on the model top-1 every race

Per-level metrics:
  n_bets, wins, win_rate, cost (yen), payout (yen), pnl, roi,
  avg_win_payout, avg_bet_odds
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import fact_extractor as fe
import fact_schema
import fact_validator as fv
import paddock_features as pf
import dual_mode_scoring as dm

ROOT = Path(__file__).parent
CACHE = ROOT / "data" / "scraper_cache" / "enrich_race"
V3 = ROOT / "data" / "enriched_backtest_results_v3.json"
RES = ROOT / "data" / "results.json"

TICKET = 100


def _norm(n): return (n or "").strip()


def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try: return float(s)
    except ValueError: return 0.0


# ── Fact extraction (mirrors run_dual_mode) ──────────

def horse_facts(name, h_sf, cached_horse, page_text, all_names):
    facts = []
    facts.extend(fe.fact_from_weight_delta(name, h_sf.get("horse_weight_delta")))
    segs = pf.extract_per_horse_comments(page_text, all_names) if page_text else {}
    seg = segs.get(name)
    if seg and seg.get("comment"):
        facts.extend(fe.extract_canonical_facts(seg["comment"], "news", horse=name))
    if cached_horse:
        raw = cached_horse.get("paddock_comment") or ""
        if raw:
            idx = raw.find(name)
            if idx >= 0:
                facts.extend(fe.extract_canonical_facts(
                    raw[idx: idx + 200], "keibalab", horse=name,
                ))
        te = cached_horse.get("training_eval") or ""
        if te and len(te) >= 3:
            facts.extend(fe.extract_canonical_facts(
                te, "netkeiba_oikiri", horse=name,
            ))
    return facts


# ── Per-horse metrics cache ───────────────────────────

def build_per_race_metrics():
    """Walk each cached race, return one dict per race with:
      horses: list of {name, odds, consensus_count, composite, state_scores,
                        is_winner, payout_if_winner, odds_fav, is_odds_fav}
    """
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    results = json.loads(RES.read_text(encoding="utf-8"))
    rows = []
    for row in v3:
        rid = row["race_id"]
        res = results.get(f"bt_{rid}")
        if not res:
            continue
        fo = res.get("finishing_order") or []
        winner = None
        for h in fo:
            try:
                if int(h.get("rank", 0) or 0) == 1:
                    winner = _norm(h.get("name")); break
            except ValueError: pass
        if not winner:
            continue
        payouts = res.get("payouts") or {}
        win_payout = float(payouts.get("単勝", 0) or 0)

        sf_horses = row["structured_features"].get("horses") or {}
        all_names = list(sf_horses.keys())

        cache_file = CACHE / f"{rid}.json"
        cached = []
        page_text = ""
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            for h in cached:
                pt = h.get("paddock_comment") or ""
                if len(pt) > len(page_text): page_text = pt

        odds_map = {}
        for h in fo:
            nm = _norm(h.get("name"))
            od = _po(h.get("odds"))
            if nm and od > 0 and nm not in odds_map:
                odds_map[nm] = od
        if not odds_map:
            continue
        odds_fav = min(odds_map, key=lambda n: odds_map[n])

        horses = []
        model_top_name = row["ranked"][0]["name"] if row["ranked"] else None
        for name in all_names:
            h_sf = sf_horses[name]
            cached_h = next((c for c in cached if _norm(c.get("name")) == name), None)
            raw = horse_facts(name, h_sf, cached_h, page_text, all_names)
            validated = fv.validate_and_transform(raw)
            merged = fe.merge_fact_layers(validated)
            agg = fe.aggregate_horse_score(merged)
            states = fv.compute_state_scores(merged)
            negatives = [f for f in merged if f.polarity < 0]
            horses.append({
                "name": name,
                "odds": h_sf.get("odds", 0),
                "consensus_count": agg["consensed_fact_count"],
                "composite": agg["composite_condition"],
                "composite_all": agg["composite_condition_all"],
                "n_facts": agg["n_facts"],
                "fatigue_score": states.get("fatigue_score", 0),
                "stress_score": states.get("stress_score", 0),
                "pain_risk": states.get("pain_risk", 0),
                "is_winner": name == winner,
                "win_payout": win_payout if name == winner else 0,
                "is_odds_fav": name == odds_fav,
                "is_model_top": name == model_top_name,
                "strong_negative_present": any(
                    float(f.confidence) > 0.6 for f in negatives
                ),
            })
        rows.append({
            "race_id": rid, "race_name": row.get("race_name", ""),
            "grade": row.get("grade", ""),
            "winner": winner, "odds_fav": odds_fav,
            "odds_fav_odds": odds_map[odds_fav],
            "win_payout": win_payout,
            "horses": horses,
        })
    return rows


# ── Trigger definitions ───────────────────────────────

def trigger_strict(h):
    return (h["consensus_count"] >= 3
            and h["composite"] >= 0.7
            and not h["strong_negative_present"])


def trigger_medium(h):
    return (h["consensus_count"] >= 2
            and h["composite"] >= 0.65
            and not h["strong_negative_present"])


def trigger_loose(h):
    return (h["consensus_count"] >= 1
            and h["composite"] >= 0.60
            and not h["strong_negative_present"])


def trigger_any_consensed(h):
    return h["consensus_count"] >= 1 and not h["strong_negative_present"]


def trigger_low_fatigue_high_condition(h):
    return (h["composite"] >= 0.6
            and h["fatigue_score"] < 0.2
            and h["stress_score"] < 0.2)


def trigger_loose_capped(h):
    """LOOSE but excluding extreme longshots (odds ≤ 15)."""
    return (h["consensus_count"] >= 1
            and h["composite"] >= 0.60
            and h["odds"] <= 15.0
            and not h["strong_negative_present"])


def trigger_loose_midpack(h):
    """LOOSE restricted to the 'value' odds band [3.0, 10.0]."""
    return (h["consensus_count"] >= 1
            and h["composite"] >= 0.60
            and 3.0 <= h["odds"] <= 10.0
            and not h["strong_negative_present"])


def trigger_clean_signal(h):
    """LOOSE + low fatigue/stress/pain — positive signal with no concerns."""
    return (h["consensus_count"] >= 1
            and h["composite"] >= 0.60
            and h["fatigue_score"] < 0.30
            and h["stress_score"] < 0.30
            and h["pain_risk"] < 0.20
            and not h["strong_negative_present"])


LEVELS: list[tuple[str, Callable[[dict], bool]]] = [
    ("STRICT (cons≥3, comp≥0.70)",            trigger_strict),
    ("MEDIUM (cons≥2, comp≥0.65)",            trigger_medium),
    ("LOOSE  (cons≥1, comp≥0.60)",            trigger_loose),
    ("LOOSE+odds≤15  (no extreme longshot)",  trigger_loose_capped),
    ("LOOSE+value 3≤odds≤10",                 trigger_loose_midpack),
    ("LOOSE+clean (low fat/str/pain)",        trigger_clean_signal),
    ("CONSENSED-ANY (cons≥1)",                 trigger_any_consensed),
    ("LOW-FATIGUE (comp≥.6, fat/str<.2)",     trigger_low_fatigue_high_condition),
]


# ── Evaluation ─────────────────────────────────────────

def eval_strategy(rows: list[dict], selector: Callable[[dict], bool]) -> dict:
    """Bet 1 unit on every horse that matches `selector`. Returns KPIs."""
    n_bets = 0
    wins = 0
    cost = 0.0
    payout = 0.0
    bet_odds: list[float] = []
    for row in rows:
        for h in row["horses"]:
            if selector(h):
                n_bets += 1
                cost += TICKET
                bet_odds.append(h["odds"])
                if h["is_winner"]:
                    wins += 1
                    payout += h["win_payout"]
    if n_bets == 0:
        return {"n_bets": 0, "wins": 0, "win_rate": 0.0,
                "cost": 0, "payout": 0, "pnl": 0, "roi": 0.0,
                "avg_bet_odds": 0.0, "avg_win_payout": 0.0}
    return {
        "n_bets": n_bets,
        "wins": wins,
        "win_rate": wins / n_bets,
        "cost": cost,
        "payout": payout,
        "pnl": payout - cost,
        "roi": (payout - cost) / cost,
        "avg_bet_odds": sum(bet_odds) / len(bet_odds),
        "avg_win_payout": payout / wins if wins else 0.0,
    }


def _fmt(m: dict) -> str:
    if not m["n_bets"]:
        return "(no bets — sample too small)"
    return (f"n={m['n_bets']:4d}  win={m['wins']:3d}/{m['n_bets']:<3d} "
            f"({m['win_rate']*100:5.1f}%)  ROI={m['roi']*100:+7.2f}%  "
            f"avg_odds={m['avg_bet_odds']:5.2f}  "
            f"avg_payout={m['avg_win_payout']:6.0f}")


def main():
    print("Loading & extracting facts across 121 cached races...")
    rows = build_per_race_metrics()
    print(f"Loaded {len(rows)} races with results\n")

    # ── Baselines ──
    print("=" * 96)
    print("BASELINES")
    print("=" * 96)
    fav = eval_strategy(rows, lambda h: h["is_odds_fav"])
    top = eval_strategy(rows, lambda h: h["is_model_top"])
    print(f"  FLAT odds-favorite   : {_fmt(fav)}")
    print(f"  FLAT model top-1     : {_fmt(top)}")

    # ── Trigger levels ──
    print("\n" + "=" * 96)
    print("TRIGGER-ONLY STRATEGIES")
    print("=" * 96)
    level_results = []
    for label, selector in LEVELS:
        m = eval_strategy(rows, selector)
        level_results.append((label, m))
        print(f"  {label:40s}  {_fmt(m)}")

    # ── Summary table ──
    print("\n" + "=" * 96)
    print("SUMMARY vs baseline (FLAT odds-favorite)")
    print("=" * 96)
    fav_win = fav["win_rate"]
    fav_roi = fav["roi"]
    print(f"{'level':42s} {'n':>5s} {'win%':>7s} {'Δwin%':>8s} {'ROI%':>8s} {'Δroi%':>8s}")
    print("-" * 96)
    for label, m in level_results:
        if not m["n_bets"]:
            print(f"{label:42s} {'0':>5s} {'—':>7s} {'—':>8s} {'—':>8s} {'—':>8s}")
            continue
        print(f"{label:42s} {m['n_bets']:>5d} "
              f"{m['win_rate']*100:>6.1f}% "
              f"{(m['win_rate']-fav_win)*100:>+7.1f}% "
              f"{m['roi']*100:>+7.2f}% "
              f"{(m['roi']-fav_roi)*100:>+7.2f}%")

    # ── Sample-size adequacy ──
    print("\n" + "=" * 96)
    print("SAMPLE-SIZE ADEQUACY (target: ≥ 100 bets per strategy)")
    print("=" * 96)
    for label, m in level_results:
        icon = "✅" if m["n_bets"] >= 100 else "⚠" if m["n_bets"] >= 30 else "❌"
        ci_halfwidth = 1.96 * (m["win_rate"] * (1 - m["win_rate"]) / max(m["n_bets"], 1)) ** 0.5 if m["n_bets"] else 0
        print(f"  {icon} {label:42s} n={m['n_bets']:>4d}  "
              f"win-rate 95% CI ± {ci_halfwidth*100:.1f}pp")


if __name__ == "__main__":
    main()
