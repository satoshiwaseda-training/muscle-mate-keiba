"""Per-horse level signal analysis on the 50-race enriched dataset.

For each (race, horse) pair, compute:
  - features
  - is_winner (1/0)
  - market_prob, model_prob, edge = model_prob - market_prob

Then:
  1. Correlate each feature with is_winner (across the whole field).
  2. Check whether (model_prob - market_prob) correlates with is_winner.
  3. Per-venue and per-grade win-rate splits.
  4. Intra-race rank comparisons: does the horse with highest X in the
     race win more than random?
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path
from collections import defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ENR = Path("data/enriched_backtest_results.json")
RES = Path("data/results.json")


def _norm(n): return (n or "").strip()
def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try: return float(s)
    except ValueError: return 0.0


def pearson(xs, ys):
    if not xs or len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def main():
    enriched = json.loads(ENR.read_text(encoding="utf-8"))
    results = json.loads(RES.read_text(encoding="utf-8"))

    # Per-horse rows
    rows = []
    winners_by_race = {}
    venue_map = {}
    for er in enriched:
        rid = er["race_id"]
        res = results.get(f"bt_{rid}")
        if not res: continue
        fo = res.get("finishing_order") or []
        winner = None
        for h in fo:
            try:
                if int(h.get("rank", 0) or 0) == 1:
                    winner = _norm(h.get("name")); break
            except ValueError: continue
        if not winner: continue
        winners_by_race[rid] = winner

        sf_horses = (er.get("structured_features") or {}).get("horses") or {}
        race_info = (er.get("structured_features") or {}).get("race") or {}
        venue = race_info.get("venue", "") or "?"
        venue_map[rid] = venue
        grade = er.get("grade", "")
        # odds map for market prob
        odds_map = {}
        for h in fo:
            nm = _norm(h.get("name"))
            od = _po(h.get("odds"))
            if nm and od > 0 and nm not in odds_map:
                odds_map[nm] = od
        for ranked in er.get("ranked", []):
            name = _norm(ranked.get("name"))
            if name not in sf_horses: continue
            sf_h = sf_horses[name]
            od = odds_map.get(name, 0)
            rows.append({
                "race_id": rid,
                "venue": venue,
                "grade": grade,
                "name": name,
                "win_prob": ranked.get("win_prob", 0),
                "score": ranked.get("score", 0),
                "odds": od,
                "market_prob": (1.0 / (od * 1.20)) if od > 0 else 0,
                "is_winner": 1 if name == winner else 0,
                "sf": sf_h,
            })

    print(f"Total horse-race rows: {len(rows)}")
    print(f"Wins: {sum(r['is_winner'] for r in rows)}")
    print()

    FEATS = [
        "jockey_win_rate", "jockey_g1_wins",
        "horse_weight_delta", "horse_weight_kg", "carried_weight",
        "age", "transport_stress",
        "training_acceleration", "training_cardio_index",
        "paddock_vascularity", "paddock_hindquarter", "paddock_gait",
        "waku", "number",
    ]

    def safe_num(v):
        try:
            if isinstance(v, (int, float)):
                return float(v)
            return float(str(v).strip())
        except (ValueError, TypeError):
            return 0.0

    # 1. Correlations with is_winner
    print("=" * 96)
    print("Per-horse Pearson correlation with is_winner (across whole field, 50 races)")
    print("=" * 96)
    ys = [r["is_winner"] for r in rows]
    for feat in FEATS:
        xs = [safe_num(r["sf"].get(feat, 0)) for r in rows]
        cov_n = sum(1 for x in xs if x != 0)
        corr = pearson(xs, ys)
        print(f"  {feat:30} r={corr:+.4f}  coverage={cov_n}/{len(xs)} ({cov_n/len(xs)*100:.0f}%)")

    # Also test odds-derived and model-derived predictors
    for lbl, key in [("win_prob (model)", "win_prob"), ("market_prob (1/odds)", "market_prob")]:
        xs = [r[key] for r in rows]
        corr = pearson(xs, ys)
        print(f"  {lbl:30} r={corr:+.4f}")

    # Edge correlation
    edge = [(r["win_prob"] - r["market_prob"]) for r in rows]
    corr = pearson(edge, ys)
    print(f"  {'edge (model-market)':30} r={corr:+.4f}")

    # 2. Intra-race rank test: does "top X in this race" win more than 1/N?
    print("\n" + "=" * 96)
    print("Does 'highest in-race X' win more often than baseline (1/N per field)?")
    print("=" * 96)
    # Group rows by race
    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_id"]].append(r)

    baseline_rate = sum(1 / len(rs) for rs in by_race.values()) / len(by_race)
    print(f"  Baseline win rate per top horse = 1/field_size avg = {baseline_rate*100:.2f}%")
    print()

    for feat in FEATS + ["win_prob", "market_prob"]:
        hits = 0
        n = 0
        for rid, rs in by_race.items():
            def getv(r):
                if feat in ("win_prob", "market_prob"):
                    return r[feat]
                return safe_num(r["sf"].get(feat, 0))
            if not rs: continue
            best = max(rs, key=getv)
            if getv(best) == 0:
                continue
            n += 1
            if best["is_winner"]:
                hits += 1
        if n == 0:
            print(f"  {feat:30} (no non-zero feature)")
            continue
        rate = hits / n
        lift = rate / baseline_rate if baseline_rate else 0
        print(f"  {feat:30} races={n:>3}  hit={hits:>2}/{n} ({rate*100:5.1f}%)  lift={lift:+.2f}x")

    # 3. Venue / grade splits
    print("\n" + "=" * 96)
    print("Per-venue win rate of odds-favorite (to see regime variance)")
    print("=" * 96)
    by_venue = defaultdict(lambda: {"n": 0, "fav_wins": 0})
    for rid, venue in venue_map.items():
        rs = by_race[rid]
        winner = winners_by_race[rid]
        odds_fav = min((r for r in rs if r["odds"] > 0), key=lambda r: r["odds"], default=None)
        if odds_fav is None: continue
        by_venue[venue]["n"] += 1
        if odds_fav["is_winner"]:
            by_venue[venue]["fav_wins"] += 1
    for v, d in sorted(by_venue.items(), key=lambda kv: -kv[1]["n"]):
        if d["n"] < 2: continue
        rate = d["fav_wins"] / d["n"] * 100
        print(f"  {v:8} n={d['n']:>3}  fav_wins={d['fav_wins']:>3}  rate={rate:5.1f}%")


if __name__ == "__main__":
    main()
