"""Audit: old vs new scoring on the 121-race backtest cache.

Computes win rate, place rate, ROI (単勝/複勝), confidence-band hit rates,
popularity-band hit rates, grade-band hit rates, and odds-band hit rates,
for BOTH pre-scale-fix (gain=1.0, k=1.5) and post-scale-fix (gain=2.0, k=4.0).

Data sources:
  data/enriched_backtest_results_v3.json — 121 races, structured_features + horses
  data/results.json                       — ground truth (finishing_order + payouts)
                                             keyed as "bt_<race_id>"

No speculation — everything is derived from these files.
"""

import json
import math
import sys
import statistics as st
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def score_runner_audit(horse, num_horses, grade, race, gain):
    W_FIELD = 0.048
    W_GRADE = 0.02
    W_JOCKEY = 0.05
    W_TRAINING = 0.03
    W_WEIGHT_P = 0.04
    W_BIO = 0.03
    W_CONSENSUS = 0.04
    W_IX_JG = 0.03
    W_IX_TB = 0.02

    odds = horse.get("odds", 0.0) or 0.0
    if odds > 1.0:
        base = _clamp((1.0 / odds) / 1.20, 0.02, 0.80)
    else:
        base = _clamp(1.0 / max(num_horses, 1), 0.02, 0.80)

    n_field = _clamp((18 - num_horses) / 10.0, 0.0, 1.0)
    n_grade = 1.0 if grade == "G1" else (0.5 if grade == "G2" else 0.0)
    n_jockey = _clamp(float(horse.get("jockey_win_rate", 0) or 0) / 0.25, 0.0, 1.0)
    tacc = float(horse.get("training_acceleration", 0) or 0)
    n_training = _clamp((tacc + 0.15) / 0.30, 0.0, 1.0)
    abs_delta = abs(float(horse.get("horse_weight_delta", 0) or 0))
    excess = _clamp(abs_delta - 4.0, 0.0, 16.0)
    n_weight_penalty = (excess / 16.0) ** 2
    pad_v = float(horse.get("paddock_vascularity", 0) or 0)
    pad_h = float(horse.get("paddock_hindquarter", 0) or 0)
    pad_g = float(horse.get("paddock_gait", 0) or 0)
    bio_raw = (pad_v + pad_h + pad_g) / 3.0
    n_bio = _clamp((bio_raw + 1.0) / 2.0, 0.0, 1.0)
    n_consensus = 0.5

    track = race.get("track_condition", "")
    cardio = float(horse.get("training_cardio_index", 0) or 0)
    n_cardio = _clamp(cardio, 0.0, 1.0)
    cond_track = 0.04 * n_cardio if track in ("重", "不良") else 0.01 * n_cardio

    adjustment = (
        W_FIELD * n_field
        + W_GRADE * n_grade
        + W_JOCKEY * n_jockey
        + W_TRAINING * n_training
        - W_WEIGHT_P * n_weight_penalty
        + W_BIO * n_bio
        + W_CONSENSUS * (n_consensus - 0.5)
        + W_IX_JG * (n_jockey * n_grade)
        + W_IX_TB * (n_training * n_bio)
        + cond_track
    )
    final = _clamp(base + gain * adjustment, 0.02, 0.95)
    return final, final - base, base


def calibrate(running, k):
    """running: list of (name, odds, structured_edge). Returns list of win_probs."""
    implied = [1.0 / o for _, o, _ in running]
    tot = sum(implied) or 1.0
    base_p = [i / tot for i in implied]
    mult = [math.exp(k * e) for _, _, e in running]
    adj = [b * m for b, m in zip(base_p, mult)]
    ts = sum(adj) or 1.0
    return [a / ts for a in adj]


def parse_int(s):
    try:
        return int(str(s).replace(",", "").replace("円", "").strip())
    except Exception:
        return 0


def audit_config(label, data, results, gain, k):
    """Return a dict of metrics for the given (gain, k) config."""
    rows = []  # per-race rows for table
    win_hits = 0
    place_hits = 0
    top3_contain = 0
    win_cost = 0
    win_payout = 0
    place_cost = 0
    place_payout = 0

    by_conf = defaultdict(lambda: [0, 0, 0.0, 0.0])  # [bets, wins, cost, payout]
    by_pop = defaultdict(lambda: [0, 0, 0.0, 0.0])
    by_grade = defaultdict(lambda: [0, 0, 0.0, 0.0])
    by_odds = defaultdict(lambda: [0, 0, 0.0, 0.0])

    def conf_bucket(p):
        if p >= 0.70: return "≥70%"
        if p >= 0.60: return "60-69%"
        if p >= 0.50: return "50-59%"
        return "<50%"

    def odds_bucket(o):
        if o < 2.0: return "<2.0"
        if o < 3.5: return "2.0-3.5"
        if o < 6.0: return "3.5-6"
        if o < 12.0: return "6-12"
        if o < 30.0: return "12-30"
        return "30+"

    n_evaluated = 0
    for row in data:
        rid = row.get("race_id", "")
        result_key = f"bt_{rid}"
        res = results.get(result_key)
        if not res:
            continue
        fo = res.get("finishing_order") or []
        if not fo:
            continue

        # actual winner + top3
        winner = None
        top3_set = set()
        for h in fo:
            try:
                r = int(h.get("rank", 0) or 0)
            except Exception:
                r = 0
            if r == 1:
                winner = (h.get("name") or "").strip()
            if 1 <= r <= 3:
                top3_set.add((h.get("name") or "").strip())

        if not winner:
            continue

        # Score every horse
        sf = row.get("structured_features", {})
        sf_horses = sf.get("horses", {})
        race = sf.get("race", {})
        grade = row.get("grade", "")
        n = len(sf_horses)

        scored = []
        for name, h in sf_horses.items():
            o = h.get("odds", 0) or 0
            final, edge, base = score_runner_audit(h, n, grade, race, gain)
            scored.append({"name": name.strip(), "odds": o, "final": final, "edge": edge, "base": base})

        running = [(r["name"], r["odds"], r["edge"]) for r in scored if r["odds"] > 1.0]
        if running:
            probs = calibrate(running, k)
            pmap = {nm: p for (nm, _, _), p in zip(running, probs)}
        else:
            pmap = {}
        for r in scored:
            r["win_prob"] = pmap.get(r["name"], 0.0)

        scored.sort(key=lambda r: (-r["win_prob"], -r["final"]))
        top1 = scored[0]
        top1_name = top1["name"]
        top1_prob = top1["win_prob"]
        top1_odds = top1["odds"]

        # popularity: 1 = lowest odds horse
        running_sorted = sorted(
            [r for r in scored if r["odds"] > 1.0], key=lambda r: r["odds"]
        )
        pop_map = {r["name"]: i + 1 for i, r in enumerate(running_sorted)}
        top1_pop = pop_map.get(top1_name, 0)

        # payouts
        pay = res.get("payouts") or {}
        win_pay = parse_int(pay.get("単勝", 0))
        place_pay = parse_int(pay.get("複勝", 0))

        # find place payout for the specific horse (complex - 複勝 can list multiple)
        # Simplified: assume 複勝 payout listed applies to top3 horses. Payoff unknown
        # per-horse from this structure → use flat: if top1 in top3, earn place_pay
        # which is the min payout typically (conservative). Better: skip place ROI if
        # we can't tell which horse the payout is for.
        # We'll compute hit rate for place, but ROI will be approximate.

        # Aggregate
        n_evaluated += 1
        is_win = (top1_name == winner)
        is_place = (top1_name in top3_set)
        if is_win:
            win_hits += 1
            win_payout += win_pay
        if is_place:
            place_hits += 1
            place_payout += place_pay
        if winner in {r["name"] for r in scored[:3]}:
            top3_contain += 1
        win_cost += 100
        place_cost += 100

        # buckets
        cb = conf_bucket(top1_prob)
        by_conf[cb][0] += 1
        by_conf[cb][2] += 100
        if is_win:
            by_conf[cb][1] += 1
            by_conf[cb][3] += win_pay

        pop_key = "1番人気" if top1_pop == 1 else ("2-3番人気" if top1_pop in (2, 3) else "4番人気以下")
        by_pop[pop_key][0] += 1
        by_pop[pop_key][2] += 100
        if is_win:
            by_pop[pop_key][1] += 1
            by_pop[pop_key][3] += win_pay

        g_key = grade if grade in ("G1", "G2", "G3") else "その他"
        by_grade[g_key][0] += 1
        by_grade[g_key][2] += 100
        if is_win:
            by_grade[g_key][1] += 1
            by_grade[g_key][3] += win_pay

        ob = odds_bucket(top1_odds)
        by_odds[ob][0] += 1
        by_odds[ob][2] += 100
        if is_win:
            by_odds[ob][1] += 1
            by_odds[ob][3] += win_pay

        rows.append({
            "race_id": rid,
            "grade": grade,
            "race_name": row.get("race_name", ""),
            "top1": top1_name,
            "win_prob": round(top1_prob, 3),
            "odds": top1_odds,
            "pop": top1_pop,
            "winner": winner,
            "hit": is_win,
            "place": is_place,
            "win_pay": win_pay,
        })

    def roi(cost, payout):
        return (payout - cost) / cost if cost > 0 else 0.0

    metrics = {
        "label": label,
        "gain": gain,
        "k": k,
        "n_races": n_evaluated,
        "win_hit_rate": win_hits / n_evaluated if n_evaluated else 0,
        "place_hit_rate": place_hits / n_evaluated if n_evaluated else 0,
        "top3_contain_winner_rate": top3_contain / n_evaluated if n_evaluated else 0,
        "win_roi": roi(win_cost, win_payout),
        "win_cost": win_cost,
        "win_payout": win_payout,
        "by_confidence": {k: {"n": v[0], "wins": v[1], "win_rate": v[1]/v[0] if v[0] else 0,
                              "roi": roi(v[2], v[3])} for k, v in by_conf.items()},
        "by_popularity": {k: {"n": v[0], "wins": v[1], "win_rate": v[1]/v[0] if v[0] else 0,
                              "roi": roi(v[2], v[3])} for k, v in by_pop.items()},
        "by_grade": {k: {"n": v[0], "wins": v[1], "win_rate": v[1]/v[0] if v[0] else 0,
                         "roi": roi(v[2], v[3])} for k, v in by_grade.items()},
        "by_odds": {k: {"n": v[0], "wins": v[1], "win_rate": v[1]/v[0] if v[0] else 0,
                        "roi": roi(v[2], v[3])} for k, v in by_odds.items()},
        "rows": rows,
    }
    return metrics


def fmt_block_title(t):
    return f"\n── {t} ──"


def print_metrics(m):
    print(fmt_block_title(m["label"]))
    print(f"  config: gain={m['gain']}, k={m['k']}")
    print(f"  races evaluated       : {m['n_races']}")
    print(f"  win hit rate (top1)   : {m['win_hit_rate']*100:.2f}%")
    print(f"  place hit rate (top1) : {m['place_hit_rate']*100:.2f}%")
    print(f"  top3 contain winner   : {m['top3_contain_winner_rate']*100:.2f}%")
    print(f"  win ROI (flat 100yen) : {m['win_roi']*100:+.2f}%  "
          f"(cost={m['win_cost']:,.0f}  payout={m['win_payout']:,.0f})")

    print("\n  by confidence band:")
    order = ["≥70%", "60-69%", "50-59%", "<50%"]
    for key in order:
        if key in m["by_confidence"]:
            v = m["by_confidence"][key]
            print(f"    {key:10} n={v['n']:>3}  win%={v['win_rate']*100:>5.1f}  ROI={v['roi']*100:+.1f}%")

    print("\n  by popularity of top1:")
    for key in ["1番人気", "2-3番人気", "4番人気以下"]:
        if key in m["by_popularity"]:
            v = m["by_popularity"][key]
            print(f"    {key:12} n={v['n']:>3}  win%={v['win_rate']*100:>5.1f}  ROI={v['roi']*100:+.1f}%")

    print("\n  by grade:")
    for key in ["G1", "G2", "G3", "その他"]:
        if key in m["by_grade"]:
            v = m["by_grade"][key]
            print(f"    {key:6} n={v['n']:>3}  win%={v['win_rate']*100:>5.1f}  ROI={v['roi']*100:+.1f}%")

    print("\n  by odds band of top1:")
    for key in ["<2.0", "2.0-3.5", "3.5-6", "6-12", "12-30", "30+"]:
        if key in m["by_odds"]:
            v = m["by_odds"][key]
            print(f"    {key:8} n={v['n']:>3}  win%={v['win_rate']*100:>5.1f}  ROI={v['roi']*100:+.1f}%")


def main():
    data = json.load(open("data/enriched_backtest_results_v3.json", encoding="utf-8"))
    results = json.load(open("data/results.json", encoding="utf-8"))

    print(f"=== AUDIT: 121-race backtest, old vs new scoring ===\n")
    old = audit_config("OLD (pre-scale-fix)", data, results, gain=1.0, k=1.5)
    new = audit_config("NEW (post-scale-fix)", data, results, gain=2.0, k=4.0)

    print_metrics(old)
    print_metrics(new)

    # Side-by-side diff
    print("\n" + "="*60)
    print("SIDE-BY-SIDE DIFF")
    print("="*60)
    def fmt(label, o, n, unit=""):
        if isinstance(o, float):
            print(f"  {label:30} old={o*100:>+7.2f}{unit}  new={n*100:>+7.2f}{unit}  diff={(n-o)*100:>+6.2f}{unit}")
        else:
            print(f"  {label:30} old={o}  new={n}")
    fmt("win hit rate", old["win_hit_rate"], new["win_hit_rate"], "%")
    fmt("place hit rate", old["place_hit_rate"], new["place_hit_rate"], "%")
    fmt("top3-contain-winner", old["top3_contain_winner_rate"], new["top3_contain_winner_rate"], "%")
    fmt("win ROI", old["win_roi"], new["win_roi"], "%")
    print(f"  payout delta                   old={old['win_payout']:>7,.0f}  new={new['win_payout']:>7,.0f}  "
          f"diff={new['win_payout']-old['win_payout']:+,.0f}")

    # Count races where top1 differs
    old_top1 = {r["race_id"]: r["top1"] for r in old["rows"]}
    new_top1 = {r["race_id"]: r["top1"] for r in new["rows"]}
    diff_races = [rid for rid in old_top1 if old_top1[rid] != new_top1.get(rid)]
    print(f"\n  races where top1 differs: {len(diff_races)} / {len(old_top1)}")

    # Among differing races, who was right more often?
    old_right_in_diff = 0
    new_right_in_diff = 0
    for rid in diff_races:
        old_row = next(r for r in old["rows"] if r["race_id"] == rid)
        new_row = next(r for r in new["rows"] if r["race_id"] == rid)
        if old_row["hit"]: old_right_in_diff += 1
        if new_row["hit"]: new_right_in_diff += 1
    print(f"  in diff races: old-method wins={old_right_in_diff}  new-method wins={new_right_in_diff}")

    # Race-by-race table (top 15 by anything interesting - let's show diff races)
    if diff_races:
        print("\n  === races where top1 differs (new vs old) ===")
        print(f"  {'race_id':>14} {'grade':>5} {'old_top1':>18} {'old_hit':>7} {'new_top1':>18} {'new_hit':>7} {'winner':>18}")
        for rid in diff_races[:20]:
            o = next(r for r in old["rows"] if r["race_id"] == rid)
            n = next(r for r in new["rows"] if r["race_id"] == rid)
            print(f"  {rid:>14} {o['grade']:>5} {o['top1'][:18]:>18} "
                  f"{'WIN' if o['hit'] else '.':>7} {n['top1'][:18]:>18} "
                  f"{'WIN' if n['hit'] else '.':>7} {o['winner'][:18]:>18}")


if __name__ == "__main__":
    main()
