"""Evaluation: baseline vs scale-rebalanced score_runner + probability layer.

Measures:
  - Top1 rank changes (different horse becomes #1)
  - Top3 membership changes (any horse enters/leaves the top-3 set)
  - Calibrated win_prob distribution shift (top1 mean, entropy)
  - Loose-rule trigger count change

Sweep grid: STRUCTURED_GAIN in {1.0, 1.5, 2.0, 2.5} × CALIBRATION_K in {1.5, 3.0, 4.0}.
Baseline is (gain=1.0, k=1.5) — the current committed state.
"""

import json
import math
import sys
import statistics as st

sys.stdout.reconfigure(encoding="utf-8")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def score_with_gain(horse, num_horses, grade, race, gain):
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
    n_consensus = 0.5  # placeholder; consensus depends on field

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
    structured_edge = final - base  # gain-scaled adjustment (clamped)
    return final, structured_edge, base


def calibrated_probs(running, k):
    """running: list[(name, odds, structured_edge)] with odds>1."""
    implied = [1.0 / o for _, o, _ in running]
    tot = sum(implied) or 1.0
    base_probs = [i / tot for i in implied]
    mults = [math.exp(k * e) for _, _, e in running]
    adj = [b * m for b, m in zip(base_probs, mults)]
    tot_adj = sum(adj) or 1.0
    return [a / tot_adj for a in adj]


def rank_race(row, gain, k):
    sf = row["structured_features"]
    horses = sf["horses"]
    race = sf.get("race", {})
    grade = row.get("grade", "")
    n = len(horses)
    scored = []
    for name, h in horses.items():
        odds = h.get("odds", 0) or 0
        final, edge, base = score_with_gain(h, n, grade, race, gain)
        scored.append({
            "name": name,
            "odds": odds,
            "final_score": final * 100,
            "structured_edge": edge,
            "base": base,
        })
    # calibrated prob layer
    running = [(r["name"], r["odds"], r["structured_edge"]) for r in scored if r["odds"] > 1.0]
    if running:
        probs = calibrated_probs(running, k)
        prob_map = {name: p for (name, _, _), p in zip(running, probs)}
    else:
        prob_map = {}
    for r in scored:
        r["win_prob"] = prob_map.get(r["name"], 0.0)
    # rank by win_prob then by final_score
    scored.sort(key=lambda r: (-r["win_prob"], -r["final_score"]))
    return scored


def entropy(probs):
    s = 0.0
    for p in probs:
        if p > 0:
            s -= p * math.log(p)
    return s


def eval_combo(data, gain, k, baseline_rankings):
    top1_changes = 0
    top3_changes = 0
    top1_probs = []
    entropies = []
    for i, row in enumerate(data):
        ranked = rank_race(row, gain, k)
        if not ranked:
            continue
        base_ranked = baseline_rankings[i]
        if base_ranked[0]["name"] != ranked[0]["name"]:
            top1_changes += 1
        base_top3 = {r["name"] for r in base_ranked[:3]}
        new_top3 = {r["name"] for r in ranked[:3]}
        if base_top3 != new_top3:
            top3_changes += 1
        top1_probs.append(ranked[0]["win_prob"])
        probs = [r["win_prob"] for r in ranked if r["win_prob"] > 0]
        entropies.append(entropy(probs))
    return {
        "gain": gain,
        "k": k,
        "top1_changes": top1_changes,
        "top3_changes": top3_changes,
        "top1_mean": st.mean(top1_probs),
        "top1_median": st.median(top1_probs),
        "entropy_mean": st.mean(entropies),
    }


def main():
    data = json.load(open("data/enriched_backtest_results_v3.json", "r", encoding="utf-8"))
    print(f"Loaded {len(data)} races.\n")

    # baseline: current committed state (gain=1.0 equivalent, k=1.5)
    baseline_rankings = [rank_race(row, 1.0, 1.5) for row in data]

    # baseline stats
    b_top1 = [r[0]["win_prob"] for r in baseline_rankings if r]
    b_ent = [entropy([h["win_prob"] for h in r if h["win_prob"] > 0]) for r in baseline_rankings if r]
    print("=== BASELINE (gain=1.0, k=1.5) ===")
    print(f"  top1 win_prob mean={st.mean(b_top1):.3f}  median={st.median(b_top1):.3f}")
    print(f"  entropy mean={st.mean(b_ent):.3f}")
    print()

    print("=== SWEEP ===")
    print(f"{'gain':>5} {'k':>5} {'Δtop1':>7} {'Δtop3':>7} {'top1_mean':>10} {'top1_med':>10} {'entropy':>9}")
    combos = [
        (1.0, 1.5),  # baseline (sanity: 0 changes)
        (1.5, 1.5),
        (1.5, 3.0),
        (1.5, 4.0),
        (2.0, 1.5),
        (2.0, 3.0),
        (2.0, 4.0),
        (2.5, 3.0),
        (2.5, 4.0),
    ]
    for gain, k in combos:
        r = eval_combo(data, gain, k, baseline_rankings)
        print(f"{r['gain']:>5.1f} {r['k']:>5.1f} {r['top1_changes']:>7d} {r['top3_changes']:>7d} "
              f"{r['top1_mean']:>10.3f} {r['top1_median']:>10.3f} {r['entropy_mean']:>9.3f}")


if __name__ == "__main__":
    main()
