"""Diagnostic: decompose score_runner's output into (base, adjustment, parts)
for every horse in the 121-race cached dataset. Answers the question:
'what fraction of score_runner's output is the odds base vs everything
else, and how does that ratio look in typical vs top-contending horses?'
"""

import json
import sys
import statistics as st

sys.stdout.reconfigure(encoding="utf-8")


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def score_runner_decomposed(horse: dict, num_horses: int, grade: str, race: dict) -> dict:
    """Mirror of train.score_runner but returns per-term contributions
    instead of the final clamped probability."""
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
        raw_prob = 1.0 / odds
        base = _clamp(raw_prob / 1.20, 0.02, 0.80)
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

    # consensus = placeholder (needs full field, simplified here)
    n_consensus = 0.5

    ix_jockey_grade = n_jockey * n_grade
    ix_training_bio = n_training * n_bio

    track = race.get("track_condition", "")
    cardio = float(horse.get("training_cardio_index", 0) or 0)
    n_cardio = _clamp(cardio, 0.0, 1.0)
    cond_track = 0.04 * n_cardio if track in ("重", "不良") else 0.01 * n_cardio

    parts = {
        "base":         base,
        "field":        W_FIELD * n_field,
        "grade":        W_GRADE * n_grade,
        "jockey":       W_JOCKEY * n_jockey,
        "training":     W_TRAINING * n_training,
        "weight_pen":  -W_WEIGHT_P * n_weight_penalty,
        "bio":          W_BIO * n_bio,
        "consensus":    W_CONSENSUS * (n_consensus - 0.5),
        "ix_jg":        W_IX_JG * ix_jockey_grade,
        "ix_tb":        W_IX_TB * ix_training_bio,
        "cond_track":   cond_track,
    }
    adjustment = sum(v for k, v in parts.items() if k != "base")
    final = _clamp(base + adjustment, 0.02, 0.95)
    parts["adjustment_total"] = adjustment
    parts["final_prob"] = final
    parts["final_score"] = final * 100
    return parts


def main():
    data = json.load(open("data/enriched_backtest_results_v3.json", "r", encoding="utf-8"))

    all_horses = []
    for row in data:
        sf_horses = row["structured_features"]["horses"]
        race = row["structured_features"].get("race", {})
        grade = row.get("grade", "")
        n = len(sf_horses)
        for name, h in sf_horses.items():
            parts = score_runner_decomposed(h, n, grade, race)
            parts["_race_id"] = row["race_id"]
            parts["_name"] = name
            parts["_odds"] = h.get("odds", 0)
            all_horses.append(parts)

    print(f"=== score_runner contribution breakdown over {len(all_horses)} horses ===")
    print()

    # Aggregate stats
    bases = [h["base"] for h in all_horses]
    adjs = [h["adjustment_total"] for h in all_horses]
    finals = [h["final_prob"] for h in all_horses]

    print(f"base         : mean={st.mean(bases):.4f} median={st.median(bases):.4f} "
          f"min={min(bases):.4f} max={max(bases):.4f}")
    print(f"adjustment   : mean={st.mean(adjs):.4f} median={st.median(adjs):.4f} "
          f"min={min(adjs):.4f} max={max(adjs):.4f}")
    print(f"final_prob   : mean={st.mean(finals):.4f} median={st.median(finals):.4f}")
    print()

    # Ratio: adjustment / (base + adjustment) — the "structured contribution %"
    ratios = []
    for h in all_horses:
        b = h["base"]
        a = h["adjustment_total"]
        total = b + abs(a)  # use abs because penalty is negative
        if total > 0:
            ratios.append(abs(a) / total * 100)
    print(f"structured contribution % of total (|adj| / (base + |adj|)):")
    print(f"  mean={st.mean(ratios):.2f}%  median={st.median(ratios):.2f}%")
    print(f"  min={min(ratios):.2f}%  max={max(ratios):.2f}%")
    print()

    # Distribution by odds bucket
    print("=== by odds bucket ===")
    print(f"{'bucket':12} {'n':>4} {'mean_base':>10} {'mean_adj':>10} {'mean_ratio%':>12} {'med_adj':>10}")
    buckets = [
        ("<2.0",   lambda o: 0 < o < 2.0),
        ("2.0-3.5", lambda o: 2.0 <= o < 3.5),
        ("3.5-6",  lambda o: 3.5 <= o < 6.0),
        ("6-12",   lambda o: 6.0 <= o < 12.0),
        ("12-30",  lambda o: 12.0 <= o < 30.0),
        ("30+",    lambda o: o >= 30.0),
    ]
    for label, pred in buckets:
        sub = [h for h in all_horses if pred(h["_odds"])]
        if not sub:
            continue
        mb = st.mean(h["base"] for h in sub)
        ma = st.mean(h["adjustment_total"] for h in sub)
        mr = st.mean(abs(h["adjustment_total"]) / (h["base"] + abs(h["adjustment_total"])) * 100
                     for h in sub if (h["base"] + abs(h["adjustment_total"])) > 0)
        med_a = st.median(h["adjustment_total"] for h in sub)
        print(f"{label:12} {len(sub):>4} {mb:>10.4f} {ma:>10.4f} {mr:>11.2f}% {med_a:>10.4f}")
    print()

    # Gap analysis: for each race, how does the score gap between top-1 and top-2
    # compare to the max possible structured swing?
    print("=== top-1 vs top-2 score gap analysis (121 races) ===")
    gaps = []
    gap_ratios = []
    for row in data:
        sf_horses = row["structured_features"]["horses"]
        race = row["structured_features"].get("race", {})
        grade = row.get("grade", "")
        n = len(sf_horses)
        scored = []
        for name, h in sf_horses.items():
            parts = score_runner_decomposed(h, n, grade, race)
            scored.append((name, parts["final_prob"], parts["adjustment_total"]))
        scored.sort(key=lambda x: -x[1])
        if len(scored) >= 2:
            top1_score = scored[0][1]
            top2_score = scored[1][1]
            gap = top1_score - top2_score
            gaps.append(gap)
            top1_adj = scored[0][2]
            top2_adj = scored[1][2]
            max_adj_flip = abs(top2_adj) + abs(top1_adj)  # rough measure
            if gap > 0:
                gap_ratios.append(max_adj_flip / gap)

    print(f"top1-top2 score gap: mean={st.mean(gaps):.4f} median={st.median(gaps):.4f} max={max(gaps):.4f}")
    print(f"  (if median gap is 0.10, top-2 needs a 0.10 adj swing to flip — current max adj = ~0.24)")
    print()
    small_gap_races = sum(1 for g in gaps if g < 0.05)
    print(f"races with small gap (<0.05): {small_gap_races} / {len(gaps)} ({small_gap_races/len(gaps)*100:.0f}%)")
    print("  these are the only races where a +20% coefficient change CAN flip the top")
    print()

    # Absolute contribution of each structured term at the horse with max of each
    print("=== max contribution of each structured term (ranked) ===")
    contribs = {}
    for key in ["field", "grade", "jockey", "training", "weight_pen", "bio",
                "consensus", "ix_jg", "ix_tb", "cond_track"]:
        vals = [abs(h[key]) for h in all_horses]
        contribs[key] = (st.mean(vals), max(vals))
    for key, (mean, mx) in sorted(contribs.items(), key=lambda x: -x[1][0]):
        print(f"  {key:12} mean={mean:.4f}  max={mx:.4f}")


if __name__ == "__main__":
    main()
