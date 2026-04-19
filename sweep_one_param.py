"""Single-parameter sweep for candidate generation.

Evaluates +10%, +20%, -10%, -20% variants of each coefficient.
Picks the single best change that improves Brier score (calibration).

Uses mocked test data with structured features to simulate evaluation.
Does NOT modify production files.

Run: python sweep_one_param.py
"""

import sys
import copy
import importlib

# ── Mock data setup (identical to test_pipeline.py) ──
import data_store

_TEST_PREDICTIONS = {}
_TEST_RESULTS = {}

data_store.load_predictions = lambda: _TEST_PREDICTIONS
data_store.load_results = lambda: _TEST_RESULTS

import prepare
importlib.reload(prepare)
import evaluator
importlib.reload(evaluator)
import feature_store


def _make_entry(i, name, odds, jockey_wr="", training_acc=0.0, hw_delta=0,
                paddock_v=0.4, paddock_h=0.6, paddock_g=0.5, cardio=0.8):
    return {
        "number": str(i + 1), "waku": str((i % 8) + 1),
        "name": name, "horse_id": f"horse_{i:04d}",
        "jockey": f"Jockey_{i}", "jockey_id": f"jky_{i:04d}",
        "trainer": f"Trainer_{i}", "trainer_id": f"tr_{i:04d}",
        "age": f"牡{3 + (i % 4)}", "weight": f"{54 + (i % 4)}",
        "horse_weight": f"{470 + i * 2}({hw_delta:+d})",
        "odds": f"{odds:.1f}倍", "stable": "栗東", "ritto": "NF天栄",
        "transport_stress": "低", "recent_form": "1着-3着-2着",
        "bloodline": "父:ディープ 母:テスト",
        "weight_trend": f"{470 + i * 2}kg({hw_delta:+d})",
        "jockey_win_rate": jockey_wr, "jockey_g1_wins": "2勝",
        "trainer_win_rate": "12%", "training_eval": "CW良 12.5-11.8-11.2",
        "training_physics": {"final_split": 11.2, "acceleration_rate": training_acc, "cardio_index": cardio},
        "training_nlp": {"coat_gloss": 0.5, "stride_quality": 0.3, "weight_status": 0.1},
        "paddock_scores": {"vascularity_index": paddock_v, "hindquarter_power": paddock_h, "gait_fluidity": paddock_g},
        "best_weight_analysis": {"diff_from_best": 2.0, "record_count": 5},
        "transport_profile": {},
    }


def generate_data(n=50):
    global _TEST_PREDICTIONS, _TEST_RESULTS
    _TEST_PREDICTIONS = {}
    _TEST_RESULTS = {}
    for i in range(n):
        race_id = f"sweep_race_{i:04d}"
        year = 2024 + (i % 3)
        month = (i % 12) + 1
        ts = f"{year}-{month:02d}-{(i % 28) + 1:02d}T12:00:00"
        top_wins = (i % 5 == 0)

        horses, entries = [], []
        for j in range(3):
            odds_val = 2.0 + j * 3.5 + (i % 7) * 0.5
            name = f"Horse_{chr(65+j)}_{i}"
            horses.append({
                "rank": j + 1, "name": name,
                "confidence": 0, "ev_gap": "0", "odds": f"{odds_val:.1f}倍", "bet": "",
            })
            entries.append(_make_entry(
                i * 3 + j, name, odds_val,
                jockey_wr=f"{8 + j * 6 + (i % 5)}%",
                training_acc=0.08 - j * 0.05 + (i % 3) * 0.02,
                hw_delta=2 - j * 4 + (i % 4),
                paddock_v=0.3 + (i % 5) * 0.1 - j * 0.2,
                paddock_h=0.4 + (i % 4) * 0.1 - j * 0.15,
                paddock_g=0.3 + (i % 3) * 0.15 - j * 0.1,
                cardio=0.6 + (i % 5) * 0.08,
            ))

        _TEST_PREDICTIONS[race_id] = {
            "race_name": f"Sweep Race {i}", "grade": ["G1", "G2", "G3"][i % 3],
            "horses": horses, "timestamp": ts,
            "structured_features": feature_store.extract_structured_features(
                entries=entries,
                track_condition=["良", "良", "稍重", "重", "不良"][i % 5],
                weather="晴", venue=["東京", "中山", "阪神", "京都"][i % 4],
            ),
        }

        winner = f"Horse_A_{i}" if top_wins else f"Horse_B_{i}"
        _TEST_RESULTS[race_id] = {
            "race_name": f"Sweep Race {i}",
            "finishing_order": [
                {"rank": 1, "name": winner, "odds": horses[0 if top_wins else 1]["odds"]},
                {"rank": 2, "name": f"Horse_{'B' if top_wins else 'C'}_{i}"},
                {"rank": 3, "name": f"Horse_{'C' if top_wins else 'A'}_{i}"},
            ],
            "payouts": {"win": 500 if top_wins else 0}, "timestamp": ts,
        }


def make_score_fn(param_name, param_value):
    """Create a score_runner variant with ONE parameter changed."""

    def variant_score_runner(features, context):
        horses = features.get("horse_features", [])
        if not horses:
            return {"top_confidence": 50}

        num_horses = features.get("num_horses", 0) or len(horses)
        top = horses[0]
        top_name = top.get("name", "")
        sf = features.get("structured_features")

        if sf and isinstance(sf, dict):
            h = sf.get("horses", {}).get(top_name, {})
            race = sf.get("race", {})
        else:
            h = {}
            race = {}

        odds = h.get("odds", 0.0) or top.get("odds", 0.0)
        if odds > 1.0:
            base = _clamp(1.0 / odds / 1.20, 0.02, 0.80)
        else:
            base = _clamp(1.0 / max(num_horses, 1), 0.02, 0.80)

        n_field = _clamp((18 - num_horses) / 10.0, 0.0, 1.0)
        grade = features.get("grade", "")
        n_grade = 1.0 if grade == "G1" else (0.5 if grade == "G2" else 0.0)
        n_jockey = _clamp(h.get("jockey_win_rate", 0.0) / 0.25, 0.0, 1.0)
        n_training = _clamp((h.get("training_acceleration", 0.0) + 0.15) / 0.30, 0.0, 1.0)
        excess = _clamp(abs(h.get("horse_weight_delta", 0.0)) - 4.0, 0.0, 16.0)
        n_weight_penalty = (excess / 16.0) ** 2
        bio_raw = (h.get("paddock_vascularity", 0.0) + h.get("paddock_hindquarter", 0.0) + h.get("paddock_gait", 0.0)) / 3.0
        n_bio = _clamp((bio_raw + 1.0) / 2.0, 0.0, 1.0)
        other_odds = [x.get("odds", 0) for x in horses[1:] if x.get("odds", 0) > 1.0]
        if odds > 1.0 and other_odds:
            n_consensus = _clamp(1.0 - odds / (sum(other_odds) / len(other_odds)), 0.0, 1.0)
        else:
            n_consensus = 0.5

        ix_jockey_grade = n_jockey * n_grade
        ix_training_bio = n_training * n_bio

        track = race.get("track_condition", "")
        n_cardio = _clamp(h.get("training_cardio_index", 0.0), 0.0, 1.0)
        cond_track = (0.04 if track in ("重", "不良") else 0.01) * n_cardio

        # Default coefficients
        params = {
            "W_FIELD": 0.06, "W_GRADE": 0.02, "W_JOCKEY": 0.05,
            "W_TRAINING": 0.03, "W_WEIGHT_P": 0.04, "W_BIO": 0.03,
            "W_CONSENSUS": 0.04, "W_IX_JG": 0.03, "W_IX_TB": 0.02,
        }
        # Apply the ONE change
        params[param_name] = param_value

        adjustment = (
            params["W_FIELD"] * n_field
            + params["W_GRADE"] * n_grade
            + params["W_JOCKEY"] * n_jockey
            + params["W_TRAINING"] * n_training
            - params["W_WEIGHT_P"] * n_weight_penalty
            + params["W_BIO"] * n_bio
            + params["W_CONSENSUS"] * (n_consensus - 0.5)
            + params["W_IX_JG"] * ix_jockey_grade
            + params["W_IX_TB"] * ix_training_bio
            + cond_track
        )
        final_prob = _clamp(base + adjustment, 0.02, 0.95)
        return {"top_confidence": final_prob * 100.0}

    return variant_score_runner


def _clamp(val, lo, hi):
    if val < lo: return lo
    if val > hi: return hi
    return val


def main():
    print("=" * 60)
    print("SINGLE-PARAMETER SWEEP")
    print("=" * 60)

    generate_data(50)

    # Baseline
    from train import score_runner as baseline_fn
    baseline = evaluator.evaluate_walk_forward(baseline_fn)
    print(f"\nBaseline: ROI={baseline['roi']:.4f}  Brier={baseline['brier']:.4f}  "
          f"MDD={baseline['max_drawdown']:.4f}  Races={baseline['num_races']}")

    # Parameters and their current values
    params = {
        "W_FIELD": 0.06, "W_GRADE": 0.02, "W_JOCKEY": 0.05,
        "W_TRAINING": 0.03, "W_WEIGHT_P": 0.04, "W_BIO": 0.03,
        "W_CONSENSUS": 0.04, "W_IX_JG": 0.03, "W_IX_TB": 0.02,
    }
    deltas = [0.90, 0.80, 1.10, 1.20]  # -10%, -20%, +10%, +20%

    print(f"\nSweeping {len(params)} params × {len(deltas)} variants = {len(params)*len(deltas)} evaluations\n")

    results = []
    for pname, pval in params.items():
        for mult in deltas:
            new_val = round(pval * mult, 4)
            label = f"{pname}: {pval} → {new_val} ({mult:.0%})"
            fn = make_score_fn(pname, new_val)
            r = evaluator.evaluate_walk_forward(fn)
            if r.get("error"):
                print(f"  ERROR: {label}")
                continue

            # Compare to baseline
            brier_delta = r["brier"] - baseline["brier"]
            roi_delta = r["roi"] - baseline["roi"]
            improved = (r["brier"] <= baseline["brier"] and r["roi"] >= baseline["roi"])

            results.append({
                "param": pname, "old": pval, "new": new_val, "mult": mult,
                "roi": r["roi"], "brier": r["brier"], "mdd": r["max_drawdown"],
                "brier_delta": brier_delta, "roi_delta": roi_delta,
                "improved": improved, "label": label,
                "fold_results": r.get("fold_results", []),
                "yearly_results": r.get("yearly_results", {}),
            })

            marker = " <<<" if improved else ""
            print(f"  {label:45s}  Brier={r['brier']:.4f} ({brier_delta:+.4f})  "
                  f"ROI={r['roi']:.4f} ({roi_delta:+.4f}){marker}")

    # Find best candidate
    improved = [r for r in results if r["improved"]]
    print(f"\n{'='*60}")
    if not improved:
        print("No variant improved both ROI and Brier. DISCARD all.")
        return

    # Sort by Brier improvement (primary), then ROI improvement (secondary)
    improved.sort(key=lambda r: (r["brier_delta"], -r["roi_delta"]))
    best = improved[0]

    print(f"BEST CANDIDATE: {best['label']}")
    print(f"  Brier: {baseline['brier']:.4f} → {best['brier']:.4f} ({best['brier_delta']:+.4f})")
    print(f"  ROI:   {baseline['roi']:.4f} → {best['roi']:.4f} ({best['roi_delta']:+.4f})")
    print(f"  MDD:   {baseline['max_drawdown']:.4f} → {best['mdd']:.4f}")

    # Show fold consistency
    b_folds = baseline.get("fold_results", [])
    c_folds = best["fold_results"]
    improved_folds = 0
    total_folds = min(len(b_folds), len(c_folds))
    for i in range(total_folds):
        if c_folds[i]["brier"] <= b_folds[i]["brier"]:
            improved_folds += 1
    print(f"  Fold consistency: {improved_folds}/{total_folds} folds improved")

    print(f"\nApply this change to train.py: {best['param']} = {best['new']}")


if __name__ == "__main__":
    main()
