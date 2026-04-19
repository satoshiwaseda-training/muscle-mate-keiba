"""Pipeline verification tests — expressive structured-feature formula.

Verifies:
1. Gemini fields NOT used
2. Structured features consumed correctly
3. Legacy records still work
4. Normalization: all signals comparable
5. Nonlinear: large weight delta penalized more than small
6. Interactions: jockey*grade changes ranking
7. Conditional: heavy track boosts cardio contribution
8. Bio composite: paddock scores affect output
9. Determinism, leakage, gates

Run: python test_pipeline.py
"""

import json
import sys
import inspect

import data_store

_TEST_PREDICTIONS = {}
_TEST_RESULTS = {}


def _mock_load_predictions():
    return _TEST_PREDICTIONS


def _mock_load_results():
    return _TEST_RESULTS


data_store.load_predictions = _mock_load_predictions
data_store.load_results = _mock_load_results

import importlib
import prepare
importlib.reload(prepare)
import evaluator
importlib.reload(evaluator)
import feature_store


def _make_entry(i, name, odds, jockey_wr="", training_acc=0.0, hw_delta=0,
                paddock_v=0.4, paddock_h=0.6, paddock_g=0.5, cardio=0.8):
    """Create a mock enriched entry as the scraper would produce."""
    return {
        "number": str(i + 1),
        "waku": str((i % 8) + 1),
        "name": name,
        "horse_id": f"horse_{i:04d}",
        "jockey": f"Jockey_{i}", "jockey_id": f"jky_{i:04d}",
        "trainer": f"Trainer_{i}", "trainer_id": f"tr_{i:04d}",
        "age": f"牡{3 + (i % 4)}",
        "weight": f"{54 + (i % 4)}",
        "horse_weight": f"{470 + i * 2}({hw_delta:+d})",
        "odds": f"{odds:.1f}倍",
        "stable": "栗東", "ritto": "NF天栄", "transport_stress": "低",
        "recent_form": "1着-3着-2着",
        "bloodline": "父:ディープ 母:テスト",
        "weight_trend": f"{470 + i * 2}kg({hw_delta:+d})",
        "jockey_win_rate": jockey_wr, "jockey_g1_wins": "2勝",
        "trainer_win_rate": "12%",
        "training_eval": "CW良 12.5-11.8-11.2 一杯",
        "training_physics": {
            "final_split": 11.2,
            "acceleration_rate": training_acc,
            "cardio_index": cardio,
        },
        "training_nlp": {"coat_gloss": 0.5, "stride_quality": 0.3, "weight_status": 0.1},
        "paddock_scores": {
            "vascularity_index": paddock_v,
            "hindquarter_power": paddock_h,
            "gait_fluidity": paddock_g,
        },
        "best_weight_analysis": {"diff_from_best": 2.0, "record_count": 5},
        "transport_profile": {},
    }


def _make_features(name, odds, sf, grade="G2", num_horses=12, other_odds=8.0):
    """Build a complete features dict for score_runner."""
    return {
        "grade": grade, "num_horses": num_horses,
        "horse_features": [
            {"name": name, "rank": 1, "odds": odds, "confidence": 0, "ev_gap": 0, "bet": ""},
            {"name": "Other", "rank": 2, "odds": other_odds, "confidence": 0, "ev_gap": 0, "bet": ""},
        ],
        "structured_features": sf,
    }


def _score(features, context=None):
    from train import score_runner
    ctx = context or {"weights": data_store.DEFAULT_WEIGHTS.copy()}
    return score_runner(features, ctx)["top_confidence"]


def _generate_test_data(n_races=20, with_structured=False):
    global _TEST_PREDICTIONS, _TEST_RESULTS
    _TEST_PREDICTIONS = {}
    _TEST_RESULTS = {}

    for i in range(n_races):
        race_id = f"test_race_{i:04d}"
        year = 2024 + (i % 3)
        month = (i % 12) + 1
        ts = f"{year}-{month:02d}-15T12:00:00"
        top_wins = (i % 5 == 0)

        horses, entries = [], []
        for j in range(3):
            odds_val = 2.0 + j * 3.0 + (i % 7) * 0.5
            name = f"Horse_{chr(65+j)}_{i}"
            horses.append({
                "rank": j + 1, "name": name,
                "confidence": 70 - j * 10, "ev_gap": f"+{5 - j * 3}",
                "odds": f"{odds_val:.1f}倍", "bet": "推奨" if j == 0 else "",
            })
            entries.append(_make_entry(
                i * 3 + j, name, odds_val,
                jockey_wr=f"{10 + j * 5}%", training_acc=0.06 - j * 0.04, hw_delta=2 - j * 3,
            ))

        pred = {
            "race_name": f"Test Race {i}",
            "grade": ["G1", "G2", "G3"][i % 3],
            "horses": horses, "timestamp": ts,
        }
        if with_structured:
            pred["structured_features"] = feature_store.extract_structured_features(
                entries=entries, track_condition="良", weather="晴", venue="東京",
            )
        _TEST_PREDICTIONS[race_id] = pred

        winner = f"Horse_A_{i}" if top_wins else f"Horse_B_{i}"
        _TEST_RESULTS[race_id] = {
            "race_name": f"Test Race {i}",
            "finishing_order": [
                {"rank": 1, "name": winner, "odds": horses[0 if top_wins else 1]["odds"]},
                {"rank": 2, "name": f"Horse_{'B' if top_wins else 'C'}_{i}"},
                {"rank": 3, "name": f"Horse_{'C' if top_wins else 'A'}_{i}"},
            ],
            "payouts": {"win": 500 if top_wins else 0}, "timestamp": ts,
        }


# ══════════════════════════════════════════════════════════
# Core constraint tests
# ══════════════════════════════════════════════════════════

def test_gemini_independence():
    """CRITICAL: score_runner must NOT use Gemini-derived fields."""
    print("Test: Gemini independence...", end=" ")
    import train
    source = inspect.getsource(train.score_runner)
    violations = []
    for line in source.split("\n"):
        s = line.strip()
        if s.startswith("#") or s.startswith('"""') or not s:
            continue
        code = s.split("#")[0]
        for f in ("confidence", "ev_gap", "bet"):
            if f'.get("{f}"' in code or f'["{f}"]' in code:
                violations.append(f"Uses '{f}': {s}")
    assert not violations, "Gemini dependency:\n" + "\n".join(violations)
    print("PASS")


def test_gemini_fields_ignored():
    """Changing Gemini fields must NOT change the score."""
    print("Test: Gemini fields ignored...", end=" ")
    base = {
        "grade": "G2", "num_horses": 14,
        "horse_features": [
            {"name": "A", "rank": 1, "odds": 5.0, "confidence": 50, "ev_gap": 0, "bet": ""},
            {"name": "B", "rank": 2, "odds": 10.0, "confidence": 50, "ev_gap": 0, "bet": ""},
        ],
    }
    s1 = _score(base)
    mod = json.loads(json.dumps(base))
    mod["horse_features"][0]["confidence"] = 99
    mod["horse_features"][0]["ev_gap"] = 50
    mod["horse_features"][0]["bet"] = "推奨"
    s2 = _score(mod)
    assert s1 == s2, f"Score changed: {s1} vs {s2}"
    print(f"PASS (both={s1:.2f})")


def test_legacy_fallback():
    """score_runner works without structured_features."""
    print("Test: Legacy fallback...", end=" ")
    f = {
        "grade": "G1", "num_horses": 10,
        "horse_features": [
            {"name": "A", "rank": 1, "odds": 3.5, "confidence": 0, "ev_gap": 0, "bet": ""},
            {"name": "B", "rank": 2, "odds": 8.0, "confidence": 0, "ev_gap": 0, "bet": ""},
        ],
    }
    s = _score(f)
    assert 0 < s < 100
    print(f"PASS (score={s:.2f})")


def test_odds_masked_fallback():
    """Valid output when odds are masked."""
    print("Test: Odds masked...", end=" ")
    f = {
        "grade": "G1", "num_horses": 10,
        "horse_features": [{"name": "A", "rank": 1, "odds": 0.0, "confidence": 0, "ev_gap": 0, "bet": ""}],
    }
    s = _score(f)
    assert 0 < s < 100
    print(f"PASS (score={s:.2f})")


def test_score_varies_with_odds():
    """Favorites score higher than longshots."""
    print("Test: Odds vary score...", end=" ")
    def mk(o):
        return {"grade": "G2", "num_horses": 12, "horse_features": [
            {"name": "A", "rank": 1, "odds": o, "confidence": 0, "ev_gap": 0, "bet": ""},
            {"name": "B", "rank": 2, "odds": 15.0, "confidence": 0, "ev_gap": 0, "bet": ""},
        ]}
    s_fav = _score(mk(2.0))
    s_long = _score(mk(30.0))
    assert s_fav > s_long
    print(f"PASS (fav={s_fav:.2f}, long={s_long:.2f})")


# ══════════════════════════════════════════════════════════
# Expressiveness tests
# ══════════════════════════════════════════════════════════

def test_nonlinear_weight_penalty():
    """Large weight delta penalized MORE than small delta (nonlinear)."""
    print("Test: Nonlinear weight penalty...", end=" ")
    def mk(delta):
        e = [_make_entry(0, "H", 4.0, jockey_wr="12%", hw_delta=delta),
             _make_entry(1, "O", 8.0)]
        sf = feature_store.extract_structured_features(entries=e)
        return _make_features("H", 4.0, sf)

    score_stable = _score(mk(0))       # 0 kg delta
    score_small = _score(mk(6))        # 6 kg delta (mild)
    score_large = _score(mk(18))       # 18 kg delta (extreme)

    # Penalty should increase: stable > small > large
    assert score_stable > score_small, f"stable ({score_stable:.2f}) should > small ({score_small:.2f})"
    assert score_small > score_large, f"small ({score_small:.2f}) should > large ({score_large:.2f})"

    # Nonlinear: gap between small→large should be bigger than stable→small
    gap_mild = score_stable - score_small
    gap_severe = score_small - score_large
    assert gap_severe > gap_mild, (
        f"Nonlinear expected: severe gap ({gap_severe:.4f}) > mild gap ({gap_mild:.4f})"
    )
    print(f"PASS (stable={score_stable:.2f}, small={score_small:.2f}, large={score_large:.2f})")


def test_interaction_jockey_grade():
    """Elite jockey contributes MORE in G1 than in G3."""
    print("Test: Interaction jockey*grade...", end=" ")
    def mk(grade, jockey_wr):
        e = [_make_entry(0, "H", 4.0, jockey_wr=jockey_wr),
             _make_entry(1, "O", 8.0)]
        sf = feature_store.extract_structured_features(entries=e)
        return _make_features("H", 4.0, sf, grade=grade)

    # Elite jockey (20%) in G1 vs G3
    elite_g1 = _score(mk("G1", "20%"))
    elite_g3 = _score(mk("G3", "20%"))

    # Average jockey (8%) in G1 vs G3
    avg_g1 = _score(mk("G1", "8%"))
    avg_g3 = _score(mk("G3", "8%"))

    # The SPREAD between elite and average should be larger in G1
    spread_g1 = elite_g1 - avg_g1
    spread_g3 = elite_g3 - avg_g3
    assert spread_g1 > spread_g3, (
        f"Jockey spread in G1 ({spread_g1:.4f}) should > G3 ({spread_g3:.4f})"
    )
    print(f"PASS (spread G1={spread_g1:.4f}, G3={spread_g3:.4f})")


def test_interaction_training_bio():
    """Training*bio interaction term changes score beyond pure additive."""
    print("Test: Interaction training*bio...", end=" ")
    def mk(training_acc, pad_v, pad_h, pad_g):
        e = [_make_entry(0, "H", 4.0, jockey_wr="12%", training_acc=training_acc,
                         paddock_v=pad_v, paddock_h=pad_h, paddock_g=pad_g),
             _make_entry(1, "O", 8.0)]
        sf = feature_store.extract_structured_features(entries=e)
        return _make_features("H", 4.0, sf)

    # Both good → interaction term is high (n_training * n_bio both high)
    both_good = _score(mk(0.12, 0.8, 0.9, 0.7))
    # Both bad → interaction term is low (n_training * n_bio both low)
    both_bad = _score(mk(-0.10, -0.5, -0.5, -0.5))
    # Mixed: good training, bad paddock → interaction term is low
    mixed = _score(mk(0.12, -0.5, -0.5, -0.5))

    # Both good should beat mixed (because interaction boosts when BOTH are good)
    assert both_good > mixed, f"both_good ({both_good:.2f}) should > mixed ({mixed:.2f})"
    # Mixed should still beat both_bad (training additive term still helps)
    assert mixed > both_bad, f"mixed ({mixed:.2f}) should > both_bad ({both_bad:.2f})"
    # The gap from mixed→both_good should be meaningful (interaction effect)
    interaction_boost = both_good - mixed
    assert interaction_boost > 0.5, f"Interaction boost too small: {interaction_boost:.4f}"
    print(f"PASS (good={both_good:.2f}, mixed={mixed:.2f}, bad={both_bad:.2f}, ix_boost={interaction_boost:.2f})")


def test_conditional_heavy_track():
    """Heavy track should make cardio index more impactful."""
    print("Test: Conditional heavy track...", end=" ")
    def mk(track, cardio):
        e = [_make_entry(0, "H", 4.0, jockey_wr="12%", cardio=cardio),
             _make_entry(1, "O", 8.0)]
        sf = feature_store.extract_structured_features(
            entries=e, track_condition=track,
        )
        return _make_features("H", 4.0, sf)

    # High cardio on heavy vs good track
    heavy_high = _score(mk("重", 0.9))
    heavy_low = _score(mk("重", 0.1))
    good_high = _score(mk("良", 0.9))
    good_low = _score(mk("良", 0.1))

    # Cardio spread should be larger on heavy track
    spread_heavy = heavy_high - heavy_low
    spread_good = good_high - good_low
    assert spread_heavy > spread_good, (
        f"Cardio spread on heavy ({spread_heavy:.4f}) should > good ({spread_good:.4f})"
    )
    print(f"PASS (heavy spread={spread_heavy:.4f}, good spread={spread_good:.4f})")


def test_bio_composite_affects_score():
    """Paddock bio composite (vascularity + hindquarter + gait) changes score."""
    print("Test: Bio composite...", end=" ")
    def mk(v, h, g):
        e = [_make_entry(0, "H", 4.0, jockey_wr="12%", paddock_v=v, paddock_h=h, paddock_g=g),
             _make_entry(1, "O", 8.0)]
        sf = feature_store.extract_structured_features(entries=e)
        return _make_features("H", 4.0, sf)

    excellent = _score(mk(0.9, 0.9, 0.9))
    poor = _score(mk(-0.5, -0.5, -0.5))
    assert excellent > poor, f"Excellent bio ({excellent:.2f}) should > poor ({poor:.2f})"
    print(f"PASS (excellent={excellent:.2f}, poor={poor:.2f})")


def test_structured_features_differentiate():
    """Strong vs weak horse with SAME odds get different scores."""
    print("Test: Differentiation (same odds)...", end=" ")
    strong_e = [_make_entry(0, "Strong", 4.0, jockey_wr="22%", training_acc=0.12,
                            hw_delta=2, paddock_v=0.8, paddock_h=0.9, paddock_g=0.7),
                _make_entry(1, "Other", 8.0)]
    weak_e = [_make_entry(0, "Weak", 4.0, jockey_wr="3%", training_acc=-0.08,
                          hw_delta=-18, paddock_v=-0.3, paddock_h=-0.2, paddock_g=-0.4),
              _make_entry(1, "Other", 8.0)]

    sf_s = feature_store.extract_structured_features(entries=strong_e)
    sf_w = feature_store.extract_structured_features(entries=weak_e)

    s_strong = _score(_make_features("Strong", 4.0, sf_s))
    s_weak = _score(_make_features("Weak", 4.0, sf_w))

    assert s_strong > s_weak
    spread = s_strong - s_weak
    assert spread > 3.0, f"Spread should be meaningful: {spread:.2f}"
    print(f"PASS (strong={s_strong:.2f}, weak={s_weak:.2f}, spread={spread:.2f})")


# ══════════════════════════════════════════════════════════
# Feature store & prepare tests
# ══════════════════════════════════════════════════════════

def test_feature_store_extraction():
    print("Test: Feature store extraction...", end=" ")
    entries = [_make_entry(0, "TestHorse", 3.5, jockey_wr="18%", training_acc=0.08, hw_delta=2)]
    sf = feature_store.extract_structured_features(
        entries=entries, track_condition="良", weather="晴",
        temperature="22", cushion_value="9.5", venue="東京",
    )
    assert sf["version"] == feature_store.SCHEMA_VERSION
    assert sf["horses"]["TestHorse"]["jockey_win_rate"] == 0.18
    assert sf["horses"]["TestHorse"]["training_acceleration"] == 0.08
    print("PASS")


def test_feature_store_no_forbidden():
    print("Test: Feature store forbidden keys...", end=" ")
    entry = _make_entry(0, "Test", 5.0)
    entry["confidence"] = 85
    sf = feature_store.extract_structured_features(entries=[entry])
    assert "confidence" not in sf["horses"]["Test"]
    print("PASS")


def test_feature_store_missing_values():
    print("Test: Feature store missing values...", end=" ")
    sparse = {"name": "Sparse", "odds": "0", "age": "", "weight": "", "horse_weight": "",
              "training_physics": {}, "training_nlp": {}, "paddock_scores": {},
              "best_weight_analysis": {}}
    sf = feature_store.extract_structured_features(entries=[sparse])
    h = sf["horses"]["Sparse"]
    assert h["odds"] == 0.0
    assert h["training_acceleration"] == 0.0
    print("PASS")


def test_prepare_passes_structured_features():
    print("Test: Prepare pass-through...", end=" ")
    _generate_test_data(5, with_structured=True)
    data = prepare.load_paired_data()
    features = prepare.extract_features(data[0]["prediction"])
    assert "structured_features" in features
    print("PASS")


def test_prepare_handles_legacy():
    print("Test: Prepare legacy...", end=" ")
    _generate_test_data(5, with_structured=False)
    data = prepare.load_paired_data()
    features = prepare.extract_features(data[0]["prediction"])
    assert "structured_features" not in features
    print("PASS")


# ══════════════════════════════════════════════════════════
# Evaluation & gate tests
# ══════════════════════════════════════════════════════════

def test_data_loading():
    print("Test: Data loading...", end=" ")
    _generate_test_data(20)
    assert len(prepare.load_paired_data()) == 20
    print("PASS")


def test_leakage_prevention():
    print("Test: Leakage...", end=" ")
    assert prepare.validate_no_leakage(
        {"horse_features": [{"name": "A", "odds": 5.0}]},
        {"finishing_order": [{"rank": 1, "name": "A"}]})
    assert not prepare.validate_no_leakage(
        {"horse_features": [{"name": "A", "finishing_order": "bad"}]},
        {"finishing_order": [{"rank": 1, "name": "A"}]})
    print("PASS")


def test_walk_forward_splits():
    print("Test: Walk-forward...", end=" ")
    _generate_test_data(20)
    data = prepare.load_paired_data()
    splits = prepare.walk_forward_splits(data, min_train_size=5, step_size=1)
    assert len(splits) > 0
    for s in splits:
        assert max(e["timestamp"] for e in s["train"]) <= min(e["timestamp"] for e in s["test"])
    print(f"PASS ({len(splits)} folds)")


def test_baseline_evaluation():
    print("Test: Baseline eval...", end=" ")
    _generate_test_data(20)
    r = evaluator.evaluate_walk_forward(evaluator.baseline_score_runner)
    assert "error" not in r and r["num_races"] == 20
    print(f"PASS (ROI={r['roi']:.4f}, Brier={r['brier']:.4f})")


def test_candidate_eval_legacy():
    print("Test: Candidate eval (legacy)...", end=" ")
    _generate_test_data(20, with_structured=False)
    from train import score_runner
    r = evaluator.evaluate_walk_forward(score_runner)
    assert "error" not in r and r["num_races"] == 20
    print(f"PASS (ROI={r['roi']:.4f}, Brier={r['brier']:.4f})")


def test_candidate_eval_structured():
    print("Test: Candidate eval (structured)...", end=" ")
    _generate_test_data(20, with_structured=True)
    from train import score_runner
    r = evaluator.evaluate_walk_forward(score_runner)
    assert "error" not in r and r["num_races"] == 20
    print(f"PASS (ROI={r['roi']:.4f}, Brier={r['brier']:.4f})")


def test_significance_gate():
    print("Test: Significance gate...", end=" ")
    _generate_test_data(20)
    from train import score_runner
    baseline = evaluator.evaluate_walk_forward(evaluator.baseline_score_runner)
    candidate = evaluator.evaluate_walk_forward(score_runner)
    masked = evaluator.evaluate_walk_forward(score_runner, mask_odds=True)
    adoption = evaluator.check_adoption(candidate, baseline, masked, min_races=300)
    assert not adoption["adopted"]
    print("PASS")


def test_determinism():
    print("Test: Determinism...", end=" ")
    _generate_test_data(20, with_structured=True)
    from train import score_runner
    r1 = evaluator.evaluate_walk_forward(score_runner)
    r2 = evaluator.evaluate_walk_forward(score_runner)
    assert r1["roi"] == r2["roi"] and r1["brier"] == r2["brier"]
    print("PASS")


def test_complexity_bounded():
    print("Test: Complexity...", end=" ")
    import train
    source = inspect.getsource(train.score_runner)
    lines = [l for l in source.split("\n") if l.strip() and not l.strip().startswith("#")
             and not l.strip().startswith('"""')]
    conds = sum(1 for l in lines if l.strip().startswith(("if ", "elif ")))
    assert len(lines) < 120, f"Too many lines: {len(lines)}"
    assert conds <= 12, f"Too many conditionals: {conds}"
    print(f"PASS (lines={len(lines)}, conditionals={conds})")


def main():
    print("=" * 60)
    print("PIPELINE TESTS (Expressive Formula v3)")
    print("=" * 60)
    print()

    tests = [
        # Constraints
        test_gemini_independence,
        test_gemini_fields_ignored,
        test_legacy_fallback,
        test_odds_masked_fallback,
        test_score_varies_with_odds,
        # Expressiveness
        test_nonlinear_weight_penalty,
        test_interaction_jockey_grade,
        test_interaction_training_bio,
        test_conditional_heavy_track,
        test_bio_composite_affects_score,
        test_structured_features_differentiate,
        # Feature store & prepare
        test_feature_store_extraction,
        test_feature_store_no_forbidden,
        test_feature_store_missing_values,
        test_prepare_passes_structured_features,
        test_prepare_handles_legacy,
        # Evaluation
        test_data_loading,
        test_leakage_prevention,
        test_walk_forward_splits,
        test_baseline_evaluation,
        test_candidate_eval_legacy,
        test_candidate_eval_structured,
        test_significance_gate,
        test_determinism,
        test_complexity_bounded,
    ]

    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
