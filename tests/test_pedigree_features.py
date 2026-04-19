"""Tests for pedigree_features.py and entity_tier.py.

Validates:
  1. Entity tier lookups (known + unknown entities)
  2. Distance/surface/heavy-track fit calculations
  3. Composite score computation
  4. Feature extraction completeness
  5. Neutral defaults for missing data (no crashes, no leakage)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import entity_tier as et
import pedigree_features as pf


def test_sire_tier_lookup():
    """Known sires return correct tier; unknown return 2."""
    assert et.get_sire_tier("ディープインパクト") == 5
    assert et.get_sire_tier("キタサンブラック") == 5
    assert et.get_sire_tier("ハーツクライ") == 4
    assert et.get_sire_tier("リアルスティール") == 3
    assert et.get_sire_tier("完全に未知の種牡馬") == 2
    assert et.get_sire_tier("") == 2
    assert et.get_sire_tier(None) == 2
    print("  ✓ test_sire_tier_lookup")


def test_damsire_tier_lookup():
    assert et.get_damsire_tier("ディープインパクト") == 5
    assert et.get_damsire_tier("クロフネ") == 4
    assert et.get_damsire_tier("不明") == 2
    print("  ✓ test_damsire_tier_lookup")


def test_breeder_tier_lookup():
    assert et.get_breeder_tier("ノーザンファーム") == 5
    assert et.get_breeder_tier("社台ファーム") == 5
    assert et.get_breeder_tier("追分ファーム") == 4
    assert et.get_breeder_tier("個人牧場") == 2
    print("  ✓ test_breeder_tier_lookup")


def test_owner_tier_lookup():
    assert et.get_owner_tier("サンデーレーシング") == 5
    assert et.get_owner_tier("個人馬主") == 2
    print("  ✓ test_owner_tier_lookup")


def test_sire_distance_profile():
    # ディープインパクト: (1600, 2000, 2500)
    prof = et.get_sire_distance_profile("ディープインパクト")
    assert prof == (1600, 2000, 2500)
    # Unknown sire
    prof2 = et.get_sire_distance_profile("不明")
    assert prof2 == et.DEFAULT_DISTANCE_PROFILE
    print("  ✓ test_sire_distance_profile")


def test_sire_distance_fit():
    # ディープインパクト at 2000m (peak) → high score
    score, sig = pf.sire_distance_fit("ディープインパクト", 2000)
    assert sig is True
    assert score >= 0.9, f"Peak distance should be ≥0.9, got {score}"

    # ディープインパクト at 1200m (too short) → low score
    score2, sig2 = pf.sire_distance_fit("ディープインパクト", 1200)
    assert sig2 is True
    assert score2 < 0.5, f"Out-of-range should be <0.5, got {score2}"

    # ディープインパクト at 1800m (in range) → good score
    score3, sig3 = pf.sire_distance_fit("ディープインパクト", 1800)
    assert sig3 is True
    assert 0.5 <= score3 <= 1.0

    # Unknown sire → neutral
    score4, sig4 = pf.sire_distance_fit("不明", 2000)
    assert sig4 is False
    assert score4 == 0.5

    # Empty → neutral
    score5, sig5 = pf.sire_distance_fit("", 2000)
    assert sig5 is False
    print("  ✓ test_sire_distance_fit")


def test_sire_surface_fit():
    # ディープインパクト on turf → high
    score, sig = pf.sire_surface_fit("ディープインパクト", "芝")
    assert sig is True
    assert score >= 0.9

    # ディープインパクト on dirt → low
    score2, sig2 = pf.sire_surface_fit("ディープインパクト", "ダート")
    assert sig2 is True
    assert score2 <= 0.3

    # ヘニーヒューズ on dirt → high
    score3, sig3 = pf.sire_surface_fit("ヘニーヒューズ", "ダート")
    assert sig3 is True
    assert score3 >= 0.85

    # Unknown → neutral
    score4, sig4 = pf.sire_surface_fit("不明", "芝")
    assert sig4 is False
    print("  ✓ test_sire_surface_fit")


def test_sire_heavy_track_fit():
    # ゴールドシップ on heavy → high
    score, sig = pf.sire_heavy_track_fit("ゴールドシップ", "重")
    assert sig is True
    assert score >= 0.8

    # ディープインパクト on heavy → lower
    score2, sig2 = pf.sire_heavy_track_fit("ディープインパクト", "重")
    assert sig2 is True
    assert score2 <= 0.5

    # Any sire on 良 → no signal
    score3, sig3 = pf.sire_heavy_track_fit("ゴールドシップ", "良")
    assert sig3 is False
    print("  ✓ test_sire_heavy_track_fit")


def test_breeder_tier_score():
    score, sig = pf.breeder_tier_score("ノーザンファーム")
    assert sig is True
    assert score == 1.0  # tier 5 → (5-1)/4 = 1.0

    score2, sig2 = pf.breeder_tier_score("個人牧場")
    assert sig2 is True
    assert score2 == 0.25  # known but not in table → 0.25

    score3, sig3 = pf.breeder_tier_score("")
    assert sig3 is False
    assert score3 == 0.5  # empty → neutral
    print("  ✓ test_breeder_tier_score")


def test_owner_tier_score():
    score, sig = pf.owner_tier_score("サンデーレーシング")
    assert sig is True
    assert score == 1.0

    score2, sig2 = pf.owner_tier_score("")
    assert sig2 is False
    print("  ✓ test_owner_tier_score")


def test_pedigree_composite():
    result = pf.pedigree_composite(
        sire_name="ディープインパクト",
        damsire_name="キングカメハメハ",
        race_distance=2000,
        surface="芝",
        track_condition="良",
    )
    assert result["has_signal"] is True
    assert result["n_dims_used"] == 4
    assert 0.7 <= result["composite"] <= 1.0, \
        f"Elite pedigree on ideal surface/distance should score high, got {result['composite']}"

    # Unknown everything → no signal
    result2 = pf.pedigree_composite("", "", 0, "", "")
    assert result2["has_signal"] is False
    assert result2["composite"] == 0.5
    print("  ✓ test_pedigree_composite")


def test_camp_composite():
    result = pf.camp_composite("ノーザンファーム", "サンデーレーシング", "リトー")
    assert result["has_signal"] is True
    assert result["composite"] > 0.7

    # Unknown everything
    result2 = pf.camp_composite("", "", "")
    assert result2["has_signal"] is False
    assert result2["composite"] == 0.5
    print("  ✓ test_camp_composite")


def test_extract_pedigree_features_completeness():
    """Verify all expected keys are present in extraction output."""
    feats = pf.extract_pedigree_features(
        sire_name="ディープインパクト",
        dam_name="テストダム",
        damsire_name="キングカメハメハ",
        breeder_name="ノーザンファーム",
        owner_name="サンデーレーシング",
        ritto="リトー",
        race_distance=2000,
        surface="芝",
        track_condition="良",
    )
    required_keys = [
        "sire_name", "dam_name", "damsire_name", "breeder_name", "owner_name",
        "sire_tier_score", "damsire_tier_score",
        "sire_distance_fit", "sire_surface_fit", "sire_heavy_track_fit",
        "breeder_tier_score", "owner_tier_score", "external_stable_score",
        "pedigree_composite", "camp_composite",
        "pedigree_has_signal", "camp_has_signal",
        "pedigree_n_dims", "camp_n_dims",
        "missing_feature_count",
    ]
    for k in required_keys:
        assert k in feats, f"Missing key: {k}"

    assert feats["missing_feature_count"] == 0  # all signals present
    assert feats["pedigree_has_signal"] is True
    assert feats["camp_has_signal"] is True
    print("  ✓ test_extract_pedigree_features_completeness")


def test_no_crash_on_all_missing():
    """Verify graceful handling when everything is unknown."""
    feats = pf.extract_pedigree_features(
        sire_name="", dam_name="", damsire_name="",
        breeder_name="", owner_name="", ritto="",
        race_distance=0, surface="", track_condition="",
    )
    assert feats["pedigree_composite"] == 0.5
    assert feats["camp_composite"] == 0.5
    assert feats["missing_feature_count"] == 7  # all 7 signals missing
    print("  ✓ test_no_crash_on_all_missing")


def test_score_runner_integration():
    """Verify score_runner doesn't crash with new pedigree fields."""
    from train import score_runner

    features = {
        "grade": "G1",
        "num_horses": 16,
        "horse_features": [
            {"name": "テスト馬", "rank": 1, "odds": 3.0,
             "confidence": 0, "ev_gap": 0, "bet": ""},
        ],
        "structured_features": {
            "version": 2,
            "race": {
                "venue": "東京", "surface": "芝", "distance": 2400,
                "track_condition": "良", "weather": "晴",
                "temperature": 18.0, "cushion_value": 9.0, "num_horses": 16,
            },
            "horses": {
                "テスト馬": {
                    "odds": 3.0, "age": 4, "jockey_win_rate": 0.15,
                    "training_acceleration": 0.05, "horse_weight_delta": 2.0,
                    "paddock_vascularity": 0.3, "paddock_hindquarter": 0.2,
                    "paddock_gait": 0.4, "training_cardio_index": 0.6,
                    # v2 fields
                    "pedigree_composite": 0.75,
                    "camp_composite": 0.80,
                    "sire_distance_fit": 0.90,
                },
            },
        },
    }
    result = score_runner(features, {})
    assert "top_confidence" in result
    conf = result["top_confidence"]
    assert 2.0 <= conf <= 95.0, f"top_confidence out of range: {conf}"

    # Compare with a horse without pedigree info
    features2 = dict(features)
    sf2 = dict(features["structured_features"])
    h2 = dict(sf2["horses"]["テスト馬"])
    h2["pedigree_composite"] = 0.5  # neutral
    h2["camp_composite"] = 0.5
    h2["sire_distance_fit"] = 0.5
    sf2["horses"] = {"テスト馬": h2}
    features2["structured_features"] = sf2
    result2 = score_runner(features2, {})
    conf2 = result2["top_confidence"]

    # The pedigree-enhanced score should be higher (elite pedigree)
    assert conf > conf2, \
        f"Elite pedigree ({conf:.2f}) should score higher than neutral ({conf2:.2f})"
    # But the difference should be small (weak coefficients).
    # Threshold bumped from 5.0 → 6.5 in v3 (camp coefficient 0.015 → 0.028).
    diff = conf - conf2
    assert diff < 6.5, f"Difference too large ({diff:.2f}), coefficients may be too strong"
    print(f"  ✓ test_score_runner_integration (elite={conf:.2f} neutral={conf2:.2f} diff={diff:.2f})")


if __name__ == "__main__":
    print("Running pedigree feature tests...")
    test_sire_tier_lookup()
    test_damsire_tier_lookup()
    test_breeder_tier_lookup()
    test_owner_tier_lookup()
    test_sire_distance_profile()
    test_sire_distance_fit()
    test_sire_surface_fit()
    test_sire_heavy_track_fit()
    test_breeder_tier_score()
    test_owner_tier_score()
    test_pedigree_composite()
    test_camp_composite()
    test_extract_pedigree_features_completeness()
    test_no_crash_on_all_missing()
    test_score_runner_integration()
    print("\nAll tests passed!")
