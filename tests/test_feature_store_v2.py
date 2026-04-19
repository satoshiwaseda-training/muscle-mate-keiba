"""Integration test for feature_store v2 (pedigree + camp features).

Tests the full path: entries → extract_structured_features → pedigree composites.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import feature_store as fs


def test_v2_schema_version():
    assert fs.SCHEMA_VERSION == 2
    print("  OK test_v2_schema_version")


def test_v2_pedigree_fields_present():
    """Verify pedigree fields appear in extracted features."""
    entries = [
        {
            "name": "TestHorse",
            "number": "1", "waku": "1", "odds": "3.5",
            "age": "3", "weight": "56.0",
            "horse_weight": "480(+2)",
            "stable": "Miho", "transport_stress": "low",
            "jockey_win_rate": "15%", "jockey_g1_wins": "3",
            "trainer_win_rate": "12%",
            "training_physics": {"final_split": 0, "acceleration_rate": 0.05, "cardio_index": 0.4},
            "training_nlp": {},
            "paddock_scores": {},
            "best_weight_analysis": {},
            "ritto": "Ritto",
            "recent_form": "", "weight_trend": "",
            # v2 fields from scraper
            "sire": "ディープインパクト",
            "dam": "TestDam",
            "damsire": "キングカメハメハ",
            "breeder": "ノーザンファーム",
            "owner": "サンデーレーシング",
        },
    ]
    sf = fs.extract_structured_features(
        entries=entries,
        race_info={"surface": "芝2000m"},
        track_condition="良",
        venue="東京",
    )
    assert sf["version"] == 2
    horse = sf["horses"]["TestHorse"]

    # Raw entity names
    assert horse["sire_name"] == "ディープインパクト"
    assert horse["dam_name"] == "TestDam"
    assert horse["damsire_name"] == "キングカメハメハ"
    assert horse["breeder_name"] == "ノーザンファーム"
    assert horse["owner_name"] == "サンデーレーシング"

    # Composite scores
    assert "pedigree_composite" in horse
    assert "camp_composite" in horse
    assert horse["pedigree_composite"] > 0.5  # elite pedigree
    assert horse["camp_composite"] > 0.5  # elite camp

    # Individual scores
    assert "sire_tier_score" in horse
    assert "sire_distance_fit" in horse
    assert "sire_surface_fit" in horse
    assert "breeder_tier_score" in horse
    assert "owner_tier_score" in horse

    # Signal quality
    assert horse["pedigree_has_signal"] is True
    assert horse["camp_has_signal"] is True
    assert horse["missing_feature_count"] == 0

    print(f"  OK test_v2_pedigree_fields_present "
          f"(ped={horse['pedigree_composite']:.3f} camp={horse['camp_composite']:.3f})")


def test_v2_missing_pedigree_safe():
    """Verify that missing pedigree data doesn't crash or penalize."""
    entries = [
        {
            "name": "UnknownHorse",
            "number": "5", "waku": "3", "odds": "10.0",
            "age": "4", "weight": "54.0",
            "horse_weight": "460(-4)",
            "stable": "", "transport_stress": "",
            "jockey_win_rate": "", "jockey_g1_wins": "",
            "trainer_win_rate": "",
            "training_physics": {}, "training_nlp": {},
            "paddock_scores": {}, "best_weight_analysis": {},
            "ritto": "", "recent_form": "", "weight_trend": "",
            # No pedigree data at all
        },
    ]
    sf = fs.extract_structured_features(entries=entries, venue="中山")
    horse = sf["horses"]["UnknownHorse"]

    # Should all be neutral
    assert horse["pedigree_composite"] == 0.5
    assert horse["camp_composite"] == 0.5
    assert horse["sire_distance_fit"] == 0.5
    assert horse["pedigree_has_signal"] is False
    assert horse["camp_has_signal"] is False
    print("  OK test_v2_missing_pedigree_safe")


def test_v2_forbidden_keys_still_blocked():
    """Ensure pedigree feature addition didn't break forbidden key check."""
    entries = [
        {
            "name": "Horse",
            "number": "1", "waku": "1", "odds": "2.0",
            "age": "3", "weight": "56.0", "horse_weight": "480(0)",
            "stable": "", "transport_stress": "",
            "jockey_win_rate": "", "jockey_g1_wins": "",
            "trainer_win_rate": "",
            "training_physics": {}, "training_nlp": {},
            "paddock_scores": {}, "best_weight_analysis": {},
            "ritto": "", "recent_form": "", "weight_trend": "",
            "sire": "Test", "dam": "Test", "damsire": "Test",
            "breeder": "Test", "owner": "Test",
        },
    ]
    sf = fs.extract_structured_features(entries=entries)
    # Should not contain forbidden keys
    for key in fs._FORBIDDEN_KEYS:
        assert key not in sf["horses"]["Horse"], f"Forbidden key found: {key}"
    print("  OK test_v2_forbidden_keys_still_blocked")


if __name__ == "__main__":
    print("Running feature_store v2 integration tests...")
    test_v2_schema_version()
    test_v2_pedigree_fields_present()
    test_v2_missing_pedigree_safe()
    test_v2_forbidden_keys_still_blocked()
    print("\nAll integration tests passed!")
