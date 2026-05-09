import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grade_strategy as gs


def test_g1_g2_use_umaren_diversified_strategy():
    assert gs.get_strategy_for_grade("G1") == "diversified_1-3_4-7_8+"
    assert gs.get_strategy_for_grade("JpnI") == "diversified_1-3_4-7_8+"
    assert gs.get_strategy_for_grade("G2") == "diversified_1-3_4-7_8+"
    assert gs.get_strategy_for_grade("JpnII") == "diversified_1-3_4-7_8+"


def test_recent_umaren_guard_returns_grade_warning():
    guard = gs.recent_umaren_guard_for_grade("G2")

    assert guard["sample"] == 9
    assert guard["umaren_roi"] < 0
    assert "馬連" in guard["message"]


def test_buying_style_plan_prioritizes_profitable_g2():
    g2 = gs.buying_style_plan_for_grade("G2")
    g3 = gs.buying_style_plan_for_grade("G3")

    assert g2["action"] == "BET"
    assert g2["stake_yen"] == 600
    assert g2["historical_roi"] > 0
    assert g3["action"] == "WATCH"
    assert g3["stake_yen"] == 0


def test_diversified_top3_picks_one_from_each_market_bucket():
    ranked = [
        {"name": "A", "odds": 2.0, "win_prob": 0.30},
        {"name": "B", "odds": 4.0, "win_prob": 0.20},
        {"name": "C", "odds": 8.0, "win_prob": 0.18},
        {"name": "D", "odds": 12.0, "win_prob": 0.16},
        {"name": "E", "odds": 20.0, "win_prob": 0.10},
        {"name": "F", "odds": 30.0, "win_prob": 0.06},
        {"name": "G", "odds": 40.0, "win_prob": 0.05},
        {"name": "H", "odds": 50.0, "win_prob": 0.04},
    ]
    market = gs.build_market_rank_map(ranked)

    picked = gs.pick_diversified_top3(
        ranked,
        market,
        strategy="diversified_1-3_4-7_8+",
    )

    assert [h["name"] for h in picked] == ["A", "D", "H"]
    assert [h["market_rank"] for h in picked] == [1, 4, 8]


def test_prediction_variants_keep_primary_and_experimental_separate():
    ranked = [
        {"name": "A", "odds": 2.0, "win_prob": 0.30, "jockey_win_rate": 0.08, "trainer_win_rate": 0.05},
        {"name": "B", "odds": 4.0, "win_prob": 0.20, "jockey_win_rate": 0.20, "trainer_win_rate": 0.18},
        {"name": "C", "odds": 8.0, "win_prob": 0.18, "jockey_win_rate": 0.06, "trainer_win_rate": 0.04},
        {"name": "D", "odds": 12.0, "win_prob": 0.16, "jockey_win_rate": 0.19, "trainer_win_rate": 0.17},
        {"name": "E", "odds": 20.0, "win_prob": 0.10, "jockey_win_rate": 0.05, "trainer_win_rate": 0.03},
        {"name": "F", "odds": 30.0, "win_prob": 0.06, "jockey_win_rate": 0.18, "trainer_win_rate": 0.16},
        {"name": "G", "odds": 40.0, "win_prob": 0.05, "jockey_win_rate": 0.04, "trainer_win_rate": 0.03},
        {"name": "H", "odds": 50.0, "win_prob": 0.04, "jockey_win_rate": 0.17, "trainer_win_rate": 0.15},
    ]

    variants = gs.build_prediction_variants(ranked, "G2")

    assert variants["primary"]["candidate_id"] == "primary_current"
    assert variants["experimental"]["candidate_id"] == "experimental_jockey_trainer"
    assert [h["name"] for h in variants["primary"]["top3"]] == ["A", "D", "H"]
    assert [h["name"] for h in variants["experimental"]["top3"]] == ["B", "D", "H"]
    assert all("experimental_score" in h for h in variants["experimental"]["top3"])
