import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import grade_strategy as gs


def test_g1_g2_use_umaren_diversified_strategy():
    assert gs.get_strategy_for_grade("G1") == "diversified_1-3_4-7_8+"
    assert gs.get_strategy_for_grade("JpnI") == "diversified_1-3_4-7_8+"
    assert gs.get_strategy_for_grade("G2") == "diversified_1-3_4-7_8+"
    assert gs.get_strategy_for_grade("JpnII") == "diversified_1-3_4-7_8+"


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
