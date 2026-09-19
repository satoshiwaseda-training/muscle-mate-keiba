"""Regression: live path must derive grade from the race name (v5.11).

scraper.fetch_race_info_netkeiba never returns a `grade` key, so before
v5.11 every live prediction carried grade="" — which silently disabled the
grade-dependent presentation layers (grade_strategy, bet_recommender) and
zeroed the jockey×grade interaction in score_runner on the live path only.
"""
import live_pipeline as lp


def test_grade_from_race_name_tags():
    assert lp._grade_from_race_name("桜花賞 (G1)") == "G1"
    assert lp._grade_from_race_name("目黒記念 (G2)") == "G2"
    assert lp._grade_from_race_name("小倉大賞典 (G3)") == "G3"
    assert lp._grade_from_race_name("帝王賞 (JpnI)") == "JpnI"
    assert lp._grade_from_race_name("ローズS (g2)") == "G2"


def test_grade_from_race_name_absent():
    assert lp._grade_from_race_name("") == ""
    assert lp._grade_from_race_name(None) == ""
    assert lp._grade_from_race_name("新馬戦") == ""


def test_race_info_grade_takes_precedence():
    # Snapshot/backtest path supplies race_info["grade"]; the name is a fallback only.
    race_info = {"grade": "G1"}
    assert (race_info.get("grade", "") or lp._grade_from_race_name("X (G2)")) == "G1"
    race_info = {}
    assert (race_info.get("grade", "") or lp._grade_from_race_name("X (G2)")) == "G2"
