import grade_strategy as gs


def _ranked():
    return [
        {
            "name": "Alpha",
            "odds": 2.0,
            "win_prob": 0.30,
            "jockey_win_rate": 0.05,
            "trainer_win_rate": 0.05,
        },
        {
            "name": "Bravo",
            "odds": 4.0,
            "win_prob": 0.22,
            "jockey_win_rate": 0.20,
            "trainer_win_rate": 0.14,
        },
        {
            "name": "Charlie",
            "odds": 7.0,
            "win_prob": 0.16,
            "jockey_win_rate": 0.18,
            "trainer_win_rate": 0.03,
        },
        {
            "name": "Delta",
            "odds": 12.0,
            "win_prob": 0.10,
            "jockey_win_rate": 0.04,
            "trainer_win_rate": 0.20,
        },
    ]


def test_build_prediction_variants_returns_primary_and_experimental():
    variants = gs.build_prediction_variants(_ranked(), "G1")

    assert variants["primary"]["top3"][0]["name"] == "Alpha"
    assert variants["experimental"]["candidate_id"] == "experimental_jockey_trainer"
    assert len(variants["experimental"]["top3"]) == 3
    assert variants["experimental"]["top3"][0]["candidate_type"] == "experimental_jockey_trainer"


def test_g2_primary_uses_market_buckets():
    variants = gs.build_prediction_variants(_ranked(), "G2")

    assert variants["primary"]["strategy"] == "diversified_1-3_4-7_8+"
    assert [h["market_rank"] for h in variants["primary"]["top3"]] == [1, 4, 2]
