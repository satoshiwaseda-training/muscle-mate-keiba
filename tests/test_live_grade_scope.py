from datetime import date

import scraper


def test_live_grade_filter_excludes_g3():
    assert scraper.LIVE_GRADE_FILTER == ("G1", "G2")


def test_fetch_race_list_filters_jra_fallback(monkeypatch):
    monkeypatch.setattr(scraper, "fetch_race_list_netkeiba", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        scraper,
        "fetch_race_list_jra",
        lambda *_args, **_kwargs: [
            {"race_id": "g1", "race_name": "Test G1", "grade": "G1"},
            {"race_id": "g2", "race_name": "Test G2", "grade": "G2"},
            {"race_id": "g3", "race_name": "Test G3", "grade": "G3"},
            {"race_id": "open", "race_name": "Test Open", "grade": ""},
        ],
    )

    races = scraper.fetch_race_list(date(2026, 5, 9))

    assert [race["race_id"] for race in races] == ["g1", "g2"]
