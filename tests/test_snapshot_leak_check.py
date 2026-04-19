"""Tests for snapshot leak detection.

Ensures that:
  1. Snapshot with future recent_races RAISES
  2. Snapshot with unparseable dates RAISES
  3. Snapshot with non-neutral jockey/trainer stats RAISES
  4. Snapshot with forbidden keys RAISES
  5. Valid snapshot passes
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.build_snapshot import LeakError, leak_check, parse_flexible_date


def test_parse_flexible_date():
    assert parse_flexible_date("2025-11-15").isoformat() == "2025-11-15"
    assert parse_flexible_date("2025/11/15").isoformat() == "2025-11-15"
    assert parse_flexible_date("2025年11月15日").isoformat() == "2025-11-15"
    assert parse_flexible_date("25/11/15").isoformat() == "2025-11-15"
    assert parse_flexible_date("25年11月15日").isoformat() == "2025-11-15"
    assert parse_flexible_date("") is None
    assert parse_flexible_date("invalid") is None
    print("  OK test_parse_flexible_date")


def test_leak_check_valid():
    snap = {
        "race_date": "2025-11-22",
        "horses": {
            "h1": {
                "recent_races": [
                    {"date": "2025-10-15", "race_name": "A"},
                    {"date": "2025-09-01", "race_name": "B"},
                ],
            },
        },
        "jockey": {"j1": {"stats": None}},
        "trainer": {"t1": {"stats": None}},
    }
    leak_check(snap)  # should not raise
    print("  OK test_leak_check_valid")


def test_leak_check_future_race():
    snap = {
        "race_date": "2025-11-22",
        "horses": {
            "h1": {
                "recent_races": [
                    {"date": "2025-12-01", "race_name": "FUTURE"},  # after race_date!
                ],
            },
        },
        "jockey": {},
        "trainer": {},
    }
    try:
        leak_check(snap)
        assert False, "Should have raised LeakError"
    except LeakError as e:
        assert "2025-12-01" in str(e) or "FUTURE" in str(e)
    print("  OK test_leak_check_future_race")


def test_leak_check_same_day_race():
    """Date == race_date is still a leak (strictly less than required)."""
    snap = {
        "race_date": "2025-11-22",
        "horses": {
            "h1": {
                "recent_races": [
                    {"date": "2025-11-22", "race_name": "SAME DAY"},
                ],
            },
        },
        "jockey": {},
        "trainer": {},
    }
    try:
        leak_check(snap)
        assert False, "Same-day race should be rejected"
    except LeakError:
        pass
    print("  OK test_leak_check_same_day_race")


def test_leak_check_unparseable_date():
    snap = {
        "race_date": "2025-11-22",
        "horses": {
            "h1": {
                "recent_races": [
                    {"date": "not a date", "race_name": "X"},
                ],
            },
        },
        "jockey": {},
        "trainer": {},
    }
    try:
        leak_check(snap)
        assert False, "Unparseable date should be rejected"
    except LeakError as e:
        assert "unparseable" in str(e).lower()
    print("  OK test_leak_check_unparseable_date")


def test_leak_check_nonneutral_jockey():
    snap = {
        "race_date": "2025-11-22",
        "horses": {},
        "jockey": {
            "j1": {"stats": {"win_rate": "12%", "g1_wins": "5"}},
        },
        "trainer": {},
    }
    try:
        leak_check(snap)
        assert False, "Non-neutral jockey stats should be rejected"
    except LeakError as e:
        assert "non-neutral" in str(e).lower() or "jockey" in str(e).lower()
    print("  OK test_leak_check_nonneutral_jockey")


def test_leak_check_forbidden_key():
    snap = {
        "race_date": "2025-11-22",
        "horses": {},
        "jockey": {},
        "trainer": {},
        "result": {"finishing_order": [{"name": "X", "rank": 1}]},
    }
    try:
        leak_check(snap)
        assert False, "Forbidden key should be rejected"
    except LeakError as e:
        assert "result" in str(e).lower() or "forbidden" in str(e).lower()
    print("  OK test_leak_check_forbidden_key")


def test_leak_check_forbidden_nested():
    snap = {
        "race_date": "2025-11-22",
        "horses": {},
        "jockey": {},
        "trainer": {},
        "entries": [
            {"name": "X", "finishing_order": 1},  # nested forbidden!
        ],
    }
    try:
        leak_check(snap)
        assert False, "Nested forbidden key should be rejected"
    except LeakError as e:
        assert "finishing_order" in str(e)
    print("  OK test_leak_check_forbidden_nested")


if __name__ == "__main__":
    print("Running snapshot leak check tests...")
    test_parse_flexible_date()
    test_leak_check_valid()
    test_leak_check_future_race()
    test_leak_check_same_day_race()
    test_leak_check_unparseable_date()
    test_leak_check_nonneutral_jockey()
    test_leak_check_forbidden_key()
    test_leak_check_forbidden_nested()
    print("\nAll snapshot leak tests passed!")
