import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import paddock_quality as pq
import paddock_sources as ps


def test_accepts_structured_observation_text():
    quality = pq.assess_paddock_report(
        "毛艶が良く、歩様もスムーズ。落ち着きがある。",
        "netkeiba",
    )

    assert quality["usable"] is True
    assert quality["quality_tier"] == "A"


def test_rejects_yahoo_news_snippets_even_with_observation_words():
    quality = pq.assess_paddock_report(
        "パドックでは毛艶が良いとの声もあるが本命候補。",
        "Yahoo!ニュース",
    )

    assert quality["usable"] is False
    assert "disallowed-source" in quality["reasons"]


def test_rejects_full_page_extraction_source():
    quality = pq.assess_paddock_report(
        "馬体に張りがあり、踏み込みも深い。",
        "netkeiba(全文抽出)",
    )

    assert quality["usable"] is False
    assert "disallowed-source" in quality["reasons"]


def test_rejects_betting_language_from_trusted_source():
    quality = pq.assess_paddock_report(
        "馬体に張りがあり、ここは買いの一頭。",
        "tospo",
    )

    assert quality["usable"] is False
    assert any(reason.startswith("speculative-language") for reason in quality["reasons"])


def test_rejects_untrusted_source_even_if_observational():
    quality = pq.assess_paddock_report(
        "毛艶が良く、踏み込みも深い。",
        "radio-nikkei",
    )

    assert quality["usable"] is False
    assert "untrusted-or-unstructured-source" in quality["reasons"]


def test_filter_preserves_shape_and_blanks_bad_reports():
    filtered = pq.filter_paddock_reports({
        "A": {"text": "毛艶が良く歩様もスムーズ。", "source": "netkeiba", "scores": {"x": 1}},
        "B": {"text": "本命候補で馬券妙味あり。", "source": "tospo", "scores": {"x": 1}},
        "C": {"text": "", "source": "", "scores": {}},
    })

    assert filtered["A"]["text"]
    assert filtered["A"]["quality"]["usable"] is True
    assert filtered["B"]["text"] == ""
    assert filtered["B"]["quality"]["usable"] is False
    assert filtered["C"]["text"] == ""


def test_multi_paddock_collection_is_g1_g2_only():
    assert ps._grade_triggers("G1") is True
    assert ps._grade_triggers("G2") is True
    assert ps._grade_triggers("G3") is False
