"""Accuracy gate for paddock and horse-condition text.

The live pipeline benefits from more public observations, but paddock
text is noisy: prediction articles, betting marks, and search snippets
often sit next to genuine horse-condition comments.  This module keeps
only sourceable observations and rejects speculative text before it can
become a canonical fact.
"""

from __future__ import annotations

import re
from typing import Any


QUALITY_GATE_VERSION = "paddock-quality-v1-2026-05-08"

TRUSTED_STRUCTURED_SOURCES = (
    "netkeiba",
    "tospo",
)

DISALLOWED_SOURCE_PATTERNS = (
    "Yahoo",
    "ニュース検索",
    "search",
    "全文抽出",
    "本文抽出",
    "snippet",
)

SPECULATIVE_PATTERNS = (
    "本命",
    "対抗",
    "単穴",
    "連下",
    "注目",
    "狙い",
    "買い",
    "買う",
    "馬券",
    "推奨",
    "期待",
    "妙味",
    "穴",
    "鉄板",
    "軸",
    "相手",
    "押さえ",
    "有力",
    "勝ち負け",
    "勝てる",
    "勝ちそう",
    "巻き返し",
    "逆転",
    "一発",
    "侮れない",
    "軽視禁物",
    "高配当",
)

OBSERVATION_ANCHORS = (
    "毛艶",
    "毛ヅヤ",
    "ツヤ",
    "艶",
    "馬体",
    "体つき",
    "張り",
    "締ま",
    "太め",
    "余裕",
    "腹",
    "トモ",
    "後肢",
    "後脚",
    "筋肉",
    "踏み込み",
    "歩様",
    "歩き",
    "脚取り",
    "フットワーク",
    "首",
    "硬",
    "柔ら",
    "発汗",
    "汗",
    "入れ込",
    "落ち着",
    "気配",
    "集中",
    "テンション",
    "チャカ",
    "パドック",
    "返し馬",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def assess_paddock_report(text: str, source: str = "") -> dict[str, Any]:
    """Return a quality decision for one paddock report.

    A report is usable only if it is a sourceable observation.  Betting
    language and search/full-page extraction sources are rejected even if
    the text also contains useful-looking words.
    """
    clean_text = _norm(text)
    clean_source = _norm(source)
    reasons: list[str] = []

    if not clean_text:
        reasons.append("empty-text")
    if any(p in clean_source for p in DISALLOWED_SOURCE_PATTERNS):
        reasons.append("disallowed-source")
    for phrase in SPECULATIVE_PATTERNS:
        if phrase in clean_text:
            reasons.append(f"speculative-language:{phrase}")
            break

    has_anchor = any(anchor in clean_text for anchor in OBSERVATION_ANCHORS)
    if not has_anchor:
        reasons.append("no-observation-anchor")

    is_structured = any(
        clean_source.startswith(src) or clean_source == src
        for src in TRUSTED_STRUCTURED_SOURCES
    )
    if clean_text and not is_structured:
        reasons.append("untrusted-or-unstructured-source")

    usable = not reasons and is_structured
    quality_tier = "A" if usable else "REJECT"
    confidence_multiplier = 1.0 if usable else 0.0

    return {
        "usable": usable,
        "quality_tier": quality_tier,
        "confidence_multiplier": confidence_multiplier,
        "reasons": reasons,
        "source": clean_source,
        "version": QUALITY_GATE_VERSION,
    }


def filter_paddock_reports(reports: dict[str, dict]) -> dict[str, dict]:
    """Drop unusable paddock reports while preserving horse keys.

    Empty entries are preserved as empty, so callers can continue to rely
    on the existing `{horse_name: report}` shape.
    """
    filtered: dict[str, dict] = {}
    for horse, report in (reports or {}).items():
        if not isinstance(report, dict):
            filtered[horse] = {"text": "", "source": "", "scores": {}}
            continue

        text = report.get("text", "") or ""
        source = report.get("source", "") or ""
        if not text:
            filtered[horse] = {
                **report,
                "text": "",
                "source": source,
                "scores": report.get("scores", {}) or {},
            }
            continue

        quality = assess_paddock_report(text, source)
        if quality["usable"]:
            filtered[horse] = {**report, "quality": quality}
        else:
            filtered[horse] = {
                "text": "",
                "source": source,
                "scores": {},
                "quality": quality,
            }
    return filtered
