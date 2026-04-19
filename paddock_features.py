"""Structured paddock feature extraction.

Schema (all in [0.0, 1.0], neutral = 0.50):

    gait_score           — stride / footwork / rhythm quality
    hindquarter_strength — rear-leg muscle / propulsion readiness
    vascularity          — skin tightness / coat finish / vein visibility
    mental_state         — composure / focus / absence of agitation

Pipeline:

    1. `extract_per_horse_comments(page_text, horse_names)`
         slices a netkeiba-paddock-page full text into one segment per
         horse, pulling out the letter grade (A/B/C/D/E) and the free
         text comment that follows each horse's name on the page.

    2. `score_comment(text) -> dict`
         maps a single comment into the 4-dim schema using longest-
         match-first weighted phrase lookup with local-window negation.

    3. `to_score_runner_keys(feat_01)`
         converts the 4 new [0, 1] features into the 3 existing
         [-1, 1] keys that `score_runner` already reads, so integration
         does not require modifying `train.py`.

No network I/O. No framework imports.
"""

from __future__ import annotations

import re
from typing import Iterable


# ── Dictionaries ─────────────────────────────────────────
# Each entry: (phrase, score) where score is in [0.0, 1.0].
# Longer phrases first so more specific matches win.

_GAIT_PHRASES: list[tuple[str, float]] = [
    # strong positive
    ("伸びのあるストライド", 0.95),
    ("軽快なフットワーク", 0.90),
    ("力強い踏み込み", 0.90),
    ("しなやかな動き", 0.88),
    ("軽快な動き", 0.85),
    ("歩様が滑らか", 0.85),
    ("弾むような", 0.85),
    ("伸びのある歩様", 0.82),
    ("リズム良く", 0.80),
    ("歩様良好", 0.80),
    ("フットワーク軽", 0.78),
    ("歩様が良い", 0.75),
    ("踏み込み深い", 0.75),
    ("脚どり軽", 0.75),
    ("首の使い良", 0.72),
    ("軽快", 0.68),
    # mild positive
    ("スムーズ", 0.62),
    ("柔軟", 0.60),
    # mild negative
    ("歩様がやや固", 0.38),
    ("歩様が硬", 0.30),
    ("歩様の硬さ", 0.28),
    ("踏み込み浅", 0.30),
    ("脚どり重", 0.28),
    ("ぎこちない", 0.22),
    ("歩幅が狭い", 0.30),
    ("歩様が固", 0.28),
    # strong negative
    ("ガタガタ", 0.15),
    ("歩様悪", 0.15),
]

_HINDQUARTER_PHRASES: list[tuple[str, float]] = [
    # strong positive
    ("後肢の踏み込み強", 0.92),
    ("トモの張りが素晴らし", 0.92),
    ("筋肉の張りが良", 0.88),
    ("パンプアップ", 0.88),
    ("トモの筋肉が充実", 0.85),
    ("後躯が発達", 0.82),
    ("トモしっかり", 0.80),
    ("力強いトモ", 0.80),
    ("腰の回転が良", 0.78),
    ("後肢しっかり", 0.75),
    ("筋肉の盛り上がり", 0.78),
    ("推進力ありそう", 0.72),
    # mild positive
    ("トモ良", 0.65),
    ("筋肉質", 0.62),
    # mild negative
    ("トモがやや甘", 0.38),
    ("後肢が細", 0.32),
    ("トモが薄", 0.30),
    ("腰が甘", 0.30),
    ("ヒップが小", 0.35),
    # strong negative
    ("トモ甘", 0.25),
    ("推進力不足", 0.15),
]

_VASCULARITY_PHRASES: list[tuple[str, float]] = [
    # strong positive
    ("皮膚が薄く血管が浮", 0.95),
    ("毛ヅヤが素晴らし", 0.92),
    ("毛艶が冴え", 0.90),
    ("仕上がり抜群", 0.90),
    ("張り艶申し分な", 0.88),
    ("毛ヅヤ良く馬体もシャープ", 0.88),
    ("毛色に光沢", 0.85),
    ("ピカピカに輝", 0.85),
    ("張り艶良", 0.80),
    ("引き締まった馬体", 0.80),
    ("毛艶良", 0.78),
    ("毛ヅヤ良", 0.78),
    ("馬体シャープ", 0.78),
    ("皮膚が薄", 0.75),
    ("血管が浮", 0.75),
    ("仕上がり良", 0.75),
    ("ツヤ良", 0.72),
    ("輝きがあり", 0.72),
    # mild positive
    ("悪くない仕上がり", 0.58),
    ("まずまずの仕上", 0.55),
    # mild negative
    ("毛ヅヤやや劣", 0.38),
    ("毛艶がくすん", 0.30),
    ("毛ヅヤ悪", 0.25),
    ("ぼてっとした", 0.25),
    ("太め残", 0.22),
    ("太め", 0.30),
    ("疲労感", 0.25),
    # strong negative
    ("仕上がり物足りな", 0.20),
    ("毛ヅヤに張りがな", 0.18),
]

_MENTAL_PHRASES: list[tuple[str, float]] = [
    # strong positive
    ("落ち着き払って", 0.90),
    ("冷静そのもの", 0.90),
    ("集中力の高さ", 0.88),
    ("気合が乗", 0.85),
    ("目力あり", 0.82),
    ("堂々とした", 0.82),
    ("落ち着きがあり", 0.80),
    ("リラックス", 0.78),
    ("精神的に安定", 0.78),
    ("気合十分", 0.78),
    ("集中でき", 0.72),
    ("落ち着き", 0.70),
    # mild positive
    ("穏やか", 0.62),
    # mild negative
    ("やや気負い", 0.40),
    ("気合が空回", 0.32),
    ("集中しきれ", 0.35),
    ("ちょっとチャカつ", 0.35),
    # strong negative
    ("イレ込み気味", 0.22),
    ("入れ込み", 0.22),
    ("チャカつく", 0.20),
    ("チャカチャカ", 0.20),
    ("落ち着きなく", 0.18),
    ("カリカリして", 0.15),
    ("急上昇まではどうか", 0.42),  # literal phrase from our sample
]


# Sort each dictionary by descending phrase length for longest-match-first
def _sort_longest_first(pairs):
    return sorted(pairs, key=lambda kv: -len(kv[0]))


_GAIT_SORTED = _sort_longest_first(_GAIT_PHRASES)
_HINDQ_SORTED = _sort_longest_first(_HINDQUARTER_PHRASES)
_VASC_SORTED = _sort_longest_first(_VASCULARITY_PHRASES)
_MENTAL_SORTED = _sort_longest_first(_MENTAL_PHRASES)


# Letter grade → [0, 1] baseline prior (used only if no phrase hits)
_GRADE_LETTER_PRIOR = {
    "S": 0.90, "A": 0.75, "B": 0.55, "C": 0.40, "D": 0.25, "E": 0.15,
}


# ── Per-horse comment extraction ─────────────────────────

_GRADE_PREFIX = re.compile(r"^\s*([SABCDE])\s+(.*)$", re.DOTALL)


def extract_per_horse_comments(
    page_text: str,
    horse_names: Iterable[str],
) -> dict[str, dict]:
    """Slice a full paddock-page string into per-horse segments.

    Strategy: find each horse name's first occurrence in the text and
    cut from that offset to the next horse name (or end of page). Within
    the cut, look for a leading letter grade (A/B/C/D/E) followed by the
    free-text comment. Returns {name: {"grade_letter": str, "comment": str}}.

    Horses whose name does not appear in the page text are omitted.
    """
    if not page_text:
        return {}
    positions: list[tuple[int, str]] = []
    for name in horse_names:
        if not name:
            continue
        idx = page_text.find(name)
        if idx >= 0:
            positions.append((idx, name))
    if not positions:
        return {}
    positions.sort()
    # Sentinel
    positions.append((len(page_text), ""))

    out: dict[str, dict] = {}
    for (start, name), (end, _) in zip(positions[:-1], positions[1:]):
        seg = page_text[start + len(name): end]
        # Trim hashes / spaces / single-char noise at head
        seg = seg.lstrip(" \u3000\t\n\r")
        m = _GRADE_PREFIX.match(seg)
        if m:
            out[name] = {"grade_letter": m.group(1), "comment": m.group(2)[:400].strip()}
        else:
            out[name] = {"grade_letter": "", "comment": seg[:400].strip()}
    return out


# ── Phrase scoring with simple negation handling ─────────

_NEGATION_WINDOW = 6  # chars after a phrase; look for negation markers
_NEGATION_MARKERS = ("ない", "ず", "ぬ", "ではな", "せず", "まで", "どうか")


def _phrase_score(
    text: str,
    sorted_phrases: list[tuple[str, float]],
) -> tuple[float, int]:
    """Return (aggregate_score_in_[0,1], n_hits).

    Longest-match-first scan consuming matched characters to avoid
    nested double counts. Neutral 0.50 if no hits.
    """
    if not text:
        return (0.50, 0)
    remaining = text
    scores: list[float] = []
    for phrase, score in sorted_phrases:
        while phrase in remaining:
            idx = remaining.find(phrase)
            end = idx + len(phrase)
            # Local-window negation flip
            tail = remaining[end: end + _NEGATION_WINDOW]
            negated = any(m in tail for m in _NEGATION_MARKERS)
            effective = score
            if negated:
                effective = 1.0 - score   # mirror across neutral
            scores.append(effective)
            # Consume match region so overlapping phrases don't double-count
            remaining = remaining[:idx] + "\u0000" * len(phrase) + remaining[end:]
    if not scores:
        return (0.50, 0)
    return (sum(scores) / len(scores), len(scores))


def score_comment(text: str) -> dict:
    """Score one horse's paddock comment against the 4-dim schema.

    Returns:
      {
        "gait_score":           float in [0, 1],
        "hindquarter_strength": float in [0, 1],
        "vascularity":          float in [0, 1],
        "mental_state":         float in [0, 1],
        "hits":                 int,   # total phrases matched across dims
      }
    """
    g, gh = _phrase_score(text, _GAIT_SORTED)
    h, hh = _phrase_score(text, _HINDQ_SORTED)
    v, vh = _phrase_score(text, _VASC_SORTED)
    m, mh = _phrase_score(text, _MENTAL_SORTED)
    return {
        "gait_score": round(g, 3),
        "hindquarter_strength": round(h, 3),
        "vascularity": round(v, 3),
        "mental_state": round(m, 3),
        "hits": gh + hh + vh + mh,
    }


def score_from_segment(segment: dict) -> dict:
    """Score a per-horse segment emitted by `extract_per_horse_comments`.

    Uses phrase hits from the free-text comment. If no phrases hit at all
    but a letter grade is present, falls back to the grade-letter prior
    applied uniformly across all 4 dimensions.
    """
    text = (segment or {}).get("comment", "") or ""
    letter = (segment or {}).get("grade_letter", "") or ""
    scored = score_comment(text)
    if scored["hits"] == 0 and letter in _GRADE_LETTER_PRIOR:
        prior = _GRADE_LETTER_PRIOR[letter]
        scored.update({
            "gait_score": prior,
            "hindquarter_strength": prior,
            "vascularity": prior,
            "mental_state": prior,
            "hits": -1,  # -1 = grade-letter fallback only
        })
    return scored


# ── score_runner integration ─────────────────────────────
# score_runner reads these keys in [-1, 1]:
#   paddock_vascularity, paddock_hindquarter, paddock_gait
#
# We map our new [0, 1] schema into those keys via `2x - 1`, and we
# fold mental_state proportionally into all three so it also affects
# the bio term without modifying train.py.
#
# Blend: 70% own dimension + 30% mental_state share

_MENTAL_SHARE = 0.30


def to_score_runner_keys(feat_01: dict) -> dict:
    """Convert 4-dim [0,1] schema → 3 score_runner keys in [-1, 1].

    Returns {'paddock_vascularity': float, 'paddock_hindquarter': float,
             'paddock_gait': float}. Empty dict if input missing.
    """
    if not feat_01:
        return {}
    g = feat_01.get("gait_score", 0.50)
    h = feat_01.get("hindquarter_strength", 0.50)
    v = feat_01.get("vascularity", 0.50)
    m = feat_01.get("mental_state", 0.50)

    def blend(x):
        return (1 - _MENTAL_SHARE) * x + _MENTAL_SHARE * m

    def to_centered(x):
        return round(2.0 * x - 1.0, 3)

    return {
        "paddock_gait": to_centered(blend(g)),
        "paddock_hindquarter": to_centered(blend(h)),
        "paddock_vascularity": to_centered(blend(v)),
    }
