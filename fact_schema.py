"""Canonical fact schema — objective, observational, opinion-free.

Each fact is a small struct with:
    type:       one of CANONICAL_FACT_TYPES
    horse:      horse name, or None for race-level facts
    polarity:   -1 (negative), 0 (neutral), +1 (positive)
    confidence: raw confidence in [0, 1] before consensus adjustment
    source:     "jra" | "keibalab" | "news" | "netkeiba" | "openmeteo"
    raw_text:   the literal substring that triggered the extraction
    meta:       extra numeric details (weight kg, cushion value, etc.)

Facts are DESIGNED to exclude predictions, 印, buy-lists, and popularity
opinions. The extractor ships a blacklist (`OPINION_PHRASES`) that
actively drops candidate facts when any of those phrases appears in the
surrounding sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ── Canonical fact types ───────────────────────────────
# Each type has a polarity and a baseline per-source confidence.
# The extractor looks for phrases in POSITIVE_PHRASES / NEGATIVE_PHRASES
# inside the free text. Longest-match-first to avoid partial collisions.

CANONICAL_FACT_TYPES: dict[str, dict] = {
    # ── Appearance / coat / body ──
    "coat_good": {
        "polarity": +1, "category": "condition",
        "phrases": [
            # strong
            ("毛ヅヤが素晴らし", 0.90), ("毛艶が素晴らし", 0.90),
            ("毛艶が冴え", 0.88), ("毛色に光沢", 0.85), ("ピカピカ", 0.82),
            ("輝きがあり", 0.82), ("毛ヅヤ申し分な", 0.85), ("張り艶申し分な", 0.88),
            # medium
            ("毛ヅヤ良", 0.75), ("毛艶良", 0.75), ("張り艶良", 0.78),
            ("毛色きれい", 0.75), ("毛並み良", 0.72), ("毛艶いい", 0.72),
            ("毛ヅヤあり", 0.72), ("ツヤあり", 0.70), ("光沢あり", 0.72),
            ("つやのある毛", 0.72), ("ハリ艶", 0.70), ("きれいな毛", 0.70),
            # observational / subtle
            ("皮膚が薄", 0.70), ("血管が浮", 0.72), ("血管の浮き", 0.72),
            ("薄皮", 0.65), ("きめ細かな肌", 0.72), ("肌艶良", 0.72),
            ("皮膚の張り", 0.70), ("毛色鮮やか", 0.72),
            # Real phrases seen in netkeiba data
            ("馬体に張り", 0.72), ("馬体の張り", 0.72), ("体に張り", 0.72),
            ("張りあって", 0.72), ("張り良", 0.70),
        ],
    },
    "coat_bad": {
        "polarity": -1, "category": "condition",
        "phrases": [
            ("毛ヅヤがくすん", 0.72), ("毛艶がくすん", 0.72), ("毛ヅヤやや劣", 0.58),
            ("毛ヅヤ悪", 0.82), ("毛艶悪", 0.82), ("くすんだ毛", 0.68),
            ("毛艶に乏し", 0.72), ("毛色に艶がな", 0.72), ("張りがな", 0.65),
            ("皮膚にくすみ", 0.65),
        ],
    },
    "body_sharp": {
        "polarity": +1, "category": "condition",
        "phrases": [
            # very strong
            ("引き締まった馬体", 0.88), ("シャープな馬体", 0.85),
            ("仕上がり抜群", 0.92), ("仕上がり申し分な", 0.88),
            # strong / medium
            ("馬体もシャープ", 0.82), ("シャープな造り", 0.80),
            ("シャープな体つき", 0.80), ("仕上がり良", 0.78), ("仕上がった造り", 0.75),
            ("仕上がっている", 0.72), ("スッキリした造り", 0.75),
            ("スッキリとした", 0.72), ("スリムな体", 0.72), ("スリムな馬体", 0.75),
            ("スリムな造り", 0.72), ("締まっ", 0.60), ("引き締まり", 0.75),
            ("無駄のない体", 0.78), ("ひきしまっ", 0.75),
            # subtle / observational
            ("馬体ＯＫ", 0.68), ("造りもＯＫ", 0.68), ("造りＯＫ", 0.68),
            ("体つきがしっかり", 0.72), ("輪郭もスッキリ", 0.72),
            ("体つき良", 0.70), ("造り良", 0.70), ("造りしっかり", 0.72),
            ("前走よりシャープ", 0.78), ("シャープさ", 0.72),
            # from actual data seen
            ("体つきＯＫ", 0.70), ("仕上がった体", 0.72),
            # oikiri editorial: strength / presence
            ("迫力十分", 0.80), ("迫力満点", 0.85), ("力強い", 0.72),
        ],
    },
    "body_heavy": {
        "polarity": -1, "category": "condition",
        "phrases": [
            # fullness / slack — only unambiguously negative forms
            ("太め残り", 0.88), ("太め残", 0.85),
            ("太め気味", 0.78), ("やや太め", 0.72),
            ("ぼてっと", 0.80), ("腹回りが張", 0.72), ("腹が出", 0.68),
            ("重たい馬体", 0.82), ("絞り切れ", 0.75),
            ("体が絞りきれ", 0.80), ("体が緩", 0.72), ("重苦しい体", 0.85),
            ("体に余裕あり過ぎ", 0.78),
            # FATIGUE / heaviness expansion
            ("疲労感", 0.85), ("疲労が残", 0.85), ("疲れが残", 0.82),
            ("疲れ気味", 0.75), ("影響が残", 0.68), ("回復途上", 0.65),
            ("張り不足", 0.75), ("張りが不足", 0.75),
            ("動きが重", 0.82), ("動きも重", 0.82), ("もたもた", 0.75),
            ("もっさり", 0.78), ("ダルさ", 0.72), ("ダルそう", 0.72),
            ("重そう", 0.68), ("重い動き", 0.80), ("重たい動き", 0.82),
            ("歩みが重", 0.80), ("足取りが重", 0.80),
            ("反応が鈍", 0.75), ("反応鈍", 0.72),
            ("活気がな", 0.78), ("元気がな", 0.75), ("覇気がな", 0.78),
        ],
    },
    # ── Hindquarter / power ──
    "hindquarter_strong": {
        "polarity": +1, "category": "physique",
        "phrases": [
            ("トモの張りが素晴らし", 0.90), ("トモの筋肉が充実", 0.88),
            ("後躯が発達", 0.85), ("パンプアップ", 0.85),
            ("力強いトモ", 0.80), ("トモしっかり", 0.75),
            ("トモの筋肉", 0.72), ("筋肉の張り", 0.75), ("トモ良", 0.65),
            ("後肢しっかり", 0.75), ("後肢に力", 0.75),
            ("トモの踏み込み", 0.78), ("トモの踏み込み良好", 0.82),
            ("腰の回転が良", 0.78), ("腰の入り", 0.72), ("腰しっかり", 0.72),
            ("推進力ありそう", 0.72), ("推進力", 0.65),
            ("後ろ脚に力", 0.72), ("四肢の運びに力強さ", 0.82),
            ("四肢の運び", 0.65), ("筋肉質", 0.62),
            ("トモの筋肉が張", 0.82),
        ],
    },
    "hindquarter_weak": {
        "polarity": -1, "category": "physique",
        "phrases": [
            ("トモがやや甘", 0.65), ("トモ甘", 0.72), ("トモが甘", 0.70),
            ("トモが薄", 0.72), ("後肢が細", 0.70), ("腰が甘", 0.70),
            ("ヒップが小", 0.65), ("推進力不足", 0.82), ("後ろ脚細", 0.65),
            ("後躯発達途上", 0.58),
        ],
    },
    # ── Gait / movement ──
    "gait_good": {
        "polarity": +1, "category": "movement",
        "phrases": [
            # strong
            ("伸びのある歩様", 0.85), ("軽快なフットワーク", 0.85),
            ("力強い踏み込み", 0.85), ("歩様が滑らか", 0.82),
            ("しなやかな動き", 0.82), ("弾むような", 0.80),
            # medium
            ("歩様良好", 0.78), ("歩様いい", 0.70), ("歩様柔軟", 0.78),
            ("踏み込み深", 0.75), ("踏み込み良好", 0.78),
            ("歩幅広", 0.72), ("ストライド", 0.65),
            ("首の使い良", 0.72), ("首の動き", 0.62),
            ("軽快", 0.62), ("リズム良", 0.70), ("リズム良く", 0.72),
            # subtle
            ("脚どり軽", 0.72), ("脚どり軽快", 0.78), ("脚さばき良", 0.72),
            ("身のこなし", 0.70), ("身のこなしがなめらか", 0.82),
            ("身のこなしもなめらか", 0.80), ("柔軟な身のこなし", 0.82),
            ("歩様もしっかり", 0.72), ("スムーズ", 0.60),
            ("柔軟", 0.60), ("フットワーク軽", 0.75),
            # oikiri editorial
            ("動き抜群", 0.88), ("末脚良し", 0.78), ("末脚鋭", 0.82),
            ("キビキビ", 0.72), ("素軽い", 0.72), ("素軽", 0.68),
            ("反応上々", 0.75), ("鋭く伸", 0.82), ("動き軽快", 0.80),
        ],
    },
    "gait_bad": {
        "polarity": -1, "category": "movement",
        "phrases": [
            ("歩様が硬", 0.82), ("歩様硬", 0.80), ("歩様が固", 0.80),
            ("歩様固", 0.78), ("歩様やや固", 0.68), ("歩様やや硬", 0.68),
            ("踏み込み浅", 0.72), ("踏み込みが浅", 0.72),
            ("脚どり重", 0.78), ("脚運び硬", 0.78), ("ぎこちない", 0.78),
            ("歩幅が狭", 0.65), ("フットワーク硬", 0.75),
            ("首が硬", 0.72), ("首を気にし", 0.72),
            ("硬さが残", 0.80), ("硬さあり", 0.75),
            ("硬さが目立", 0.82), ("硬さを感じ", 0.75),
            ("ガタガタ", 0.82),
            # PAIN / STIFFNESS expansion
            ("動きが硬", 0.80), ("動きに硬さ", 0.82),
            ("動きの硬さ", 0.82), ("動きが固", 0.78),
            ("しなやかさに欠", 0.72), ("伸び欠", 0.72),
            ("跛行", 0.92), ("引きず", 0.82), ("痛そう", 0.85),
            ("脚を気にし", 0.82), ("足を気にし", 0.82),
            ("かばう", 0.78), ("かばい", 0.78),
            ("フラつ", 0.72), ("ぐらつ", 0.70),
            ("足取りが悪", 0.78),
        ],
    },
    # ── Mental state ──
    "mental_calm": {
        "polarity": +1, "category": "mental",
        "phrases": [
            # strong
            ("落ち着き払って", 0.88), ("冷静そのもの", 0.88),
            ("集中力の高さ", 0.85),
            # medium
            ("落ち着きがあり", 0.78), ("落ち着き", 0.70), ("精神的に安定", 0.78),
            ("リラックス", 0.75), ("堂々とした", 0.75), ("堂々と", 0.68),
            ("気合十分", 0.75), ("気合が乗", 0.80), ("目力あり", 0.78),
            ("集中でき", 0.72), ("穏やか", 0.62),
            # subtle / common
            ("気配はとてもいい", 0.78), ("気配良", 0.70),
            ("気配いい", 0.68), ("活気あり", 0.72),
            ("活気のある", 0.72), ("張り切っ", 0.65),
            ("テンション良", 0.68), ("状態キープ", 0.62),
            ("いい顔", 0.62),
            # oikiri
            ("好気合", 0.78),
        ],
    },
    "mental_tense": {
        "polarity": -1, "category": "mental",
        "phrases": [
            # core — only unambiguously tense forms
            ("イレ込み気味", 0.85), ("イレ込み", 0.82), ("入れ込み", 0.82),
            ("入れ込ん", 0.80), ("入れ込む", 0.78),
            ("チャカチャカ", 0.80), ("チャカついて", 0.78),
            ("カリカリ", 0.80), ("カリカリして", 0.85),
            ("落ち着きなく", 0.82), ("落ち着きがな", 0.82),
            ("テンション高い", 0.78), ("テンションが高", 0.78),
            ("気負いが", 0.70), ("神経質", 0.75),
            ("集中しきれ", 0.68),
            # STRESS expansion
            ("気性難", 0.82), ("気性激", 0.80), ("気性が激", 0.78),
            ("首を振", 0.72), ("首振", 0.70), ("頭を上下", 0.70),
            ("頭が高", 0.65), ("ウロウロ", 0.75),
            ("耳を伏せ", 0.72), ("尻尾を振", 0.68), ("尾を振", 0.65),
            ("騒がしい", 0.72), ("騒々しい", 0.72), ("煩そう", 0.68),
            ("噛みつ", 0.78), ("周回が速", 0.62),
            ("テンション上", 0.72), ("テンションが上", 0.72),
            ("そわそわ", 0.70), ("ソワソワ", 0.70),
        ],
    },
    "sweating_concern": {
        "polarity": -1, "category": "condition",
        "phrases": [
            # core
            ("発汗", 0.72), ("汗を流", 0.78), ("汗をかい", 0.75),
            ("首筋に汗", 0.85), ("大量の汗", 0.92),
            ("ボタボタ", 0.85), ("汗びっしょり", 0.88), ("汗多", 0.75),
            # SWEATING expansion
            ("汗が", 0.68), ("汗染み", 0.72), ("発汗多", 0.80),
            ("汗だく", 0.82), ("汗が光", 0.72),
            ("首に汗", 0.80), ("脇に汗", 0.78),
        ],
    },
    # ── Suitability ──
    "heavy_track_fit": {
        "polarity": +1, "category": "suitability",
        "phrases": [
            ("道悪巧者", 0.88), ("重馬場巧者", 0.88),
            ("不良馬場で好走", 0.82),
        ],
    },
    "distance_fit_mile": {
        "polarity": +1, "category": "suitability",
        "phrases": [
            ("マイル戦向き", 0.78), ("1600m巧者", 0.82),
            ("マイルがベスト", 0.80),
        ],
    },
    # ── Weight change (JRA NUMERIC ONLY) ────────────────
    # These fact types are populated exclusively from the JRA weight-delta
    # pipeline (`fact_from_weight_delta`). Text phrases like "馬体増" are
    # intentionally EMPTY because they collocate with both positive and
    # negative framings ("馬体増でも体つきＯＫ" is positive, "馬体増で動きが重" is
    # negative). Trusting numeric deltas avoids sentiment contamination.
    "weight_up_large": {
        "polarity": -1, "category": "condition",
        "phrases": [],
    },
    "weight_down_large": {
        "polarity": -1, "category": "condition",
        "phrases": [],
    },
    "good_weight_stable": {
        "polarity": +1, "category": "condition",
        "phrases": [],
    },
    # ── Stable / trainer comments ──
    "stable_positive_comment": {
        "polarity": +1, "category": "stable",
        "phrases": [
            ("状態良好", 0.78), ("順調", 0.72), ("好調", 0.78),
            ("態勢整", 0.72), ("態勢は整", 0.75), ("万全", 0.85),
            ("絶好調", 0.92), ("仕上がり万全", 0.88),
            ("調整順調", 0.75), ("調子上", 0.68),
            ("文句なし", 0.85), ("期待以上", 0.62),
            # netkeiba oikiri editorial phrases (2026-04 seen)
            ("好調持続", 0.80), ("出来は良", 0.75), ("出来良", 0.72),
            ("出来安定", 0.75), ("元気一杯", 0.78),
            ("仕上良好", 0.82), ("仕上上々", 0.82), ("仕上十分", 0.80),
            ("乗込十分", 0.78), ("乗込入念", 0.72),
            ("気配上々", 0.75), ("気配上昇", 0.78), ("気配抜群", 0.85),
            ("好気配", 0.75), ("気配充実", 0.75),
            ("更に上昇", 0.78), ("上積十分", 0.80), ("上昇気配", 0.75),
            ("一歩前進", 0.60), ("前走以上", 0.65),
            ("攻め熱心", 0.68), ("多少良化", 0.60), ("良化", 0.58),
            ("意欲十分", 0.72),
        ],
    },
    "stable_concern_comment": {
        "polarity": -1, "category": "stable",
        "phrases": [
            ("状態に疑問", 0.82), ("物足りない", 0.70), ("物足りず", 0.75),
            ("不安", 0.65), ("上昇途上", 0.55), ("状態上向き", 0.55),
            ("完調一歩手前", 0.65), ("ひと叩き", 0.62),
            # negative stable quotes
            ("本来の姿ではな", 0.78), ("本来ほどでは", 0.72),
        ],
    },
}


# ── Fuzzy phrase clusters ──
# Phrases within the same cluster key are treated as corroborating each
# other at consensus time, even if they are different canonical types.
# Used by merge_fact_layers to let "毛艶冴え" (coat_good) and "張り艶冴え"
# (body_sharp) cross-corroborate as "coat/shine quality" observations.
#
# Each cluster key maps to a set of (substring, weight_multiplier) where
# weight_multiplier stays at 1.0 by default. The extractor's natural
# substring matching handles most cases; clusters are an extra layer
# applied at merge time to forgive small wording differences.

FUZZY_CLUSTERS: dict[str, tuple[str, ...]] = {
    # Narrow clusters — each phrase lives in EXACTLY ONE cluster.
    # Ordered so the broader cluster_any_positive_physical is matched
    # last (at phrase_to_cluster time, longest substring wins, so broader
    # clusters need shorter keys).

    # BROAD: any positive physical observation. This is the key consensus
    # widener — a horse called "動き抜群" from one writer and "仕上がり良"
    # from another writer, landing in different canonical types, both
    # map here and corroborate at the cluster level.
    "cluster_any_positive_physical": (
        # coat / shine
        "毛ヅヤ", "毛艶", "張り艶", "ハリ艶", "色艶", "ツヤ",
        "光沢", "輝き", "ピカピカ", "艶やか",
        # body firmness
        "引き締ま", "シャープ", "スリム", "ひきしま",
        "無駄のない", "締まっ", "スッキリ",
        "仕上がっ", "仕上がり", "仕上良", "仕上上",
        "仕上十分", "出来良", "出来安定", "迫力",
        "造り良", "造りしっかり",
        # hindquarter
        "トモの張", "トモしっかり", "トモの筋肉", "パンプ",
        "四肢の運び", "踏み込み良", "踏み込み深",
        "後肢しっかり", "腰しっかり", "推進力",
        # gait / movement
        "歩様良", "歩様柔", "軽快", "身のこなし", "脚どり軽",
        "末脚良", "末脚鋭", "動き抜群", "動き良", "反応上々",
        "鋭く伸", "しなやか", "キビキビ", "素軽",
    ),

    # BROAD: any negative physical observation
    "cluster_any_negative_physical": (
        "毛ヅヤ悪", "毛艶悪", "太め", "ぼてっと",
        "歩様硬", "歩様が硬", "歩様が固", "脚どり重",
        "トモ甘", "後肢が細", "腰が甘",
        "物足りな", "物足りず",
    ),

    # Mental clusters stay separate — calm vs tense should not merge.
    "cluster_mental_positive": (
        "落ち着き", "冷静", "集中", "リラックス", "堂々",
        "気合十分", "気合が乗", "気配上", "気配良", "気配いい",
        "気配はとてもいい", "気配抜群", "活気",
        "意欲", "目力", "好気合",
    ),
    "cluster_mental_negative": (
        "イレ込", "入れ込", "チャカつ", "チャカチャカ",
        "カリカリ", "神経質", "気負", "落ち着きな", "集中しきれ",
    ),
    # Stable comments — only consensus-merged when positive AND from
    # multiple sources. "状態良" from a single quote is one source only.
    "cluster_stable_positive": (
        "状態良", "好調", "順調", "万全", "絶好", "乗込",
        "上昇", "上積", "前進", "良化", "調整", "文句な",
    ),
    "cluster_stable_concern": (
        "疑問", "不安", "本来", "手前", "一息",
    ),
}

# Reverse index: for each fuzzy substring, which cluster it belongs to
_FUZZY_SUBSTR_TO_CLUSTER: dict[str, str] = {}
for cluster_key, substrs in FUZZY_CLUSTERS.items():
    for s in substrs:
        _FUZZY_SUBSTR_TO_CLUSTER[s] = cluster_key


def phrase_to_cluster(phrase: str) -> str | None:
    """Return the fuzzy-cluster key that best matches a given phrase
    (longest matching substring), or None if no cluster matches."""
    if not phrase:
        return None
    best_key = None
    best_len = 0
    for substr, key in _FUZZY_SUBSTR_TO_CLUSTER.items():
        if substr in phrase and len(substr) > best_len:
            best_key = key
            best_len = len(substr)
    return best_key


# ── Opinion / prediction blacklist ──
# If any of these phrases appears in a candidate sentence, the extractor
# drops the fact. We never want human betting opinions in the pipeline.

OPINION_PHRASES: tuple[str, ...] = (
    "本命", "対抗", "単穴", "連下", "押さえ", "穴",
    "◎", "○", "▲", "△", "☆", "×",
    "買い目", "狙い目", "推奨", "本線", "相手",
    "負けられない", "絶対勝", "必勝", "鉄板",
    "人気", "期待", "狙える", "おすすめ",
    # Prediction verbs
    "勝つと思", "勝つだろう", "勝てる", "勝ちそう",
    "敗れる", "敗れそう",
    # Popularity / betting-frame
    "◎本命", "1番人気", "2番人気", "穴馬",
)


def sentence_is_opinion(sentence: str) -> bool:
    """True if a sentence contains any prediction/印/buy-list phrase.

    Sentences classified as opinion are dropped entirely by the
    extractor — no facts are emitted from them, even if observational
    phrases are present.
    """
    if not sentence:
        return False
    for p in OPINION_PHRASES:
        if p in sentence:
            return True
    return False


# ── Source tier → base confidence ──

SOURCE_BASE_CONFIDENCE = {
    "jra":             1.00,   # JRA official
    "netkeiba":        0.90,   # netkeiba shutuba (near-official)
    "netkeiba_oikiri": 0.80,   # netkeiba oikiri training evaluation
    "keibalab":        0.70,   # KeibaLab 馬体FOCUS, structured obs
    "hochi":           0.55,   # Sports Hochi race article
    "sanspo":          0.55,   # Sankei Sports race article
    "daily":           0.55,   # Daily Sports race article
    "news":            0.50,   # generic sports newspaper article (Yahoo-linked)
    "openmeteo":       0.85,   # weather API
}


# ── Fact struct ──

@dataclass
class Fact:
    type: str
    horse: Optional[str]
    polarity: int                  # -1, 0, +1
    confidence: float              # [0, 1]
    source: str                    # key of SOURCE_BASE_CONFIDENCE
    raw_text: str = ""
    category: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Fact":
        return cls(**d)
