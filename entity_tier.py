"""Entity tier tables — breeder, owner, sire static rankings.

These tables are derived from public JRA / netkeiba statistics.
They encode *historical performance* only — no post-race leakage.

Update cadence:
  - SIRE_TIER: annually (new crop stats available March)
  - BREEDER_TIER: annually
  - OWNER_TIER: annually
  - DAMSIRE tables: annually

Tier scale (all tables):
  5 = elite (top 5 by G1 wins or win rate)
  4 = strong (top 6-15)
  3 = above average (top 16-30)
  2 = average
  1 = below average / unknown

IMPORTANT: These tiers are a PROXY for resource quality, not a
guarantee. A tier-5 breeder still produces plenty of losers.
The scoring layer must use weak coefficients (0.01–0.05).

Last updated: 2026-04 (based on 2023-2025 JRA records)
"""

# ═══════════════════════════════════════════════════════════
# Sire tier — top JRA sires by 2023-2025 earnings + G1 wins
# ═══════════════════════════════════════════════════════════

SIRE_TIER: dict[str, int] = {
    # Tier 5 — elite active sires
    "ロードカナロア":   5,
    "ディープインパクト": 5,
    "キタサンブラック":  5,
    "エピファネイア":   5,
    "ドゥラメンテ":    5,
    "モーリス":       5,
    # Tier 4 — strong
    "ハーツクライ":    4,
    "キズナ":        4,
    "ルーラーシップ":   4,
    "オルフェーヴル":   4,
    "サトノクラウン":   4,
    "ダイワメジャー":   4,
    "ヘニーヒューズ":   4,
    "ドレフォン":     4,
    "スワーヴリチャード": 4,
    "サトノダイヤモンド": 4,
    # Tier 3 — above average
    "リアルスティール":  3,
    "ミッキーアイル":   3,
    "イスラボニータ":   3,
    "シルバーステート":  3,
    "ゴールドシップ":   3,
    "マジェスティックウォリアー": 3,
    "シニスターミニスター": 3,
    "サトノアラジン":   3,
    "マインドユアビスケッツ": 3,
    "ブリックスアンドモルタル": 3,
    "コントレイル":    3,
    "シュヴァルグラン":  3,
    "レイデオロ":     3,
    "サリオス":      3,
}

# ═══════════════════════════════════════════════════════════
# Damsire (母の父) tier
# ═══════════════════════════════════════════════════════════

DAMSIRE_TIER: dict[str, int] = {
    # Tier 5
    "ディープインパクト": 5,
    "キングカメハメハ":  5,
    "ハーツクライ":    5,
    # Tier 4
    "ダイワメジャー":   4,
    "クロフネ":      4,
    "シンボリクリスエス": 4,
    "ステイゴールド":   4,
    "マンハッタンカフェ": 4,
    "フジキセキ":     4,
    "ゼンノロブロイ":   4,
    "ネオユニヴァース":  4,
    # Tier 3
    "アグネスタキオン":  3,
    "タニノギムレット":  3,
    "スペシャルウィーク": 3,
    "サンデーサイレンス": 3,
    "ブライアンズタイム": 3,
    "Storm Cat":       3,
    "Kingmambo":       3,
    "War Front":       3,
}

# ═══════════════════════════════════════════════════════════
# Breeder tier — top JRA breeders by annual winners
# ═══════════════════════════════════════════════════════════

BREEDER_TIER: dict[str, int] = {
    # Tier 5 — mega farms
    "ノーザンファーム":   5,
    "社台ファーム":     5,
    # Tier 4
    "追分ファーム":     4,
    "白老ファーム":     4,
    "ダーレー・ジャパン・ファーム": 4,
    "下河辺牧場":      4,
    "千代田牧場":      4,
    # Tier 3
    "ヤナガワ牧場":     3,
    "木村牧場":       3,
    "ケイアイファーム":   3,
    "大栄牧場":       3,
    "村田牧場":       3,
    "三嶋牧場":       3,
    "グランド牧場":     3,
    "杵臼牧場":       3,
    "岡田スタッド":     3,
}

# ═══════════════════════════════════════════════════════════
# Owner tier — top JRA owners by G1 wins (2020-2025)
# ═══════════════════════════════════════════════════════════

OWNER_TIER: dict[str, int] = {
    # Tier 5
    "サンデーレーシング":   5,
    "シルクレーシング":    5,
    "キャロットファーム":   5,
    "社台レースホース":    5,
    "(有)社台レースホース": 5,
    "ゴドルフィン":      5,
    # Tier 4
    "ダノックス":       4,
    "金子真人ホールディングス": 4,
    "(有)キャロットファーム": 4,
    "(株)キャロットファーム": 4,
    "G1レーシング":      4,
    "ノルマンディーオーナーズクラブ": 4,
    "ラ・メール":       4,
    "DMMドリームクラブ":   4,
    "(株)ウイン":       4,
    "吉田照哉":        4,
    "近藤利一":        4,
    "前田幸治":        4,
    # Tier 3
    "ロードホースクラブ":   3,
    "グリーンファーム":    3,
    "東京ホースレーシング":  3,
    "大樹ファーム":      3,
}

# ═══════════════════════════════════════════════════════════
# Sire distance bias — per-sire sweet-spot distance range (meters)
#
# Format: (min_distance, peak_distance, max_distance)
# Interpretation:
#   - race distance near peak → high fitness
#   - outside [min, max] → low fitness
#   - unknown sires → neutral (1200, 1800, 2400)
# ═══════════════════════════════════════════════════════════

SIRE_DISTANCE_PROFILE: dict[str, tuple[int, int, int]] = {
    # Sprint-mile
    "ロードカナロア":   (1000, 1400, 1800),
    "ダイワメジャー":   (1200, 1600, 2000),
    "ミッキーアイル":   (1000, 1400, 1800),
    "ヘニーヒューズ":   (1000, 1400, 1800),
    # Mile-middle
    "モーリス":       (1400, 1800, 2200),
    "ドレフォン":     (1200, 1600, 2000),
    "エピファネイア":   (1600, 2000, 2500),
    "ドゥラメンテ":    (1600, 2000, 2400),
    "キタサンブラック":  (1800, 2200, 3200),
    "コントレイル":    (1600, 2000, 2400),
    # Classic-stayer
    "ディープインパクト": (1600, 2000, 2500),
    "ハーツクライ":    (1800, 2200, 3200),
    "キズナ":        (1800, 2200, 2600),
    "ルーラーシップ":   (1600, 2000, 2400),
    "オルフェーヴル":   (1800, 2200, 3000),
    "ゴールドシップ":   (1800, 2400, 3200),
    "ステイゴールド":   (1800, 2200, 3200),
    # Dirt specialist
    "シニスターミニスター": (1200, 1800, 2100),
    "マジェスティックウォリアー": (1200, 1600, 2000),
    "マインドユアビスケッツ": (1200, 1600, 2000),
}

# Default profile for unknown sires
DEFAULT_DISTANCE_PROFILE = (1200, 1800, 2400)

# ═══════════════════════════════════════════════════════════
# Sire surface bias — turf vs dirt affinity
#
# Format: {"turf": float, "dirt": float}
#   Values in [0, 1]. 1.0 = strong affinity.
#   Unknown sires → {"turf": 0.5, "dirt": 0.5}
# ═══════════════════════════════════════════════════════════

SIRE_SURFACE_BIAS: dict[str, dict[str, float]] = {
    # Turf dominant
    "ディープインパクト": {"turf": 0.95, "dirt": 0.20},
    "ハーツクライ":    {"turf": 0.90, "dirt": 0.25},
    "キタサンブラック":  {"turf": 0.85, "dirt": 0.30},
    "エピファネイア":   {"turf": 0.85, "dirt": 0.30},
    "ドゥラメンテ":    {"turf": 0.80, "dirt": 0.40},
    "モーリス":       {"turf": 0.80, "dirt": 0.35},
    "ロードカナロア":   {"turf": 0.75, "dirt": 0.50},
    "キズナ":        {"turf": 0.85, "dirt": 0.30},
    "オルフェーヴル":   {"turf": 0.80, "dirt": 0.35},
    "ゴールドシップ":   {"turf": 0.80, "dirt": 0.25},
    "ルーラーシップ":   {"turf": 0.80, "dirt": 0.35},
    "コントレイル":    {"turf": 0.85, "dirt": 0.30},
    "ダイワメジャー":   {"turf": 0.75, "dirt": 0.40},
    "ミッキーアイル":   {"turf": 0.80, "dirt": 0.30},
    # Dual-surface
    "ドレフォン":     {"turf": 0.55, "dirt": 0.65},
    "サトノクラウン":   {"turf": 0.65, "dirt": 0.50},
    "スワーヴリチャード": {"turf": 0.65, "dirt": 0.55},
    # Dirt dominant
    "ヘニーヒューズ":   {"turf": 0.15, "dirt": 0.90},
    "シニスターミニスター": {"turf": 0.10, "dirt": 0.90},
    "マジェスティックウォリアー": {"turf": 0.15, "dirt": 0.85},
    "マインドユアビスケッツ": {"turf": 0.20, "dirt": 0.85},
}

DEFAULT_SURFACE_BIAS = {"turf": 0.50, "dirt": 0.50}

# ═══════════════════════════════════════════════════════════
# Sire heavy-track bias — performance on 重/不良
#
# Scale: 0.0 (bad on heavy) to 1.0 (excels on heavy)
# ═══════════════════════════════════════════════════════════

SIRE_HEAVY_TRACK_BIAS: dict[str, float] = {
    "ゴールドシップ":   0.85,
    "ステイゴールド":   0.80,
    "ハーツクライ":    0.70,
    "オルフェーヴル":   0.75,
    "キタサンブラック":  0.65,
    "エピファネイア":   0.55,
    "ディープインパクト": 0.40,
    "ロードカナロア":   0.45,
    "ヘニーヒューズ":   0.70,
    "シニスターミニスター": 0.75,
}

DEFAULT_HEAVY_TRACK_BIAS = 0.50


# ═══════════════════════════════════════════════════════════
# Lookup functions with safe fallback
# ═══════════════════════════════════════════════════════════

def get_sire_tier(name: str) -> int:
    """Return sire tier (1-5). Unknown → 2."""
    return SIRE_TIER.get((name or "").strip(), 2)


def get_damsire_tier(name: str) -> int:
    """Return damsire tier (1-5). Unknown → 2."""
    return DAMSIRE_TIER.get((name or "").strip(), 2)


def get_breeder_tier(name: str) -> int:
    """Return breeder tier (1-5). Unknown → 2."""
    return BREEDER_TIER.get((name or "").strip(), 2)


def get_owner_tier(name: str) -> int:
    """Return owner tier (1-5). Unknown → 2."""
    return OWNER_TIER.get((name or "").strip(), 2)


def get_sire_distance_profile(name: str) -> tuple[int, int, int]:
    """Return (min, peak, max) distance in meters. Unknown → default."""
    return SIRE_DISTANCE_PROFILE.get((name or "").strip(),
                                     DEFAULT_DISTANCE_PROFILE)


def get_sire_surface_bias(name: str) -> dict[str, float]:
    """Return {"turf": float, "dirt": float}. Unknown → neutral."""
    return SIRE_SURFACE_BIAS.get((name or "").strip(),
                                 DEFAULT_SURFACE_BIAS)


def get_sire_heavy_track_bias(name: str) -> float:
    """Return heavy-track affinity 0-1. Unknown → 0.5."""
    return SIRE_HEAVY_TRACK_BIAS.get((name or "").strip(),
                                     DEFAULT_HEAVY_TRACK_BIAS)
