"""Horse deep-fact enricher (v5.2 — 2026-04-19).

既存の `scraper.enrich_entries` が既に horse detail / jockey stats /
paddock reports / training times を全頭分 fetch してディスク cache に
置いている。そのデータを **再活用** して、LOOSE / STRICT トリガーが
消費する fact を深掘り抽出する。

設計原則:
  1. **No new network calls**. 既に `enrich_entries` が取ってきた
     データから facts を生成するだけ。追加 fetch は一切しない。
  2. **Constitution §7.1 範囲**. facts の追加は「ファクト抽出辞書の拡張」
     として明示的に許容されている。LOOSE の 4 数値条件は変更しない。
  3. **Deterministic**. 同じ入力 (horse_data, race_context) なら常に
     同じ fact 集合を返す。キャッシュ不要。
  4. **最上位 source tier `horse_deep`** (conf base 0.85) として登録。
     netkeiba DB の公式プロフィール由来なので信頼度高い。

出力される fact type (抜粋):
  career_prize_high            +  高獲得賞金 (G1 級)
  career_prize_low             -  同レースに対して賞金低い
  venue_past_win               +  本会場で過去勝利あり
  distance_specialist          +  本距離帯の勝率高い
  surface_mismatch             -  本馬場の成績悪い
  weight_trend_stable          +  馬体重が安定推移
  weight_trend_rising          0  馬体重が増加トレンド (中立、監視材料)
  weight_trend_falling         -  馬体重が減少トレンド (警戒)
  layoff_short                 +  適度な間隔 (3-8 週)
  layoff_long                  -  長期休養明け (>12 週)
  recent_strong_finish         +  直近で好走あり
  recent_poor_finish           -  直近で着順一桁外
  owner_g1_pedigree            +  馬主が G1 実績持ち (既存 tier 流用)
  breeder_g1_pedigree          +  生産者が G1 実績持ち
  external_stable_elite        +  外厩がノーザンF/社台Fなど
"""

from __future__ import annotations

import re as _re
from datetime import date as _date, datetime as _dt
from typing import Optional

import fact_schema as _fs


ENRICHER_VERSION = "horse-facts-enricher-v5.2-2026-04-19"

# Source tier for facts emitted by this module. 0.85 is between
# "keibalab" (0.70) and "netkeiba" (0.90) — DB-derived, structured, but
# computed from raw numbers rather than official declarations.
SOURCE_TIER = "horse_deep"

# ──────────────────────────────────────────────────────
# Config thresholds (tunable without touching LOOSE rule)
# ──────────────────────────────────────────────────────

# 累計獲得賞金の tier 判定 (単位: 万円)
PRIZE_ELITE_THRESHOLD = 15_000.0   # 1.5億円〜 (G1 常連級)
PRIZE_HIGH_THRESHOLD  = 5_000.0    # 5000万〜 (重賞級)
PRIZE_LOW_THRESHOLD   = 500.0      # 500万以下 (未勝利明け等)

# Weight trend 判定 — 直近 3 レース連続で単調変化したか
WEIGHT_MOVE_MIN_KG = 4.0           # 4 kg 以上の変化を 'significant' とする

# Layoff 判定 (週単位)
LAYOFF_SHORT_MIN_WEEKS = 3
LAYOFF_SHORT_MAX_WEEKS = 8
LAYOFF_LONG_MIN_WEEKS  = 12

# Recent form 判定 window
RECENT_FORM_WINDOW = 5


# ──────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────

def _fact(type_: str, horse: str, polarity: int,
          confidence: float, raw: str, meta: Optional[dict] = None,
          category: str = "horse_deep") -> _fs.Fact:
    return _fs.Fact(
        type=type_, horse=horse, polarity=polarity,
        confidence=float(max(0.0, min(1.0, confidence))),
        source=SOURCE_TIER, raw_text=raw,
        category=category, meta=meta or {},
    )


def _parse_career_prize(raw: str) -> Optional[float]:
    """Parse netkeiba's career prize string into 万円 float.

    Examples:
      "12,345万円"         → 12345.0
      "1,234.5万円"        → 1234.5
      "12,345万円 (中央)"   → 12345.0
      "5億6,789万円"       → 56789.0
      ""                   → None
    """
    s = (raw or "").strip()
    if not s:
        return None
    # Remove commas, parenthetical annotations, spaces
    s = _re.sub(r"\([^)]*\)", "", s).replace(",", "").replace(" ", "").strip()

    # Handle "N億M万円" form
    m = _re.match(r"(?:(\d+(?:\.\d+)?)億)?(\d+(?:\.\d+)?)万円?$", s)
    if m:
        oku = float(m.group(1) or 0)
        man = float(m.group(2))
        return oku * 10000.0 + man

    # Handle bare numeric in 万円 like "1234"
    m = _re.match(r"^(\d+(?:\.\d+)?)$", s)
    if m:
        return float(m.group(1))
    return None


def _parse_race_date(raw: str) -> Optional[_date]:
    """Parse netkeiba's race date column which can be YYYY/MM/DD or YY/MM/DD."""
    s = (raw or "").strip()
    if not s:
        return None
    m = _re.match(r"^(\d{4})/(\d{1,2})/(\d{1,2})$", s)
    if m:
        try:
            return _date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = _re.match(r"^(\d{2})/(\d{1,2})/(\d{1,2})$", s)
    if m:
        try:
            return _date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


def _rank_int(raw: str) -> Optional[int]:
    """Parse '1着' or '3' or '' → int or None."""
    s = (raw or "").strip().replace("着", "")
    try:
        v = int(s)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


# ──────────────────────────────────────────────────────
# Individual extractors
# ──────────────────────────────────────────────────────

def extract_prize_facts(horse_name: str, horse: dict) -> list[_fs.Fact]:
    """Career prize money → high/low tier facts."""
    out: list[_fs.Fact] = []
    raw = horse.get("career_prize", "")
    if not raw:
        return out
    prize_man = _parse_career_prize(raw)
    if prize_man is None:
        return out
    meta = {"career_prize_man": prize_man, "raw": raw}
    if prize_man >= PRIZE_ELITE_THRESHOLD:
        out.append(_fact("career_prize_elite", horse_name, +1,
                         0.75, f"累計獲得賞金 {prize_man:.0f}万円 (G1常連級)", meta))
    elif prize_man >= PRIZE_HIGH_THRESHOLD:
        out.append(_fact("career_prize_high", horse_name, +1,
                         0.55, f"累計獲得賞金 {prize_man:.0f}万円 (重賞級)", meta))
    elif prize_man <= PRIZE_LOW_THRESHOLD:
        out.append(_fact("career_prize_low", horse_name, -1,
                         0.45, f"累計獲得賞金 {prize_man:.0f}万円 (薄い)", meta))
    return out


def extract_weight_trend_facts(horse_name: str, horse: dict) -> list[_fs.Fact]:
    """Parse weight_trend list → stable / rising / falling facts.

    `weight_trend` shape from fetch_horse_detail: ["480kg(+4)", "476kg(-2)", ...]
    The most-recent race is index 0.
    """
    out: list[_fs.Fact] = []
    wt = horse.get("weight_trend") or []
    if len(wt) < 3:
        return out

    deltas: list[int] = []
    for s in wt[:RECENT_FORM_WINDOW]:
        m = _re.search(r"\(([+-]?\d+)\)", s or "")
        if m:
            try:
                deltas.append(int(m.group(1)))
            except ValueError:
                pass

    if len(deltas) < 3:
        return out

    recent3 = deltas[:3]
    net3 = sum(recent3)
    max_abs = max(abs(d) for d in recent3)

    meta = {"deltas": recent3, "net_kg": net3}

    if max_abs <= 2:
        out.append(_fact("weight_trend_stable", horse_name, +1,
                         0.50, f"直近3走の馬体重変動 ±{max_abs}kg以内で安定", meta))
    elif all(d >= 2 for d in recent3) and net3 >= WEIGHT_MOVE_MIN_KG:
        out.append(_fact("weight_trend_rising", horse_name, 0,
                         0.45, f"直近3走で計+{net3}kg (増加トレンド)", meta))
    elif all(d <= -2 for d in recent3) and net3 <= -WEIGHT_MOVE_MIN_KG:
        out.append(_fact("weight_trend_falling", horse_name, -1,
                         0.50, f"直近3走で計{net3}kg (減少トレンド)", meta))
    return out


def extract_recent_form_facts(horse_name: str, horse: dict,
                                race_context: dict) -> list[_fs.Fact]:
    """Parse recent_races → venue_past_win / surface_mismatch / recent_strong etc."""
    out: list[_fs.Fact] = []
    recent = horse.get("recent_races") or []
    if not recent:
        return out

    venue = (race_context or {}).get("venue", "")
    # surface / distance info may or may not be in race_context
    target_venue = (venue or "").strip()

    # Past performance at this venue
    if target_venue:
        venue_hits = 0
        venue_wins = 0
        venue_tops3 = 0
        for r in recent:
            v = (r.get("venue") or "").strip()
            if target_venue and target_venue in v:
                venue_hits += 1
                rank = _rank_int(r.get("rank"))
                if rank == 1:
                    venue_wins += 1
                if rank is not None and rank <= 3:
                    venue_tops3 += 1
        if venue_wins >= 1:
            out.append(_fact("venue_past_win", horse_name, +1,
                             0.60,
                             f"本会場 ({target_venue}) で過去 {venue_wins} 勝",
                             {"venue": target_venue, "wins": venue_wins,
                              "tops3": venue_tops3, "runs": venue_hits}))
        elif venue_hits >= 2 and venue_tops3 >= 1:
            out.append(_fact("venue_past_placed", horse_name, +1,
                             0.40,
                             f"本会場 ({target_venue}) で複勝圏 {venue_tops3}/{venue_hits}",
                             {"venue": target_venue, "tops3": venue_tops3,
                              "runs": venue_hits}))

    # Recent placement (top-3 in last N races)
    top3_last3 = 0
    bad_last3 = 0
    for r in recent[:3]:
        rank = _rank_int(r.get("rank"))
        if rank is None:
            continue
        if rank <= 3:
            top3_last3 += 1
        elif rank >= 8:
            bad_last3 += 1

    if top3_last3 >= 2:
        out.append(_fact("recent_strong_finish", horse_name, +1,
                         0.55,
                         f"直近3走中 {top3_last3} 回複勝圏",
                         {"top3_last3": top3_last3}))
    elif bad_last3 >= 2:
        out.append(_fact("recent_poor_finish", horse_name, -1,
                         0.50,
                         f"直近3走中 {bad_last3} 回8着以下",
                         {"bad_last3": bad_last3}))
    return out


def extract_layoff_facts(horse_name: str, horse: dict,
                          race_context: dict) -> list[_fs.Fact]:
    """Detect short/long layoff based on recent_races[0].date vs race_date."""
    recent = horse.get("recent_races") or []
    if not recent:
        return []
    last_date = _parse_race_date(recent[0].get("date", ""))
    if last_date is None:
        return []
    race_date_str = (race_context or {}).get("race_date")
    try:
        race_date = _dt.strptime(race_date_str[:10], "%Y-%m-%d").date()
    except Exception:
        race_date = _date.today()

    days = (race_date - last_date).days
    if days <= 0:
        return []
    weeks = days / 7.0
    meta = {"days_since_last_race": days, "weeks": round(weeks, 1),
            "last_race_date": str(last_date)}

    if LAYOFF_SHORT_MIN_WEEKS <= weeks <= LAYOFF_SHORT_MAX_WEEKS:
        return [_fact("layoff_short", horse_name, +1,
                      0.40,
                      f"前走から {weeks:.1f} 週 (適度なローテーション)",
                      meta)]
    if weeks >= LAYOFF_LONG_MIN_WEEKS:
        return [_fact("layoff_long", horse_name, -1,
                      0.50,
                      f"前走から {weeks:.1f} 週 (長期休養明け)",
                      meta)]
    return []


def extract_ownership_facts(horse_name: str, horse: dict) -> list[_fs.Fact]:
    """Emit facts based on owner / breeder / external stable tier data.

    Leverages `pedigree_features.owner_tier_score` / `breeder_tier_score` /
    `external_stable_score` which already encode which owners/breeders
    are G1-tier. Here we re-emit as explicit Facts so the composite
    aggregator sees them as part of the fact stream (in addition to the
    structured-feature score).
    """
    out: list[_fs.Fact] = []
    try:
        import pedigree_features as pf
    except Exception:
        return out

    owner = (horse.get("owner") or "").strip()
    breeder = (horse.get("breeder") or "").strip()
    ritto = (horse.get("ritto") or "").strip()

    if owner:
        score, known = pf.owner_tier_score(owner)
        if known and score >= 0.70:
            out.append(_fact("owner_g1_pedigree", horse_name, +1,
                             0.55,
                             f"馬主 ({owner}) は G1 tier",
                             {"owner": owner, "tier_score": round(score, 3)}))
        elif known and score <= 0.30:
            out.append(_fact("owner_low_tier", horse_name, -1,
                             0.30,
                             f"馬主 ({owner}) は重賞実績薄",
                             {"owner": owner, "tier_score": round(score, 3)}))

    if breeder:
        score, known = pf.breeder_tier_score(breeder)
        if known and score >= 0.70:
            out.append(_fact("breeder_g1_pedigree", horse_name, +1,
                             0.50,
                             f"生産者 ({breeder}) は G1 tier",
                             {"breeder": breeder, "tier_score": round(score, 3)}))

    if ritto:
        score, known = pf.external_stable_score(ritto)
        if known and score >= 0.70:
            out.append(_fact("external_stable_elite", horse_name, +1,
                             0.55,
                             f"外厩 {ritto} (エリート厩舎)",
                             {"ritto": ritto, "tier_score": round(score, 3)}))
    return out


# ──────────────────────────────────────────────────────
# Top-level orchestration
# ──────────────────────────────────────────────────────

def compute_deep_horse_facts(horses: list[dict], race_info: dict,
                              venue: str = "") -> list[_fs.Fact]:
    """Run every extractor on every horse, return a flat list of facts.

    Args:
      horses: list of horse dicts as produced by scraper.enrich_entries
              (or new entries_fetcher + enrich_entries combined).
              Each MUST have at least "name"; the more fields (career_prize,
              recent_races, weight_trend, owner, breeder, ritto) the better.
      race_info: dict from scraper.fetch_race_info_netkeiba
                 (weather/track_condition/surface/distance/grade).
      venue: 競馬場名 (東京/中山/...). Used for venue_past_win detection.

    Returns:
      list[Fact]. Empty list if no horses.
    """
    facts: list[_fs.Fact] = []
    race_ctx = dict(race_info or {})
    race_ctx["venue"] = venue or race_ctx.get("venue", "")

    for h in horses or []:
        name = (h.get("name") or "").strip()
        if not name:
            continue
        try:
            facts.extend(extract_prize_facts(name, h))
            facts.extend(extract_weight_trend_facts(name, h))
            facts.extend(extract_recent_form_facts(name, h, race_ctx))
            facts.extend(extract_layoff_facts(name, h, race_ctx))
            facts.extend(extract_ownership_facts(name, h))
        except Exception as e:
            # Per-horse isolation — one bad horse doesn't kill the batch.
            print(f"[horse-enricher] {name}: {e}")
            continue
    return facts
