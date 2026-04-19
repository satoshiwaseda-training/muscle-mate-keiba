"""出馬表取得層 (scratch rewrite, v5.0 — 2026-04-19).

従来の `scraper.fetch_entries_netkeiba` を踏襲せず、新規ロジックで shutuba
HTML を解析する。**オッズは絶対に読まない** (netkeiba の shutuba オッズ列は
JavaScript で動的に埋められるため requests 経由では取れないという事実を
踏まえた設計上の選択)。単勝オッズは `odds_fetcher.fetch_win_odds` で
別ルートから取得する。

公開 API:

  fetch_horse_entries(race_id: str, venue: str = "") -> list[HorseEntry]

  HorseEntry:
    number:       int             # 馬番 (1..18)
    waku:         int              # 枠番
    name:         str              # 馬名
    horse_id:     str              # netkeiba 馬 ID (URL の数字)
    age:          str              # "牡3" "牝3" "セ4" 等
    weight:       str              # 斤量 (例: "57.0")
    jockey:       str
    jockey_id:    str
    trainer:      str
    trainer_id:   str
    horse_weight: str              # "488(+4)" 等 (そのまま保持)
    owner:        str
    # ↓ 旧 scraper が付けていた派生フィールド (互換のため)
    sire:         str
    dam:          str

  オッズ関係のフィールドは一切持たない。下流 pipeline は
  `odds_fetcher.fetch_win_odds(race_id).by_number` を馬番 join で
  適用すること。

設計原則:

  1. **単一責務**: shutuba の "人の情報" (馬・騎手・厩舎) のみを返す。
     オッズは対象外。
  2. **class-based selector のみ**: cells[N] 位置指定を使わない。
     netkeiba の列挿入・並び変えに強い。
  3. **取得できなかったフィールドは空文字 or 0**。None は使わない。
  4. **外部依存を最小化**: requests + BeautifulSoup のみ。
     旧 scraper の retry / cache は**意図的に使わない**。
  5. **version marker**: 呼び出し元が「新実装を使ってるか」を runtime で
     確認できるよう `ENTRIES_FETCHER_VERSION` を expose。
"""

from __future__ import annotations

import re as _re
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests as _requests
from bs4 import BeautifulSoup as _BS


ENTRIES_FETCHER_VERSION = "entries-fetcher-v5.1-encoding-fix-2026-04-19"

_SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={race_id}"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
}


@dataclass
class HorseEntry:
    number:       int = 0
    waku:         int = 0
    name:         str = ""
    horse_id:     str = ""
    age:          str = ""
    weight:       str = ""       # 斤量 (str で保持、feature_store 側でパース)
    jockey:       str = ""
    jockey_id:    str = ""
    trainer:      str = ""
    trainer_id:   str = ""
    horse_weight: str = ""       # "488(+4)" 形式そのまま
    owner:        str = ""
    sire:         str = ""
    dam:          str = ""

    def to_dict(self) -> dict:
        """Return a dict compatible with legacy entry shape.

        下流 (feature_store, etc.) は dict 入力を前提にしているので、
        ここで互換のため dict に変換する。`odds` は意図的に付けない。
        """
        d = asdict(self)
        # Legacy compatibility: some consumers expect these keys even if empty.
        d.setdefault("stable", "")
        d.setdefault("ritto", "")
        d.setdefault("transport_stress", "")
        d.setdefault("recent_form", "")
        d.setdefault("bloodline", "")
        d.setdefault("weight_trend", "")
        d.setdefault("jockey_win_rate", "")
        d.setdefault("jockey_g1_wins", "")
        d.setdefault("trainer_win_rate", "")
        d.setdefault("training_eval", "")
        d.setdefault("training_physics", {"final_split": 0.0,
                                          "acceleration_rate": 0.0,
                                          "cardio_index": 0.0})
        d.setdefault("training_nlp", {})
        d.setdefault("paddock_scores", {})
        d.setdefault("best_weight_analysis", {})
        d.setdefault("transport_profile", {})
        d.setdefault("damsire", "")
        d.setdefault("breeder", "")
        # IMPORTANT: odds は付けない。この dict は odds 非保有の契約。
        return d


def _int_or_zero(s) -> int:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return 0


def _clean(s) -> str:
    return (s or "").strip() if isinstance(s, str) else ""


def _parse_entry_row(row) -> Optional[HorseEntry]:
    """Parse one <tr class="HorseList"> into a HorseEntry.

    Uses class-based selectors throughout — no positional cell indexing.
    Returns None if the row doesn't look like a horse entry.
    """
    # 馬番 — td.Umaban or Num class variants
    um_td = row.select_one("td.Umaban, td[class*='Umaban'], td.Num")
    num = _int_or_zero(um_td.get_text(strip=True)) if um_td else 0
    if num <= 0:
        return None

    waku_td = row.select_one("td[class*='Waku']")
    waku = _int_or_zero(waku_td.get_text(strip=True)) if waku_td else 0

    # 馬名 + horse_id
    name = ""
    horse_id = ""
    name_anchor = row.select_one("td.HorseInfo a, span.HorseName a, .HorseName a")
    if name_anchor is not None:
        name = _clean(name_anchor.get_text(strip=True))
        href = name_anchor.get("href", "") or ""
        m = _re.search(r"/horse/(\d+)", href)
        if m:
            horse_id = m.group(1)

    # 性齢 — typically td.Txt_C near the middle; be defensive
    age_td = row.select_one("td.Barei, td[class*='Barei'], td.BMGreen")
    age = _clean(age_td.get_text(strip=True)) if age_td else ""
    if not age:
        # fallback: search any td with 牡/牝/セ + digit
        for td in row.find_all("td"):
            t = td.get_text(strip=True)
            if _re.match(r"^[牡牝セ][0-9]+$", t):
                age = t
                break

    # 斤量 — class "Txt_C" often but shared. Try to locate "XX.X" next to age.
    weight = ""
    for td in row.find_all("td"):
        t = td.get_text(strip=True)
        if _re.match(r"^\d{2}\.\d$", t):  # 57.0 等
            weight = t
            break

    # 騎手 + jockey_id
    jockey = ""
    jockey_id = ""
    j_anchor = row.select_one("td.Jockey a, .Jockey a")
    if j_anchor is not None:
        jockey = _clean(j_anchor.get_text(strip=True))
        href = j_anchor.get("href", "") or ""
        m = _re.search(r"/jockey/(?:result/recent/)?(\w+)", href)
        if m:
            jockey_id = m.group(1)

    # 調教師 + trainer_id
    trainer = ""
    trainer_id = ""
    t_anchor = row.select_one("td.Trainer a, .Trainer a")
    if t_anchor is not None:
        trainer = _clean(t_anchor.get_text(strip=True))
        href = t_anchor.get("href", "") or ""
        m = _re.search(r"/trainer/(?:result/recent/)?(\w+)", href)
        if m:
            trainer_id = m.group(1)

    # 馬体重 — td.Weight or similar
    hw_td = row.select_one("td.Weight, td[class*='Weight']")
    horse_weight = _clean(hw_td.get_text(strip=True)) if hw_td else ""

    # 馬主 — .Owner a
    owner = ""
    own_anchor = row.select_one(".Owner a, td.Owner a")
    if own_anchor is not None:
        owner = _clean(own_anchor.get_text(strip=True))

    return HorseEntry(
        number=num, waku=waku, name=name, horse_id=horse_id,
        age=age, weight=weight,
        jockey=jockey, jockey_id=jockey_id,
        trainer=trainer, trainer_id=trainer_id,
        horse_weight=horse_weight,
        owner=owner,
    )


def fetch_horse_entries(race_id: str, venue: str = "",
                        *, timeout: int = 15,
                        session: Optional[_requests.Session] = None) -> list[dict]:
    """Fetch shutuba entries for `race_id` and return a list of legacy-compatible dicts.

    Each dict has the schema documented in HorseEntry.to_dict(). The dicts
    do NOT contain an `odds` field — **odds must be fetched separately via
    `odds_fetcher.fetch_win_odds`** and joined by 馬番.
    """
    url = _SHUTUBA_URL.format(race_id=race_id)
    sess = session or _requests.Session()
    try:
        resp = sess.get(url, headers=_BROWSER_HEADERS, timeout=timeout)
        if resp.status_code != 200 or not resp.content:
            return []
        raw_bytes = resp.content
    except Exception:
        return []

    # Encoding: netkeiba は Content-Type に charset を含めない場合があり、
    # `resp.text` を使うと requests が ISO-8859-1 で decode して馬名が
    # 文字化けする (フォルテアンジェロ事件 後の文字化け報告の原因)。
    # BeautifulSoup に生 bytes を渡し、HTML 内の <meta charset> / BOM /
    # chardet で自動検出させる — これは旧 scraper._get と同じ方式。
    # 明示フォールバック: EUC-JP (netkeiba の歴史的デフォルト) と UTF-8
    # のどちらにも対応するよう from_encoding は指定せず BS に委ねる。
    soup = _BS(raw_bytes, "html.parser")
    rows = soup.select("tr.HorseList")
    out: list[dict] = []
    for row in rows:
        entry = _parse_entry_row(row)
        if entry and entry.name and entry.number > 0:
            out.append(entry.to_dict())
    return out
