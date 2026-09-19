"""単勝オッズ取得層 (scratch rewrite, v5.0 — 2026-04-19).

憲法と推論層 (score_runner / dual_mode_scoring / probability_engine /
trigger_loose_capped) を変更せずに、**データ取得側だけ**を完全に書き直した
モジュール。旧 `scraper.fetch_odds_netkeiba` / `odds_sources.*` / 旧
pipeline 内の injection/consensus ロジックはすべて飛び越える。

設計原則:

  1. **単一入口**: `fetch_win_odds(race_id) -> WinOddsResult` のみ。
     他の関数 (複勝, 馬連, 3連単 …) は呼び出さない。呼ばれない。
  2. **欠損は None**。"0" やら "---" やらのプレースホルダーを一切使わない。
     数値が取れなかった馬は dict に入れない。
  3. **サニティを最初に強制**。1.0 ≤ odds ≤ 500.0 の範囲外は存在しないものとして
     扱い、`rejected_by_sanity` に別管理する。
  4. **原生レスポンス保持**。生の JSON body を結果に同梱し、怪しい値が出た
     時に事後解析できるようにする。
  5. **公開 API 以外を絶対に信じない**。shutuba HTML は JS-populated で
     オッズが原理的に取れないため**読まない**。

公開 API:

  fetch_win_odds(race_id: str, *, timeout=10, trace_dir=None)
    → WinOddsResult:
        status:      "result" | "not-published" | "http-error" | "parse-error" | "shape-error"
        race_id:     str
        by_number:   dict[int, float]       # 馬番 → 単勝オッズ (欠損は含まない)
        rejected:    dict[int, float]       # サニティで落ちた値の audit
        fetched_at:  ISO datetime str
        official_time: str | None           # netkeiba が返した official_datetime
        update_count: int
        raw_reason:  str                    # 問題あった時の短文説明
        http_status: int
        response_url: str
        raw_body:    str (≤ 32KB trim)      # 生 JSON body の頭
        pipeline_version: str                # "odds-fetcher-v5.0-2026-04-19"

旧モジュールとの分離:

  - `scraper.py` は**直接触らない** (fetch_entries_netkeiba 等の既存機能は
    別の事物なので共存)。
  - 旧 `odds_sources.py` も呼ばない。本モジュールは自己完結。
  - consensus / disagreement / overlay の概念を一切持ち込まない。
    単勝オッズは JRA が公開する確定値が 1 本あるのみ。複数ソース比較で
    値が変動するような性質のものではない。
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import os as _os
import re as _re
from dataclasses import dataclass, field, asdict
from pathlib import Path as _Path
from typing import Optional

import requests as _requests


# ──────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────

FETCHER_VERSION = "odds-fetcher-v5.1-encoding-fix-2026-04-19"

# 単勝オッズのサニティ範囲。JRA 公式の pari-mutuel は下限 1.0, 実質上限
# 500 程度 (超大穴でも)。これを超える値を返すソースは単勝ではない別データ。
ODDS_MIN = 1.0
ODDS_MAX = 500.0

# netkeiba の単勝オッズ JSON API。既知 OSS ライブラリ
# (new-village/KeibaScraper) で実働している signature:
#   type=1       … 単勝
#   action=init  … 初回取得フラグ (付けないと他の bet type が返るケースが
#                   報告されている)
#   locale=ja    … 日本語ページ (レスポンス自体には影響しないが UX 一貫性)
_API_URL = (
    "https://race.netkeiba.com/api/api_get_jra_odds.html"
    "?type=1&action=init&locale=ja&race_id={race_id}"
)

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
    "X-Requested-With": "XMLHttpRequest",
    # 本物のリクエスト時 netkeiba は Referer を見てる可能性がある
    "Referer": (
        "https://race.netkeiba.com/odds/index.html"
        "?race_id={race_id}&rf=race_submenu&type=b1"
    ),
}


# ──────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────

@dataclass
class WinOddsResult:
    """Single-source, flat, explicit 単勝オッズ result.

    Contrasted with the legacy SourceResult dict — this is a dataclass,
    strongly typed, with no optional semantics crammed into dict keys.
    """
    status: str                        # "result" | "not-published" | "http-error" | "parse-error" | "shape-error"
    race_id: str
    by_number: dict                    # {int 馬番: float 単勝オッズ}
    rejected: dict = field(default_factory=dict)  # {int 馬番: float rejected value}
    fetched_at: str = ""
    official_time: Optional[str] = None
    update_count: int = 0
    raw_reason: str = ""
    http_status: int = 0
    response_url: str = ""
    raw_body: str = ""
    pipeline_version: str = FETCHER_VERSION

    def has_odds(self) -> bool:
        return self.status == "result" and bool(self.by_number)

    def for_horse(self, umaban: int) -> Optional[float]:
        """Return the odds for a specific 馬番, or None if missing."""
        return self.by_number.get(int(umaban))

    def as_dict(self) -> dict:
        d = asdict(self)
        # raw_body をヘッドだけに切る
        if len(d.get("raw_body") or "") > 8192:
            d["raw_body"] = d["raw_body"][:8192] + "...<truncated>"
        return d


# ──────────────────────────────────────────────────────
# Core fetcher
# ──────────────────────────────────────────────────────

def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _coerce_umaban(k) -> int:
    """Coerce a netkeiba 馬番 key ('01', '02', 1, '1') into int, 0 on fail."""
    try:
        return int(str(k).lstrip("0") or "0")
    except (TypeError, ValueError):
        return 0


def _coerce_odds_value(v) -> Optional[float]:
    """Parse any numeric-ish representation into a float. None on fail."""
    if v is None:
        return None
    s = str(v).replace(",", "").replace("倍", "").strip()
    if not s or s in ("---", "---.-", "--", "--.-", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _maybe_dump_trace(race_id: str, body: str) -> None:
    """If ODDS_TRACE_DIR env var is set, save raw body for offline audit."""
    d = _os.environ.get("ODDS_TRACE_DIR")
    if not d:
        return
    try:
        p = _Path(d)
        p.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        out = p / f"odds_fetcher_v5_{race_id}_{ts}.json"
        out.write_text(body, encoding="utf-8")
        print(f"[odds-fetcher] trace saved: {out}")
    except Exception as e:
        print(f"[odds-fetcher] trace save failed: {e}")


def fetch_win_odds(race_id: str,
                   *,
                   timeout: int = 10,
                   session: Optional[_requests.Session] = None) -> WinOddsResult:
    """Fetch 単勝 odds for a single race via netkeiba JSON API.

    Returns a fully-populated `WinOddsResult` regardless of whether the
    fetch succeeded — callers can inspect `.status` and `.has_odds()`.

    This function does **not** fall back to other sources. It does one
    thing (hit netkeiba's single-win odds API) and returns exactly what
    that API gave us. Cross-source reconciliation is intentionally not
    done here — if the API response is wrong, we surface that via the
    raw body and the caller can decide what to do.
    """
    url = _API_URL.format(race_id=race_id)
    headers = dict(_BROWSER_HEADERS)
    headers["Referer"] = headers["Referer"].format(race_id=race_id)

    result = WinOddsResult(
        status="http-error",
        race_id=race_id,
        by_number={},
        fetched_at=_now_iso(),
    )

    # ── STEP 1: HTTP fetch ──
    body = ""
    try:
        sess = session or _requests.Session()
        resp = sess.get(url, headers=headers, timeout=timeout)
        result.http_status = resp.status_code
        result.response_url = resp.url or url
        # Encoding: JSON API は通常 UTF-8 だが、Content-Type に charset を
        # 書かないケースがあるので `resp.text` (requests 既定 ISO-8859-1)
        # に依存せず、bytes → UTF-8 を明示する。文字化け対策。
        body = resp.content.decode("utf-8", errors="replace") if resp.content else ""
    except Exception as e:
        result.raw_reason = f"fetch-failed: {e.__class__.__name__}: {e}"
        return result

    result.raw_body = body[:8192] + ("...<trim>" if len(body) > 8192 else "")
    _maybe_dump_trace(race_id, body)

    if result.http_status != 200:
        result.raw_reason = f"HTTP {result.http_status}"
        return result

    # ── STEP 2: JSON parse ──
    try:
        payload = _json.loads(body)
    except Exception as e:
        result.status = "parse-error"
        result.raw_reason = f"json-parse-failed: {e.__class__.__name__}: {e}"
        return result

    if not isinstance(payload, dict):
        result.status = "shape-error"
        result.raw_reason = f"top-level not dict: {type(payload).__name__}"
        return result

    # netkeiba's own status / reason
    raw_status = str(payload.get("status") or "").strip()
    result.raw_reason = str(payload.get("reason") or "").strip()
    try:
        result.update_count = int(payload.get("update_count") or 0)
    except (TypeError, ValueError):
        result.update_count = 0

    # data is "" (empty string) when netkeiba is in "middle" / pre-publish state
    data = payload.get("data")
    if not isinstance(data, dict) or not data:
        result.status = "not-published" if raw_status in ("middle", "before", "") else "shape-error"
        result.raw_reason = result.raw_reason or f"data empty (raw_status={raw_status!r})"
        return result

    result.official_time = data.get("official_datetime") or None

    # ── STEP 3: locate 単勝 odds block ──
    # Expected shape (as of 2026-04): data.odds["1"] = { "01": ["4.5","","3"], ... }
    odds_root = data.get("odds")
    if not isinstance(odds_root, dict):
        result.status = "shape-error"
        result.raw_reason = f"data.odds not dict: type={type(odds_root).__name__}"
        return result

    tan_block = odds_root.get("1")
    if not isinstance(tan_block, dict) or not tan_block:
        # 単勝 keyed section missing — possible API change
        result.status = "shape-error"
        result.raw_reason = (
            f"data.odds['1'] missing or empty; top-level odds keys: "
            f"{sorted(odds_root.keys())[:6]}"
        )
        return result

    # ── STEP 4: parse each entry ──
    parsed: dict[int, float] = {}
    rejected: dict[int, float] = {}

    for key, arr in tan_block.items():
        um = _coerce_umaban(key)
        if um <= 0:
            continue
        # Expected arr shape: [<odds str>, "", <popularity str>]
        # Be tolerant of variations: [<odds>, <pop>] / <odds str>
        if isinstance(arr, (list, tuple)) and arr:
            v = _coerce_odds_value(arr[0])
        elif isinstance(arr, (str, int, float)):
            v = _coerce_odds_value(arr)
        else:
            continue
        if v is None:
            continue
        if ODDS_MIN <= v <= ODDS_MAX:
            parsed[um] = v
        else:
            rejected[um] = v

    result.by_number = parsed
    result.rejected = rejected

    if parsed:
        result.status = "result"
    else:
        # Got past shape but nothing parsed — either all rejected or empty
        result.status = "not-published" if raw_status in ("middle", "before", "") else "shape-error"
        result.raw_reason = result.raw_reason or "no parseable odds rows"

    return result


# ──────────────────────────────────────────────────────
# Self-test helpers (not called by production code)
# ──────────────────────────────────────────────────────

def _selftest_parse(sample_json: str) -> WinOddsResult:
    """Parse a JSON string as if it came from the API. Used in unit tests.

    This bypasses the HTTP step so we can feed known responses and verify
    parsing behavior without the network.
    """
    result = WinOddsResult(
        status="http-error", race_id="selftest", by_number={},
        fetched_at=_now_iso(), http_status=200, raw_body=sample_json,
    )
    try:
        payload = _json.loads(sample_json)
    except Exception as e:
        result.status = "parse-error"
        result.raw_reason = str(e)
        return result
    data = payload.get("data")
    if not isinstance(data, dict) or not data:
        result.status = "not-published"
        return result
    tan = (data.get("odds") or {}).get("1") or {}
    parsed = {}
    rejected = {}
    for key, arr in tan.items():
        um = _coerce_umaban(key)
        if um <= 0:
            continue
        v = _coerce_odds_value(arr[0] if isinstance(arr, (list, tuple)) else arr)
        if v is None:
            continue
        if ODDS_MIN <= v <= ODDS_MAX:
            parsed[um] = v
        else:
            rejected[um] = v
    result.by_number = parsed
    result.rejected = rejected
    result.status = "result" if parsed else "shape-error"
    return result
