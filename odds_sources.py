"""Multi-source 単勝オッズ consensus.

Context:
  `docs/odds_pipeline_audit.md` §2 の RC-1 / RC-3 に関連して、
  ユーザ決定 (2026-04-18):
    - 主権 (primary) ソース: **JRA 公式** (www.jra.go.jp)
    - 並列ソース: netkeiba JSON API, Yahoo 競馬
    - 衝突時: 主権を握り、他と 20% 以上ずれたら警告ドロップ
      (= LOOSE 発火を保留する、憲法 §7.1 のバグ修正として)

  憲法 §7.2 で固定されているのは LOOSE 4 条件の**数値**。
  「不正確なオッズでは LOOSE を発火させない」という安全弁の追加は
  数値条件の改変ではなく、入力の整合性保証に当たるので bug-fix 枠。

第2波修正 (2026-04-18):
  - Yahoo regex parser が斤量や賞金を誤ってオッズとして拾う重大バグ
    (`168.8 / 681.5 / 782.7` 事件) を受けて、BeautifulSoup + CSS
    selectors ベースに書き直し。
  - `ODDS_MIN / ODDS_MAX` による全ソース共通のサニティバウンドを導入。
  - JRA 公式を実動させるため accessN → accessO の POST セッション
    フローを実装。成功時は primary 権を取得する。
  - Yahoo は `enable_yahoo=False` をデフォルトに変更
    (オプトインでのみ有効、cross-check 専用のノイジーソース)。

Structure:
  各ソースは `SourceResult` shape を返す:
    {
      "source":        str,                 # "jra-official" | "netkeiba-api" | "yahoo-keiba"
      "status":        str,                 # "result" | "not-published" | "error" | "not-implemented"
      "by_number":     {int: float},        # 馬番 → 単勝オッズ
      "fetched_at":    str,                 # ISO local timestamp
      "raw_reason":    str,
      "http_status":   int,
      "response_url":  str,
      "parse_error":   Optional[str],
      "schema_guess":  str,
      "rejected":      {int: float},        # sanity 落ちした値 (audit 用)
    }

  aggregator `fetch_odds_consensus()` は:
    {
      "primary_source":  str,               # 実際に採用した主ソース名
      "primary_by_number": {int: float},   # 採用オッズ
      "per_source":       {src: SourceResult},
      "disagreements":    {int: {...}},    # 馬番 → 不一致詳細
      "fetched_at":       str,
      "has_disagreement_any": bool,
    }

  disagreement[umaban] = {
    "max_pct":  float,   # 最大相対差 = |a-b| / min(a,b)
    "values":   {src: odds},
    "flag":     bool,    # >= DISAGREEMENT_PCT_THRESHOLD
  }
"""

from __future__ import annotations

import datetime as _dt
import json as _json
import re as _re
from typing import Optional

import requests as _requests
from bs4 import BeautifulSoup as _BS


# ──────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────

DISAGREEMENT_PCT_THRESHOLD = 0.20   # 20% 差を "大きい" と定義

# 単勝オッズのサニティ境界。これを超える値はあらゆるソースで拒否する。
# 現実の JRA 単勝オッズは最大 ~500 倍程度 (超巨大フィールドの大穴でも)
# なので、それ以上の値を返してくるソースは別カラムを拾っている。
ODDS_MIN = 1.0      # JRA 単勝最小値
ODDS_MAX = 500.0    # 超保守的な上限

# Primary の優先順位 — 左が最優先
#
# 2026-04-19 update (フォルテアンジェロ事件 第4波):
#   netkeiba の JSON API (`netkeiba-api`) が単勝以外のオッズを返すケースが
#   観測されたため、**shutuba HTML から直接パースした値 (`shutuba-direct`)
#   を最上位**に昇格。JSON API は fill-in / cross-check の役に降格。
#
#   jra-official は現状未実装なので実質スキップ、
#   shutuba-direct → netkeiba-api → yahoo-keiba の順で fall-through する。
PRIMARY_PRIORITY = ("jra-official", "shutuba-direct", "netkeiba-api", "yahoo-keiba")

# JRA venue: race_id の 3-4 文字目 → 表示名 + JRA 内部コード
# race_id 例: 202401010411 → year=2024, venue=01(札幌), kaisai=01, day=04, race=11
RACE_ID_VENUE_TO_NAME = {
    "01": "札幌", "02": "函館", "03": "福島", "04": "新潟", "05": "東京",
    "06": "中山", "07": "中京", "08": "京都", "09": "阪神", "10": "小倉",
}

# JRA 公式サイトで使うコード (racetrackCd): JRA 内部仕様
# 東京=05, 中山=06, 京都=08, 阪神=09, 中京=07, 小倉=10, 新潟=04, 福島=03,
# 札幌=01, 函館=02
JRA_RACETRACK_CD = {
    "01": "01", "02": "02", "03": "03", "04": "04", "05": "05",
    "06": "06", "07": "07", "08": "08", "09": "09", "10": "10",
}


_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
}

_REQUEST_TIMEOUT = 12


def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def _empty_result(source: str, status: str = "error", reason: str = "") -> dict:
    return {
        "source":        source,
        "status":        status,
        "by_number":     {},
        "fetched_at":    _now_iso(),
        "raw_reason":    reason,
        "http_status":   0,
        "response_url":  "",
        "parse_error":   None,
        "schema_guess":  "",
        "rejected":      {},
    }


def parse_race_id(race_id: str) -> dict:
    """netkeiba race_id を構成要素に分解する。

    race_id 例: 202401010411 →
      {
        "year": "2024",
        "venue_code": "01",
        "venue_name": "札幌",
        "kaisai": "01",
        "kaisai_day": "04",
        "race_no": 11,
      }

    形式が合わない場合は空 dict を返す。
    """
    m = _re.match(r"^(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})$", str(race_id or ""))
    if not m:
        return {}
    y, v, k, d, r = m.groups()
    return {
        "year":       y,
        "venue_code": v,
        "venue_name": RACE_ID_VENUE_TO_NAME.get(v, ""),
        "kaisai":     k,
        "kaisai_day": d,
        "race_no":    int(r),
    }


def _apply_sanity_bounds(raw_by_number: dict) -> tuple[dict, dict]:
    """Return (kept, rejected). Reject values outside [ODDS_MIN, ODDS_MAX]."""
    kept: dict[int, float] = {}
    rejected: dict[int, float] = {}
    for um, v in (raw_by_number or {}).items():
        try:
            um_i = int(um)
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not (ODDS_MIN <= vf <= ODDS_MAX):
            rejected[um_i] = vf
            continue
        kept[um_i] = vf
    return kept, rejected


# ──────────────────────────────────────────────────────
# Source 1: netkeiba JSON API (既存の scraper.fetch_odds_netkeiba を薄くラップ)
# ──────────────────────────────────────────────────────

def fetch_odds_from_entries(entries: list) -> dict:
    """Convert entries[*].odds (already parsed from shutuba HTML) into a
    SourceResult.

    entries は scraper.fetch_entries_netkeiba の戻り値。各要素は
    {number, name, odds, ...} 形式。`odds` は `_parse_shutuba_odds` に
    よって既にサニティ済みで、"0" は欠損を表す。

    このソースはネットワーク fetch を行わず、既に取れている値を
    consensus の最上位 primary に昇格するためだけに存在する。

    ユーザから「スクレイピング結果は正しそう」と確認された後、
    shutuba パスを最優先にするための虚構ソース (virtual source) として
    追加された (2026-04-19 フォルテアンジェロ事件 第4波)。
    """
    result = _empty_result("shutuba-direct", "not-published",
                            "no valid odds in entries")
    parsed: dict[int, float] = {}
    for e in entries or []:
        try:
            num = int(str(e.get("number", "")).strip() or 0)
        except (TypeError, ValueError):
            continue
        try:
            v = float(str(e.get("odds", "0")).replace("---", "0")
                      .replace("--", "0").replace(",", "").strip() or 0)
        except ValueError:
            continue
        if num > 0 and ODDS_MIN <= v <= ODDS_MAX:
            parsed[num] = v
    kept, rejected = _apply_sanity_bounds(parsed)
    result["rejected"] = rejected
    if kept:
        result["status"] = "result"
        result["by_number"] = kept
        result["schema_guess"] = "shutuba-direct-v1"
        result["raw_reason"] = ""
    return result


def fetch_odds_netkeiba_api(race_id: str) -> dict:
    """Call the existing scraper.fetch_odds_netkeiba and normalize the shape."""
    try:
        import scraper as _s
        raw = _s.fetch_odds_netkeiba(race_id) or {}
    except Exception as e:
        return _empty_result("netkeiba-api", "error",
                             f"exception: {e.__class__.__name__}: {e}")

    status = raw.get("status") or "error"
    if status not in ("result", "not-published", "error"):
        status = "error"

    kept, rejected = _apply_sanity_bounds(raw.get("by_number") or {})
    # サニティで全滅したら status は error とみなす
    if raw.get("by_number") and not kept:
        status = "error"

    return {
        "source":        "netkeiba-api",
        "status":        status,
        "by_number":     kept,
        "fetched_at":    raw.get("fetched_at") or _now_iso(),
        "raw_reason":    raw.get("raw_reason") or "",
        "http_status":   int(raw.get("http_status") or 0),
        "response_url":  raw.get("response_url") or "",
        "parse_error":   raw.get("parse_error"),
        "schema_guess":  raw.get("schema_version_guess") or "",
        "rejected":      rejected,
    }


# ──────────────────────────────────────────────────────
# Source 2: JRA official
# ──────────────────────────────────────────────────────
#
# JRA 公式 (www.jra.go.jp/JRADB/) は SPA のトップ → POST-session flow で
# 各画面に到達する。単勝オッズ画面へのフローは以下:
#
#   (1) GET  https://www.jra.go.jp/JRADB/accessO.html
#            → session cookie を確立、トップのフォーム HTML を取得
#   (2) POST https://www.jra.go.jp/JRADB/accessO.html
#            cname=<encoded race key>
#            → 単勝・複勝オッズ HTML を受信
#
# cname の race key 形式: pwXXsdtYYYYMMDDZZZZ... の pw 付き文字列で、
# 日付 + 会場コード + レース番号を含む (具体的な encoding は JRA 内部)。
# このコード化は年によって微妙に変わるので、失敗時は "not-implemented"
# で返し、consensus 層が netkeiba API にフォールバックする。


def _build_jra_cname(race_id: str) -> Optional[str]:
    """Best-effort build of JRA's cname key from netkeiba race_id.

    執筆時点 (2026) の観測では:
      cname = "pw15odd" + <session-id-like> + "sdt<YYYYMMDD>" + <kaisai>
    ただしこの形式は JRA が随時変える。確証なしのため、ここでは単純な
    パターンだけ構築し、失敗したら None を返して "not-implemented" に
    グレースフルに落ちる設計にする。
    """
    meta = parse_race_id(race_id)
    if not meta:
        return None
    # ここで具体的な cname を構築するロジックは JRA 側の仕様依存。
    # 現時点で確証のあるフォーマットは公開されていないので、
    # 失敗扱い (None) にしてフォールバックさせる。
    # 実運用で確実に動かすには、JRA トップページを取得して hidden form
    # フィールドを抽出する 3-step flow が必要。
    return None


def fetch_odds_jra_official(race_id: str,
                            race_date: Optional[str] = None,
                            venue: Optional[str] = None,
                            race_no: Optional[int] = None) -> dict:
    """Try to fetch 単勝 odds from JRA 公式 site via POST session flow.

    Returns a SourceResult dict. 現状 JRA 側の cname 符号化が
    随時変わるため、最低限のフローを試みて失敗したら
    `status="not-implemented"` を明示的に返す。
    """
    result = _empty_result("jra-official", "not-implemented")
    meta = parse_race_id(race_id)
    venue = venue or meta.get("venue_name") or ""
    race_no = race_no if race_no is not None else meta.get("race_no")

    if not venue or not race_no:
        result["status"] = "error"
        result["raw_reason"] = f"cannot resolve venue/race_no from race_id={race_id!r}"
        return result

    # Step 1: GET accessO.html to establish session
    session = _requests.Session()
    session.headers.update(_DEFAULT_HEADERS)
    session.headers["Referer"] = "https://www.jra.go.jp/"
    top_url = "https://www.jra.go.jp/JRADB/accessO.html"
    try:
        resp = session.get(top_url, timeout=_REQUEST_TIMEOUT)
        result["http_status"] = resp.status_code
        result["response_url"] = resp.url or top_url
    except Exception as e:
        result["status"] = "error"
        result["raw_reason"] = f"session-init-failed: {e.__class__.__name__}: {e}"
        return result

    if resp.status_code != 200:
        result["status"] = "error"
        result["raw_reason"] = f"session-init HTTP {resp.status_code}"
        return result

    # Step 2: Build cname for the target race
    cname = _build_jra_cname(race_id)
    if cname is None:
        # 明示的に「実装未完了」を伝える。consensus が netkeiba へ
        # フォールバックするだけで運用上の問題はない。
        result["status"] = "not-implemented"
        result["raw_reason"] = (
            "JRA cname encoding is version-dependent and not yet wired. "
            "Consensus layer will degrade to netkeiba-api."
        )
        result["schema_guess"] = "jra-cname-unresolved"
        return result

    # Step 3: POST to get odds HTML
    try:
        resp2 = session.post(top_url, data={"cname": cname},
                             timeout=_REQUEST_TIMEOUT)
        result["http_status"] = resp2.status_code
        result["response_url"] = resp2.url or top_url
        body = resp2.text
    except Exception as e:
        result["status"] = "error"
        result["raw_reason"] = f"post-failed: {e.__class__.__name__}: {e}"
        return result

    if resp2.status_code != 200 or not body:
        result["status"] = "error"
        result["raw_reason"] = f"post HTTP {resp2.status_code}"
        return result

    # Step 4: Parse 単勝 odds from the HTML
    parsed = _parse_jra_odds_html(body)
    kept, rejected = _apply_sanity_bounds(parsed)
    result["rejected"] = rejected
    if kept:
        result["status"] = "result"
        result["by_number"] = kept
        result["schema_guess"] = "jra-accessO-v1"
    else:
        result["status"] = "error"
        result["raw_reason"] = "parsed 0 odds rows (unexpected JRA HTML shape)"
        result["schema_guess"] = "jra-accessO-empty-or-unknown"
    return result


def _parse_jra_odds_html(body: str) -> dict:
    """Parse 単勝オッズ from a JRA accessO.html response.

    JRA の単勝オッズテーブルは典型的に `<table>` > `<tr>` 行で、
    馬番セル `class="num"` / オッズセル `class="odds_tan"` の形。
    仕様変更に備えて複数の selector を試す。
    """
    soup = _BS(body, "html.parser")
    parsed: dict[int, float] = {}

    # Attempt 1: 明示的な class 指定
    for row in soup.select("tr"):
        num_cell = row.select_one(".num, .umaban, td.waku + td")
        odds_cell = row.select_one(".odds_tan, .oc, td.odds")
        if not num_cell or not odds_cell:
            continue
        num_txt = num_cell.get_text(strip=True)
        odds_txt = odds_cell.get_text(strip=True)
        if not _re.match(r"^\d+$", num_txt):
            continue
        if not _re.match(r"^\d+(\.\d+)?$", odds_txt):
            continue
        try:
            parsed[int(num_txt)] = float(odds_txt)
        except ValueError:
            continue

    return parsed


# ──────────────────────────────────────────────────────
# Source 3: Yahoo 競馬
# ──────────────────────────────────────────────────────
#
# 2026-04-18 重大バグ修正:
#   旧実装は <td> 単位の regex で「行内の最初の数値セル」を odds 扱い
#   していたが、これは斤量 (57.0) や収得賞金 (168.8) を誤って拾う。
#   観測された事故値:
#     フォルテアンジェロ=168.8 / アスクエジンバラ=681.5 /
#     アドマイヤクワッズ=782.7
#   → BeautifulSoup + 具体的な CSS selector ベースに書き直した上で、
#      サニティバウンド (1.0 ≤ odds ≤ 500.0) で最終防衛する。

def fetch_odds_yahoo_keiba(race_id: str,
                            race_date: Optional[str] = None,
                            venue: Optional[str] = None,
                            race_no: Optional[int] = None) -> dict:
    """Fetch 単勝 odds from Yahoo 競馬 using BS + specific selectors.

    Best-effort. 構造変更 / ページ構造不明時は "error" を返す。
    絶対にランダム数値を "odds" として返さない (サニティ保証)。
    """
    meta = parse_race_id(race_id)
    year = meta.get("year")
    venue_code = meta.get("venue_code")
    kaisai = meta.get("kaisai")
    day = meta.get("kaisai_day")
    race_no = race_no if race_no is not None else meta.get("race_no")

    if not (year and venue_code and race_no):
        return _empty_result("yahoo-keiba", "error",
                             f"cannot resolve components from race_id={race_id!r}")

    yahoo_key = f"{year}{venue_code}{kaisai}{day}{race_no:02d}"
    url = f"https://keiba.yahoo.co.jp/race/denma/{yahoo_key}/odds/tfw/"

    result = _empty_result("yahoo-keiba", "error")
    result["response_url"] = url

    try:
        resp = _requests.get(url, headers=_DEFAULT_HEADERS,
                             timeout=_REQUEST_TIMEOUT, allow_redirects=True)
        result["http_status"] = resp.status_code
        result["response_url"] = resp.url or url
        if resp.status_code != 200:
            result["raw_reason"] = f"http-error: {resp.status_code}"
            return result
        body = resp.text
    except Exception as e:
        result["raw_reason"] = f"fetch-failed: {e.__class__.__name__}: {e}"
        return result

    parsed = _parse_yahoo_odds_html(body)
    kept, rejected = _apply_sanity_bounds(parsed)
    result["rejected"] = rejected
    if kept:
        result["status"] = "result"
        result["by_number"] = kept
        result["schema_guess"] = "yahoo-keiba-html-v2"
    else:
        # 取得は成功したが odds 行が読めない → 構造変更 or 未公開
        result["status"] = "not-published"
        result["raw_reason"] = result["raw_reason"] or "no valid odds rows after sanity"
        result["schema_guess"] = "yahoo-keiba-html-empty-or-shape-changed"
    return result


def _parse_yahoo_odds_html(body: str) -> dict:
    """Parse 単勝オッズ rows from Yahoo 競馬 odds HTML.

    戦略:
      (1) <table> を BS で取り、各 <tr> を走査。
      (2) 馬番セル (`td.num` / class を持つ短い数値セル) を特定。
      (3) 単勝オッズセルを class ベースで特定 (`.oc`, `td.txc.ttl + td`, etc.)。
      (4) それでも見つからなければ、そのレースの全 <td> 数値値を取り出し、
          サニティバウンドに収まるもの「だけ」を馬番順に 1 対 1 対応させる。
          (これは最後の手段。false positive は _apply_sanity_bounds で最終的に
           振るい落とされる。)
    """
    soup = _BS(body, "html.parser")
    parsed: dict[int, float] = {}

    # (1)+(2)+(3): 明示的 selector
    for row in soup.select("tr"):
        # 馬番セル候補
        num_cell = (
            row.select_one("td.num") or
            row.select_one("td.umaban") or
            row.select_one("td[class*='num']")
        )
        # 単勝オッズセル候補
        odds_cell = (
            row.select_one("td.oc") or
            row.select_one("td[class*='odds']:not([class*='fuku']):not([class*='place'])") or
            row.select_one("td.tfw") or
            row.select_one("td[class*='tan']")
        )
        if not num_cell or not odds_cell:
            continue
        num_txt = num_cell.get_text(strip=True)
        odds_txt = odds_cell.get_text(strip=True)
        if not _re.match(r"^\d{1,2}$", num_txt):
            continue
        if not _re.match(r"^\d+(\.\d+)?$", odds_txt):
            continue
        try:
            v = float(odds_txt)
        except ValueError:
            continue
        parsed[int(num_txt)] = v

    # セレクタ戦略で取れなかった場合はフォールバックしない
    # (旧 regex パーサと違い、"分からない時は空で返す" が正しい挙動)。
    return parsed


# ──────────────────────────────────────────────────────
# Aggregator
# ──────────────────────────────────────────────────────

def _compute_disagreements(per_source: dict) -> tuple[dict, bool]:
    """Return (disagreements, has_any_flag).

    disagreements[umaban] = {
      "max_pct":  float,
      "values":   {source: odds_float},
      "flag":     bool (True if max_pct >= threshold AND ≥2 sources reported),
    }
    """
    # 収集: 馬番 → {source: odds}
    by_horse: dict[int, dict] = {}
    for src, res in per_source.items():
        if res.get("status") != "result":
            continue
        for um, v in (res.get("by_number") or {}).items():
            try:
                um_i = int(um)
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not (ODDS_MIN <= vf <= ODDS_MAX):
                continue
            by_horse.setdefault(um_i, {})[src] = vf

    disagreements: dict[int, dict] = {}
    any_flag = False
    for um, vals in by_horse.items():
        if len(vals) < 2:
            disagreements[um] = {
                "max_pct": 0.0,
                "values":  vals,
                "flag":    False,
            }
            continue
        nums = list(vals.values())
        lo, hi = min(nums), max(nums)
        max_pct = (hi - lo) / lo if lo > 0 else 0.0
        flag = max_pct >= DISAGREEMENT_PCT_THRESHOLD
        if flag:
            any_flag = True
        disagreements[um] = {
            "max_pct": round(max_pct, 4),
            "values":  vals,
            "flag":    flag,
        }
    return disagreements, any_flag


def _pick_primary(per_source: dict) -> str:
    """Return primary source name by priority (fallback to 'none')."""
    for name in PRIMARY_PRIORITY:
        res = per_source.get(name) or {}
        if res.get("status") == "result" and res.get("by_number"):
            return name
    return "none"


def fetch_odds_consensus(race_id: str,
                         race_date: Optional[str] = None,
                         venue: Optional[str] = None,
                         race_no: Optional[int] = None,
                         entries: Optional[list] = None,
                         enable_jra: bool = True,
                         enable_yahoo: bool = False) -> dict:
    """Run all enabled source fetchers and return a consensus dict.

    Priority (2026-04-19 第4波):
      shutuba-direct > jra-official > netkeiba-api > yahoo-keiba

    `entries` を受け取ると shutuba-direct が consensus に参加する。
    これによりユーザ環境の scraper が拾った値 (フォルテアンジェロ事件で
    scraping 結果は正しいと確認された) が最優先 primary として採用され、
    JSON API が単勝以外を返してくるケース (現象ケース) では JSON API が
    fill-in / cross-check 専用に降格する。

    Defaults:
      - enable_jra=True: JRA 公式を試みる
      - enable_yahoo=False: Yahoo パーサは opt-in

    Returns:
      {
        "primary_source":        "shutuba-direct" | "jra-official" | "netkeiba-api" | ...
        "primary_by_number":     {int: float},
        "per_source":            {source_name: SourceResult},
        "disagreements":         {int: {max_pct, values, flag}},
        "has_disagreement_any":  bool,
        "fetched_at":            str,
        "enabled_sources":       [str],
      }
    """
    per_source: dict = {}

    # 0. shutuba-direct (scraper 側で既に取れた値を優先最上位)
    if entries is not None:
        per_source["shutuba-direct"] = fetch_odds_from_entries(entries)

    # 1. netkeiba JSON API (always; fill-in / cross-check)
    per_source["netkeiba-api"] = fetch_odds_netkeiba_api(race_id)

    # 2. JRA official (opt-in; default True)
    if enable_jra:
        per_source["jra-official"] = fetch_odds_jra_official(
            race_id, race_date=race_date, venue=venue, race_no=race_no,
        )

    # 3. Yahoo 競馬 (opt-in; default False until verified per race)
    if enable_yahoo:
        per_source["yahoo-keiba"] = fetch_odds_yahoo_keiba(
            race_id, race_date=race_date, venue=venue, race_no=race_no,
        )

    primary = _pick_primary(per_source)
    primary_by_number = {}
    if primary != "none":
        primary_by_number = dict(per_source[primary].get("by_number") or {})

    # Fill-in: primary が欠けた馬 (scratch で shutuba に無い等) は、
    # 次点ソースで補完する。優先順位は PRIMARY_PRIORITY。
    if primary_by_number:
        for src_name in PRIMARY_PRIORITY:
            if src_name == primary:
                continue
            src = per_source.get(src_name) or {}
            if src.get("status") != "result":
                continue
            for um, v in (src.get("by_number") or {}).items():
                if um not in primary_by_number:
                    primary_by_number[um] = v

    disagreements, any_flag = _compute_disagreements(per_source)

    return {
        "primary_source":       primary,
        "primary_by_number":    primary_by_number,
        "per_source":           per_source,
        "disagreements":        disagreements,
        "has_disagreement_any": any_flag,
        "fetched_at":           _now_iso(),
        "enabled_sources": [s for s, e in
                            (("shutuba-direct", entries is not None),
                             ("netkeiba-api", True),
                             ("jra-official", enable_jra),
                             ("yahoo-keiba", enable_yahoo))
                            if e],
    }


# ──────────────────────────────────────────────────────
# Helpers for live_pipeline integration
# ──────────────────────────────────────────────────────

def disagreement_for_number(consensus: dict, umaban: int) -> dict:
    """Return disagreement record for a given 馬番 (always a dict)."""
    return (consensus.get("disagreements") or {}).get(int(umaban)) or {
        "max_pct": 0.0, "values": {}, "flag": False,
    }


def summarize_disagreement(consensus: dict) -> str:
    """Human-readable summary of which sources disagreed."""
    flagged = [(um, d) for um, d in (consensus.get("disagreements") or {}).items()
               if d.get("flag")]
    if not flagged:
        return "all-agree"
    parts = []
    for um, d in sorted(flagged):
        vals = ", ".join(f"{s}={v:.1f}" for s, v in d["values"].items())
        parts.append(f"#{um}:{d['max_pct']*100:.0f}% ({vals})")
    return " | ".join(parts)
