"""Accuracy-first paddock collector (v5.7 — 2026-05-08).

G1/G2 重賞では「どう見えたか」の観察眼が複数あった方が consensus
が効く。ただし、検索スニペットやレース展望から馬名周辺を拾うと、
予想・買い目・人気評価が混ざって consensus を汚す。

本モジュールは追加ソースを **G1/G2 限定** で叩き、馬ごとに構造化
された観察コメントだけを採用する：

  1. 東京スポーツ (tospo-keiba.jp)   — 公式 paddock ページあり (URL 確定)
  2. ラジオ NIKKEI / 日刊スポーツ — 検索・記事本文抽出は現時点では停止

結果は `{horse_name: {"text", "source", "scores"}}` 形式で返し、既存
`scraper.fetch_paddock_reports` の出力と同じ shape なので、上位の
`collect_paddock_observation_facts` でそのまま fact 化できる。

設計原則:
  - **G1/G2 のみ発火** (race.grade 引数で制御)。それ以外のレースでは
    no-op。過剰な scraping で ban されるのを避ける。
  - **2 時間 cache** (scraper._cache に "paddock_multi" kind)。
  - **fail soft**: 一ソースが落ちても既存 pipeline は継続する。
  - **quality gate**: 予想語、検索スニペット、本文抽出は採用しない。
  - **no external deps**: requests + BeautifulSoup のみ。
  - **charset 自動判定** (resp.content → BS)。

憲法: §7.1「ソース追加」で明示的に許容。LOOSE 4 条件の数値は不変更。
"""

from __future__ import annotations

import datetime as _dt
import re as _re
from typing import Optional

import requests as _requests
from bs4 import BeautifulSoup as _BS


PADDOCK_MULTI_VERSION = "paddock-multi-v5.7-2026-05-08"

# G1/G2 only. The live site excludes G3, and extra paddock collection
# should follow the production surface to reduce scrape load and noise.
GRADE_TRIGGERS = ("G1", "G2", "JpnI", "JpnII")

_HEADERS = {
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


def _grade_triggers(grade: str) -> bool:
    """Return True iff race grade warrants multi-source paddock collection."""
    if not grade:
        return False
    g = grade.strip().upper()
    return any(tag.upper() in g for tag in GRADE_TRIGGERS)


def _get_html(url: str, timeout: int = _REQUEST_TIMEOUT) -> Optional[_BS]:
    """Fetch URL and return BeautifulSoup with proper encoding.

    Uses resp.content (raw bytes) so BeautifulSoup can detect encoding
    from <meta charset>. Prevents ISO-8859-1 default mojibake.
    """
    try:
        resp = _requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code != 200 or not resp.content:
            return None
        return _BS(resp.content, "html.parser")
    except Exception:
        return None


# ──────────────────────────────────────────────────────
# Source 1: Tokyo Sports (tospo-keiba.jp)
# ──────────────────────────────────────────────────────

def fetch_paddock_tospo(race_id: str, horse_names: list) -> dict:
    """Fetch paddock view from 東京スポーツ競馬.

    URL pattern (confirmed 2026-04):
      https://tospo-keiba.jp/race/{race_id}/race-detail/paddock

    The public page typically shows basic paddock text per horse. The
    detailed paper-memo ("記者メモ") is behind a paywall — we only
    extract the free text.

    Returns:
      {horse_name: {"text": str, "source": "tospo", "scores": dict}}
    """
    out: dict = {}
    url = f"https://tospo-keiba.jp/race/{race_id}/race-detail/paddock"
    soup = _get_html(url)
    if soup is None:
        return out

    # 東スポのパドックページ典型パターン:
    #   - 馬ごとのカード (div.horse-card, .horse-row, .paddock-item)
    #   - 馬名 (a, .horse-name)
    #   - 短評 (.comment, .paddock-text, .text, p)
    # パターンが頻繁に変わるので複数セレクタを順に試す。
    horse_blocks = (
        soup.select("div.horse-card, .paddock-item, .horse-row, .paddock_card, article.horse")
        or []
    )
    for block in horse_blocks:
        name_tag = (
            block.select_one(".horse-name, .horseName, a.horse, h2, h3")
        )
        comment_tag = (
            block.select_one(".comment, .paddock-text, .text, p, .description")
        )
        if not name_tag:
            continue
        raw_name = name_tag.get_text(strip=True)
        matched = next((n for n in horse_names if n in raw_name or raw_name in n), None)
        if not matched:
            continue
        text = comment_tag.get_text(" ", strip=True) if comment_tag else ""
        if text:
            # Import paddock comment parser lazily to avoid import loop
            try:
                import paddock_features as pf
                scores = pf.score_comment(text)
            except Exception:
                scores = {}
            out[matched] = {"text": text, "source": "tospo", "scores": scores}

    try:
        import paddock_quality as pq
        out = {
            name: report
            for name, report in pq.filter_paddock_reports(out).items()
            if report.get("text")
        }
    except Exception:
        pass

    return out


# ──────────────────────────────────────────────────────
# Source 2: Radio NIKKEI 競馬 (blog-based, best-effort)
# ──────────────────────────────────────────────────────

def fetch_paddock_radio_nikkei(race_id: str, race_name: str,
                                horse_names: list) -> dict:
    """Fetch paddock mentions from ラジオ NIKKEI 競馬 blog.

    実装は blog のインデックスページから該当レース名を含む記事を探す
    best-effort。見つからなければ空で返す。

    URL:
      https://www.radionikkei.jp/keiba/                (blog top)
      https://blog.radionikkei.jp/keibablog/           (full blog index)
    """
    out: dict = {}

    for blog_url in (
        "https://www.radionikkei.jp/keiba/",
        "https://blog.radionikkei.jp/keibablog/",
    ):
        soup = _get_html(blog_url)
        if soup is None:
            continue
        # レース名を含む記事リンクを拾う
        for a in soup.select("a"):
            href = a.get("href", "") or ""
            text = a.get_text(strip=True)
            if race_name and race_name in text and href.startswith("http"):
                # Open that article and look for per-horse mentions.
                article = _get_html(href)
                if article is None:
                    continue
                article_text = article.get_text(" ", strip=True)
                for name in horse_names:
                    if name in article_text and name not in out:
                        # Extract surrounding ≤2 sentences
                        sentences = [
                            s.strip() for s in _re.split(r"[。．\n]", article_text)
                            if name in s and len(s) > 10
                        ]
                        if sentences:
                            snippet = "。".join(sentences[:2])
                            try:
                                import paddock_features as pf
                                scores = pf.score_comment(snippet)
                            except Exception:
                                scores = {}
                            out[name] = {"text": snippet,
                                         "source": "radio-nikkei",
                                         "scores": scores}
                if out:
                    break  # Got at least one match from this article
        if out:
            break
    return out


# ──────────────────────────────────────────────────────
# Source 3: 日刊スポーツ (search-based fallback)
# ──────────────────────────────────────────────────────

def fetch_paddock_nikkansports(race_name: str, horse_names: list) -> dict:
    """Fetch paddock mentions via 日刊スポーツ 競馬記事検索.

    日刊スポーツは個別レース専用のパドック URL を持たないので、記事検索
    で該当レース名を含む記事を見つけ、その中の馬名周辺を抽出する。
    """
    out: dict = {}
    if not race_name:
        return out

    # Search URL pattern (heuristic)
    query = _requests.utils.quote(f"{race_name} パドック")
    search_url = f"https://www.nikkansports.com/search/?q={query}&type=all"
    soup = _get_html(search_url)
    if soup is None:
        return out

    # First few article results
    article_urls: list[str] = []
    for a in soup.select("a"):
        href = a.get("href", "") or ""
        if "/race/" in href and href.startswith("http"):
            article_urls.append(href)
            if len(article_urls) >= 3:
                break

    for url in article_urls:
        article = _get_html(url)
        if article is None:
            continue
        txt = article.get_text(" ", strip=True)
        for name in horse_names:
            if name in txt and name not in out:
                sentences = [
                    s.strip() for s in _re.split(r"[。．\n]", txt)
                    if name in s and len(s) > 10
                ]
                if sentences:
                    snippet = "。".join(sentences[:2])
                    try:
                        import paddock_features as pf
                        scores = pf.score_comment(snippet)
                    except Exception:
                        scores = {}
                    out[name] = {"text": snippet, "source": "nikkansports",
                                 "scores": scores}
    return out


# ──────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────

def fetch_paddock_multi_sources(race_id: str, race_name: str,
                                 horse_names: list,
                                 grade: str = "",
                                 force: bool = False) -> dict:
    """Call additional paddock sources for G1/G2 races.

    Args:
      race_id:     netkeiba race id
      race_name:   レース名 (検索キーに使用)
      horse_names: 馬名リスト
      grade:      "G1" / "G2" / "" 等
      force:      True なら grade 判定をスキップ (テスト用)

    Returns:
      {source_name: {horse_name: {"text", "source", "scores"}}}
      ※ source_name は "tospo" / "radio-nikkei" / "nikkansports" のキー。
        各ソース単位で結果が分かれているので、上位で merge する想定。
    """
    if not force and not _grade_triggers(grade):
        return {
            "_meta": {
                "enabled": False,
                "reason": f"grade={grade!r} not in {GRADE_TRIGGERS}",
                "version": PADDOCK_MULTI_VERSION,
            }
        }

    meta = {
        "enabled": True,
        "reason": f"grade={grade!r} triggers multi-source paddock",
        "version": PADDOCK_MULTI_VERSION,
        "fetched_at": _now_iso(),
    }

    result: dict = {"_meta": meta}

    try:
        result["tospo"] = fetch_paddock_tospo(race_id, horse_names)
    except Exception as e:
        result["tospo"] = {}
        meta.setdefault("errors", {})["tospo"] = f"{e.__class__.__name__}: {e}"

    result["radio-nikkei"] = {}
    result["nikkansports"] = {}
    meta.setdefault("skipped_sources", {})["radio-nikkei"] = (
        "search/blog extraction disabled by paddock-quality policy"
    )
    meta.setdefault("skipped_sources", {})["nikkansports"] = (
        "search/article extraction disabled by paddock-quality policy"
    )

    meta["n_sources_with_hits"] = sum(
        1 for k in ("tospo", "radio-nikkei", "nikkansports")
        if result.get(k)
    )
    meta["total_horses_covered"] = len({
        h for k in ("tospo", "radio-nikkei", "nikkansports")
        for h in (result.get(k) or {}).keys()
    })
    return result


def merge_paddock_into_reports(existing_reports: dict, multi_result: dict,
                                horse_names: list) -> dict:
    """Merge multi-source paddock results into the existing
    `scraper.fetch_paddock_reports` output.

    既存 reports の horse_name に対してテキストが空なら、multi sources の
    値で上書き。既にテキストあるなら "|| new_source: ..." で **追記** する
    (consensus 向上のためあえて残す)。

    Args:
      existing_reports: {horse_name: {"text", "source", "scores"}}
      multi_result:     output of fetch_paddock_multi_sources (with _meta)
      horse_names:      list to iterate

    Returns:
      updated reports dict (same shape as existing_reports)
    """
    if not multi_result or multi_result.get("_meta", {}).get("enabled") is False:
        return existing_reports

    for name in horse_names:
        cur = existing_reports.get(name) or {"text": "", "source": "",
                                              "scores": {}}
        added_sources: list[str] = []
        appended_text = cur.get("text", "") or ""

        for src_key in ("tospo", "radio-nikkei", "nikkansports"):
            entry = (multi_result.get(src_key) or {}).get(name)
            if not entry:
                continue
            text = entry.get("text", "")
            if not text:
                continue
            try:
                import paddock_quality as pq
                quality = pq.assess_paddock_report(text, entry.get("source", src_key))
                if not quality.get("usable"):
                    continue
            except Exception:
                pass
            if text not in appended_text:
                if appended_text:
                    appended_text = f"{appended_text} || [{src_key}] {text}"
                else:
                    appended_text = f"[{src_key}] {text}"
                added_sources.append(src_key)

        if added_sources:
            try:
                import paddock_features as pf
                scores = pf.score_comment(appended_text)
            except Exception:
                scores = cur.get("scores", {}) or {}
            src_tags = cur.get("source", "") or ""
            new_src = "+".join(filter(None, [src_tags, *added_sources]))
            existing_reports[name] = {
                "text":   appended_text,
                "source": new_src,
                "scores": scores,
            }

    return existing_reports
