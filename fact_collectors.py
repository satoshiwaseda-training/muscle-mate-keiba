"""Source adapters — JRA-first, Yahoo!競馬 hub, secondary articles.

Each collector returns a list of `Fact` objects along with a provenance
record for the Streamlit progress cards:

  {
    "source": "jra" | "yahoo" | "keibalab" | "news",
    "status": "ok" | "partial" | "failed" | "skipped",
    "facts": [Fact, ...],
    "items_seen": int,
    "error": str | None,
  }
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import scraper
from fact_schema import Fact
from fact_extractor import (
    extract_canonical_facts,
    fact_from_weight_delta,
    fact_from_track_condition,
    fact_from_scratch,
)


# ── Tier 1: JRA (via scraper, the closest public proxy) ────

def collect_jra_facts(race_id: str, venue: str = "") -> dict:
    """Collect Tier-1 objective facts — race info, entries, weights,
    scratches, track condition, cushion value.

    Uses the existing scraper module (which already knows how to hit
    JRA fallbacks and netkeiba shutuba). We tag these facts with
    source="jra" when the information is numeric and official-class,
    and source="netkeiba" when it's observational from the shutuba
    page but still close to official.
    """
    facts: list[Fact] = []
    items = 0
    error = None
    status = "ok"

    try:
        race_info = scraper.fetch_race_info_netkeiba(race_id) or {}
        entries = scraper.fetch_entries_netkeiba(race_id, venue) or []
        changes = {}
        try:
            changes = scraper.fetch_jra_race_changes(race_id) or {}
        except Exception:
            pass

        # Race-level facts
        tc = race_info.get("track_condition") or ""
        facts.extend(fact_from_track_condition(tc))

        cushion = race_info.get("cushion_value") or ""
        try:
            cv = float(cushion)
            if cv >= 9.5:
                facts.append(Fact(type="track_firm_cushion", horse=None, polarity=+1,
                                  confidence=0.85, source="jra",
                                  raw_text=f"cushion {cv}",
                                  category="track", meta={"cushion_value": cv}))
            elif cv <= 8.0 and cv > 0:
                facts.append(Fact(type="track_soft_cushion", horse=None, polarity=-1,
                                  confidence=0.85, source="jra",
                                  raw_text=f"cushion {cv}",
                                  category="track", meta={"cushion_value": cv}))
        except (ValueError, TypeError):
            pass

        scratched_names = set(changes.get("scratched") or [])

        for e in entries:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            items += 1
            if name in scratched_names:
                facts.append(fact_from_scratch(name))
                continue

            # Weight delta facts
            hw = e.get("horse_weight", "") or ""
            m = re.match(r"(\d{3,4})\(([+-]?\d+)\)", hw)
            if m:
                try:
                    facts.extend(fact_from_weight_delta(name, int(m.group(2))))
                except ValueError:
                    pass

            # Carried-weight fact (high burden)
            try:
                carried = float((e.get("weight") or "0").replace("kg", ""))
                if carried >= 58.0:
                    facts.append(Fact(
                        type="high_carried_weight", horse=name, polarity=-1,
                        confidence=0.80, source="jra",
                        raw_text=f"斤量 {carried:g}kg",
                        category="weight", meta={"carried_weight": carried},
                    ))
            except (ValueError, TypeError):
                pass

    except Exception as e:
        status = "failed"
        error = str(e)

    return {
        "source": "jra", "status": status if facts else "partial",
        "facts": facts, "items_seen": items, "error": error,
    }


# ── Tier 2: Yahoo!競馬 as a discovery hub ────────────────

_YAHOO_RACE_URL = "https://keiba.yahoo.co.jp/race/denma/{race_id}/"
_YAHOO_NEWS_SEARCH = "https://news.yahoo.co.jp/search?p={q}&ei=UTF-8"


def collect_yahoo_links(race_id: str, race_name: str = "") -> dict:
    """Discover news/column/video links relevant to a race via Yahoo!競馬.

    Yahoo!競馬 itself is used ONLY as a discovery hub — we pull out the
    anchor targets for downstream tier-3 article fetching. We do not
    extract opinions from Yahoo's own page text.
    """
    items = 0
    status = "ok"
    error = None
    links = {"news": [], "column": [], "video": []}
    try:
        soup = scraper._get(_YAHOO_RACE_URL.format(race_id=race_id), delay=0.8)
        if soup:
            for a in soup.select("a[href]"):
                href = a.get("href", "")
                if not href:
                    continue
                full = urljoin(_YAHOO_RACE_URL, href)
                if "news.yahoo.co.jp" in full:
                    links["news"].append(full)
                    items += 1
                elif "column" in full.lower() or "article" in full.lower():
                    links["column"].append(full)
                    items += 1
                elif "video" in full.lower() or "douga" in full.lower():
                    links["video"].append(full)
                    items += 1
        # Limit to 6 per category — Yahoo pages have a lot of chrome
        for k in links:
            links[k] = list(dict.fromkeys(links[k]))[:6]
    except Exception as e:
        status = "failed"
        error = str(e)

    # No facts yet; links will be fed to collect_text_observations
    return {
        "source": "yahoo", "status": status,
        "facts": [], "items_seen": items, "error": error,
        "links": links,
    }


# ── Tier 3: Article text collection → observations ─────

def _text_from_url(url: str, encoding: str | None = None) -> str:
    soup = scraper._get(url, encoding=encoding, delay=0.8)
    if not soup:
        return ""
    for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)


def collect_text_observations(
    urls: Iterable[str],
    horse_names: list[str],
    source_tag: str,
) -> dict:
    """Fetch a batch of article URLs, extract per-horse canonical facts.

    For each article, the text is split into sentences; any sentence
    mentioning a horse by name becomes a candidate for that horse's
    fact list. Opinion sentences are rejected upstream in the extractor.
    """
    all_facts: list[Fact] = []
    items = 0
    errors: list[str] = []

    for url in urls:
        try:
            text = _text_from_url(url)
            if not text or len(text) < 40:
                continue
            items += 1
            # Split into sentences and assign to horses by name mention
            for sentence in re.split(r"[。．\n\r]+", text):
                sentence = sentence.strip()
                if not sentence or len(sentence) < 8:
                    continue
                mentioned = [n for n in horse_names if n and n in sentence]
                target_horse = mentioned[0] if len(mentioned) == 1 else None
                # If 0 mentions → skip (race-level fact extraction could go
                # here but risk is too high of grabbing unrelated text).
                # If 2+ mentions → still extract, horse=None (race-level)
                if not mentioned:
                    continue
                facts = extract_canonical_facts(sentence, source_tag,
                                                 horse=target_horse)
                all_facts.extend(facts)
        except Exception as e:
            errors.append(f"{url}: {e}")
            continue

    status = "ok" if items > 0 else "skipped"
    if errors and not all_facts:
        status = "failed"
    return {
        "source": source_tag, "status": status,
        "facts": all_facts, "items_seen": items,
        "error": "; ".join(errors[:3]) if errors else None,
    }


# ── Tier 3 specialized: KeibaLab 馬体FOCUS ─────────────

def collect_keibalab_facts(race_id: str, horse_names: list[str]) -> dict:
    """Specialized adapter for KeibaLab 馬体FOCUS / observation pages.

    KeibaLab publishes per-race observation articles at
      https://www.keibalab.jp/db/race/{race_id}/focus/
    The HTML structure is often loose, so we reuse the generic
    article extractor with source_tag='keibalab'.
    """
    url = f"https://www.keibalab.jp/db/race/{race_id}/focus/"
    return collect_text_observations([url], horse_names, source_tag="keibalab")


# ── Tier 3 specialized: sports newspaper race pages ──
# Each adapter targets a best-known URL per newspaper. Structures change
# often — the generic text extractor is resilient to that.

def _fetch_and_crawl_landing(
    landing_url: str,
    encoding: str | None = None,
    article_filter: str = "",
    max_articles: int = 10,
) -> list[str]:
    """Discover article links from a newspaper keiba landing page.

    These sports papers don't expose stable per-race URLs; they
    publish daily articles that mention recent races. We fetch the
    landing page and return candidate article URLs that look like
    keiba articles. Per-race filtering happens later via horse-name
    matching in `collect_text_observations`.
    """
    soup = scraper._get(landing_url, encoding=encoding, delay=0.8)
    if not soup:
        return []
    seen: list[str] = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if not href:
            continue
        full = urljoin(landing_url, href)
        # Keep article-like links in the horseracing section
        if article_filter and article_filter not in full:
            continue
        if full in seen:
            continue
        # Keep only article-shaped URLs (usually have date segments or IDs)
        if re.search(r"/20\d{2}/\d{2}/|/article/|\d{7,}", full):
            seen.append(full)
            if len(seen) >= max_articles:
                break
    return seen


def collect_hochi_facts(race_id: str, horse_names: list[str],
                        race_name: str = "") -> dict:
    """Sports Hochi — umatoku.hochi.co.jp is the keiba subdomain. The
    base landing page links to recent race articles. On historical
    backtest these are typically empty; live prediction ± 1 day gets
    real observations.
    """
    urls = _fetch_and_crawl_landing(
        "https://umatoku.hochi.co.jp/",
        article_filter="umatoku.hochi.co.jp",
        max_articles=8,
    )
    if not urls:
        return {"source": "hochi", "status": "skipped", "facts": [],
                "items_seen": 0, "error": "no article links found"}
    return collect_text_observations(urls, horse_names, source_tag="hochi")


def collect_sanspo_facts(race_id: str, horse_names: list[str],
                         race_name: str = "") -> dict:
    """Sankei Sports (sanspo.com) — only the /race/article/general/...
    paths. Deliberately SKIPS deep.race.sanspo.com/yosou/ (prediction/
    opinion pages) and /race/keiba/yosou/ to stay on the fact-only
    side of the source trust policy.
    """
    urls = _fetch_and_crawl_landing(
        "https://www.sanspo.com/race/",
        article_filter="/race/article/general/",
        max_articles=8,
    )
    if not urls:
        return {"source": "sanspo", "status": "skipped", "facts": [],
                "items_seen": 0, "error": "no article links found"}
    return collect_text_observations(urls, horse_names, source_tag="sanspo")


def collect_daily_facts(race_id: str, horse_names: list[str],
                        race_name: str = "") -> dict:
    """Daily Sports (daily.co.jp) — /horse/ section publishes dated
    articles at /horse/YYYY/MM/DD/NNNNNNNNNN.shtml."""
    urls = _fetch_and_crawl_landing(
        "https://www.daily.co.jp/horse/",
        article_filter="daily.co.jp/horse/",
        max_articles=8,
    )
    if not urls:
        return {"source": "daily", "status": "skipped", "facts": [],
                "items_seen": 0, "error": "no article links found"}
    return collect_text_observations(urls, horse_names, source_tag="daily")


# ── Tier 3 specialized: Paddock-comment text (already scraped) ──

def collect_paddock_observation_facts(
    race_id: str, horse_names: list[str], cached_horses: list[dict],
) -> dict:
    """Use already-cached paddock text from scraper cache.

    The enrich_entries pipeline populates h['paddock_comment']; we run
    the canonical extractor on whatever is there. Source tag is
    'news' (the paddock texts come from various news-desk contributors).
    """
    all_facts: list[Fact] = []
    items = 0
    for h in cached_horses:
        text = h.get("paddock_comment") or ""
        name = (h.get("name") or "").strip()
        if not text or not name or name not in horse_names:
            continue
        items += 1
        # Restrict per-horse: find the horse's sentence segment
        # (reuses paddock_features.extract_per_horse_comments if available).
        # For simplicity here we pass the full text with horse=name; the
        # extractor will apply the opinion blacklist per sentence.
        facts = extract_canonical_facts(text, "news", horse=name)
        all_facts.extend(facts)
    return {
        "source": "news", "status": "ok" if items else "skipped",
        "facts": all_facts, "items_seen": items, "error": None,
    }
