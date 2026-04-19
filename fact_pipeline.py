"""Orchestration: JRA-first collection → merge → per-horse aggregation →
bio-injection into score_runner → Top-3 selection.

One public entry point:

    run(race_id, venue, race_name, use_detail_sources=True,
        use_news_supplements=True, progress_cb=None) -> dict

Returns a dict with:
  - facts:           flattened merged fact list
  - per_horse_facts: {name: [Fact, ...]}
  - per_horse_score: {name: aggregate_horse_score dict}
  - trust_grade:     "A" | "B" | "C" | "D"
  - coverage:        {"jra": float, "supplemental": float, "consensus_count": int}
  - ranked:          per-horse ranked list with win probabilities
  - selected_top3:   top-3 set selected under current α/β
  - collection_log:  list of provenance dicts from each collector
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import scraper
import feature_store as fs
import probability_engine as pe
from train import score_runner
from data_store import load_weights
from fact_schema import Fact
from fact_extractor import merge_fact_layers, aggregate_horse_score
import fact_collectors as fc


ROOT = Path(__file__).parent
FACT_STORE = ROOT / "data" / "facts"


def _log(cb: Optional[Callable[[str], None]], msg: str):
    if cb:
        cb(msg)


def _trust_grade(jra_cov: float, supp_cov: float, consensus_count: int) -> str:
    """Trust grade rubric.

      A: JRA coverage ≥ 0.9 AND (consensus ≥ 5 OR supplemental ≥ 0.4)
      B: JRA coverage ≥ 0.7
      C: JRA coverage ≥ 0.4
      D: JRA coverage < 0.4
    """
    if jra_cov >= 0.9 and (consensus_count >= 5 or supp_cov >= 0.4):
        return "A"
    if jra_cov >= 0.7:
        return "B"
    if jra_cov >= 0.4:
        return "C"
    return "D"


def _group_by_horse(facts: list[Fact]) -> dict[str | None, list[Fact]]:
    out: dict[str | None, list[Fact]] = defaultdict(list)
    for f in facts:
        out[f.horse].append(f)
    return out


def run(
    race_id: str,
    venue: str = "",
    race_name: str = "",
    use_detail_sources: bool = True,
    use_news_supplements: bool = True,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> dict:
    collection_log: list[dict] = []

    # ── Tier 1: JRA ───────────────────────────────────
    _log(progress_cb, "Tier 1: JRA 公式情報を収集中")
    t0 = time.time()
    jra = fc.collect_jra_facts(race_id, venue)
    jra["elapsed_s"] = round(time.time() - t0, 1)
    collection_log.append(jra)

    # We need horse names for downstream tiers — get them from fresh entries
    entries = scraper.fetch_entries_netkeiba(race_id, venue) or []
    horse_names = [(e.get("name") or "").strip() for e in entries if e.get("name")]

    # Scratch-filter the horse_names for downstream (don't fetch per-horse
    # facts for scratched horses).
    scratched = {f.horse for f in jra["facts"] if f.type == "scratched"}
    running_names = [n for n in horse_names if n not in scratched]

    # ── Tier 2: Yahoo hub ─────────────────────────────
    yahoo_links = []
    if use_detail_sources:
        _log(progress_cb, "Tier 2: Yahoo!競馬 からリンクを収集")
        t0 = time.time()
        y = fc.collect_yahoo_links(race_id, race_name)
        y["elapsed_s"] = round(time.time() - t0, 1)
        collection_log.append(y)
        yahoo_links = (y.get("links", {}).get("news", [])
                       + y.get("links", {}).get("column", []))

    # ── Tier 3: article text observations ────────────
    article_facts: list[Fact] = []
    if use_detail_sources:
        _log(progress_cb, "Tier 3: KeibaLab 観察情報を取得")
        t0 = time.time()
        kl = fc.collect_keibalab_facts(race_id, running_names)
        kl["elapsed_s"] = round(time.time() - t0, 1)
        collection_log.append(kl)
        article_facts.extend(kl["facts"])

    if use_news_supplements and yahoo_links:
        _log(progress_cb, f"Tier 3: 記事本文を取得 ({len(yahoo_links)}件)")
        t0 = time.time()
        news = fc.collect_text_observations(yahoo_links, running_names, "news")
        news["elapsed_s"] = round(time.time() - t0, 1)
        collection_log.append(news)
        article_facts.extend(news["facts"])

    # ── Tier 3 (cached): paddock-comment text from existing enrich cache ──
    cached_race = scraper._cache_load("enrich_race", race_id)
    if cached_race:
        _log(progress_cb, "Tier 3: キャッシュ済みパドック観察を統合")
        pd = fc.collect_paddock_observation_facts(race_id, running_names, cached_race)
        collection_log.append(pd)
        article_facts.extend(pd["facts"])

    # ── Merge ─────────────────────────────────────────
    _log(progress_cb, "ファクトをマージ中 (consensus bonus)")
    merged = merge_fact_layers(jra["facts"], article_facts)

    # ── Per-horse aggregation ────────────────────────
    _log(progress_cb, "horse 単位で集約")
    by_horse = _group_by_horse(merged)
    per_horse_score: dict[str, dict] = {}
    for name in running_names:
        per_horse_score[name] = aggregate_horse_score(by_horse.get(name, []))

    # ── Coverage + trust grade ───────────────────────
    jra_horses = {f.horse for f in jra["facts"] if f.horse}
    supp_horses = {f.horse for f in article_facts if f.horse}
    n_running = max(len(running_names), 1)
    jra_cov = len(jra_horses & set(running_names)) / n_running
    supp_cov = len(supp_horses & set(running_names)) / n_running
    consensus_count = sum(
        1 for f in merged
        if f.meta.get("n_sources", 1) >= 2
    )
    grade = _trust_grade(jra_cov, supp_cov, consensus_count)

    # ── Scoring: run score_runner with fact-augmented bio layer ──
    _log(progress_cb, "スコア計算中 (score_runner)")
    race_info = scraper.fetch_race_info_netkeiba(race_id) or {}
    # Keep entries that are still running
    entries_run = [e for e in entries if (e.get("name") or "").strip() in set(running_names)]

    sf = fs.extract_structured_features(
        entries=entries_run,
        race_info=race_info,
        track_condition=race_info.get("track_condition", ""),
        weather=race_info.get("weather", ""),
        temperature=race_info.get("temperature", ""),
        cushion_value=race_info.get("cushion_value", ""),
        venue=venue,
    )
    # Inject fact-composite into paddock pathway (only where empty)
    sf_horses = sf.get("horses") or {}
    for name, h in sf_horses.items():
        c = per_horse_score.get(name, {}).get("composite_condition", 0.5)
        centered = round(2.0 * c - 1.0, 3)
        for key in ("paddock_gait", "paddock_hindquarter", "paddock_vascularity"):
            if not isinstance(h.get(key), (int, float)) or h.get(key) == 0:
                h[key] = centered
    sf["horses"] = sf_horses

    ctx = {"weights": load_weights()}
    grade_str = race_info.get("grade", "")
    scored = []
    entry_names = list(sf_horses.keys())
    for this_name in entry_names:
        rotated = [{
            "name": this_name, "rank": 1,
            "odds": sf_horses[this_name].get("odds", 0),
            "confidence": 0, "ev_gap": 0, "bet": "",
        }]
        for other in entry_names:
            if other == this_name:
                continue
            rotated.append({
                "name": other, "rank": 2,
                "odds": sf_horses[other].get("odds", 0),
                "confidence": 0, "ev_gap": 0, "bet": "",
            })
        feat = {"grade": grade_str, "num_horses": len(entry_names),
                "horse_features": rotated, "structured_features": sf}
        s = score_runner(feat, ctx).get("top_confidence", 50.0)
        scored.append({
            "name": this_name,
            "odds": sf_horses[this_name].get("odds", 0),
            "score": float(s),
        })
    ranked = pe.assign_win_probs(scored, temperature=pe.DEFAULT_TEMPERATURE)
    sel = pe.select_top3(ranked, alpha=pe.DEFAULT_ALPHA, beta=pe.DEFAULT_BETA)

    # Override decision
    if ranked:
        running_ranked = [r for r in ranked if r["odds"] > 0]
        if running_ranked:
            odds_fav = min(running_ranked, key=lambda r: r["odds"])
            model_top = ranked[0]
            allowed, reason = pe.should_override_market(
                selected_top={**model_top, "confirmed_running": True},
                odds_favorite=odds_fav,
                feature_coverage=max(jra_cov, supp_cov),
            )
        else:
            allowed, reason = False, "no-running-horses"
            odds_fav = ranked[0]
    else:
        allowed, reason = False, "empty-ranking"
        odds_fav = None

    # ── Persist ───────────────────────────────────────
    _log(progress_cb, "ファクトを保存")
    _persist(race_id, race_name, merged, collection_log, ranked, sel, grade,
             jra_cov, supp_cov, consensus_count)

    return {
        "race_id": race_id,
        "race_name": race_name,
        "facts": [f.to_dict() for f in merged],
        "per_horse_facts": {
            name: [f.to_dict() for f in by_horse.get(name, [])]
            for name in running_names
        },
        "per_horse_score": per_horse_score,
        "trust_grade": grade,
        "coverage": {
            "jra": round(jra_cov, 3),
            "supplemental": round(supp_cov, 3),
            "consensus_count": consensus_count,
        },
        "ranked": ranked,
        "selected_top3": sel["selected"],
        "p1": sel["p1"], "p2": sel["p2"],
        "override_decision": {
            "allow": allowed, "reason": reason,
            "model_top": ranked[0]["name"] if ranked else None,
            "odds_fav": odds_fav["name"] if odds_fav else None,
        },
        "scratched": sorted(scratched),
        "n_running": len(running_names),
        "collection_log": collection_log,
    }


def _persist(
    race_id: str,
    race_name: str,
    merged_facts: list[Fact],
    collection_log: list[dict],
    ranked: list[dict],
    sel: dict,
    grade: str,
    jra_cov: float,
    supp_cov: float,
    consensus_count: int,
) -> None:
    FACT_STORE.mkdir(parents=True, exist_ok=True)
    payload = {
        "race_id": race_id,
        "race_name": race_name,
        "saved_at": datetime.now().isoformat(),
        "trust_grade": grade,
        "coverage": {
            "jra": round(jra_cov, 3),
            "supplemental": round(supp_cov, 3),
            "consensus_count": consensus_count,
        },
        "facts": [f.to_dict() for f in merged_facts],
        "ranked": ranked,
        "selected_top3": sel.get("selected", []),
        "p1": sel.get("p1", 0), "p2": sel.get("p2", 0),
        "collection_log": [
            {k: v for k, v in rec.items() if k != "facts"}
            for rec in collection_log
        ],
    }
    out = FACT_STORE / f"{race_id}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8")
