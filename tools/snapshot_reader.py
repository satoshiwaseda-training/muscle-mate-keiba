"""Snapshot reader — scraper-compatible API backed by frozen snapshot JSON.

Provides a `patch_scraper(snapshot)` context manager that monkey-patches
all scraper network calls to read from the given snapshot instead.

This is the ONLY way leak-safe backtest prediction should invoke
live_pipeline.predict_live — it guarantees no network calls escape
and no post-race data is pulled in.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scraper


class SnapshotScraperError(Exception):
    """Raised when snapshot lookup fails and we refuse to fall back to network."""
    pass


# ═══════════════════════════════════════════════════════════
# Factory: build snapshot-backed replacement functions
# ═══════════════════════════════════════════════════════════

def make_snapshot_scraper(snapshot: dict) -> dict:
    """Return a dict of scraper-attribute -> replacement-function mappings."""

    race_id_target = snapshot["race_id"]
    race_date = snapshot["race_date"]
    entries = snapshot.get("entries", [])
    horses = snapshot.get("horses", {}) or {}
    jockey_db = snapshot.get("jockey", {}) or {}
    trainer_db = snapshot.get("trainer", {}) or {}
    paddock_db = snapshot.get("paddock", {}) or {}
    training_db = snapshot.get("training", []) or []
    race_info_stored = snapshot.get("race_info", {}) or {}

    # ── fetch_entries_netkeiba ──
    def _fetch_entries(race_id, venue=""):
        if race_id != race_id_target:
            raise SnapshotScraperError(
                f"snapshot is for {race_id_target}, "
                f"got request for {race_id}"
            )
        # Return entries JOINED with snapshot.horses data so sire/dam/
        # damsire/breeder reach feature_store. live_pipeline.predict_live
        # does not call scraper.enrich_entries, so feature_store reads
        # sire directly from the entry dict — we must pre-join here.
        out = []
        for e in entries:
            hid = e.get("horse_id", "")
            horse_data = horses.get(hid, {}) if hid else {}
            out.append({
                **e,
                # Bloodline (leak-safe: static, from snapshot.horses)
                "sire":       horse_data.get("sire", ""),
                "dam":        horse_data.get("dam", ""),
                "damsire":    horse_data.get("damsire", ""),
                "breeder":    horse_data.get("breeder", ""),
                # owner: prefer snapshot.horses value (from horse detail)
                # but keep entry's value if snapshot is empty
                "owner":      e.get("owner") or horse_data.get("owner", ""),
                # Enrichment defaults (filled later by bridge / fact layer)
                "recent_form": "",
                "bloodline":  (f"\u7236:{horse_data.get('sire','?')} "
                                f"\u6bcd:{horse_data.get('dam','?')}"
                                if horse_data.get("sire") else ""),
                "weight_trend": "",
                # Jockey / trainer stats: neutral (leak-safe)
                "jockey_win_rate": "",
                "jockey_g1_wins": "",
                "trainer_win_rate": "",
                # Training & paddock: empty defaults
                "training_eval": "",
                "training_physics": {"final_split": 0.0,
                                      "acceleration_rate": 0.0,
                                      "cardio_index": 0.0},
                "training_nlp": {},
                "paddock_scores": {},
                "best_weight_analysis": {},
                "transport_profile": {},
            })
        return out

    # ── fetch_horse_detail ──
    def _fetch_horse_detail(horse_id):
        h = horses.get(horse_id)
        if not h:
            return {"horse_id": horse_id, "recent_races": [], "sire": "",
                    "dam": "", "damsire": "", "breeder": "", "owner": "",
                    "weight_trend": []}
        return {
            "horse_id": horse_id,
            "sire":     h.get("sire", ""),
            "dam":      h.get("dam", ""),
            "damsire":  h.get("damsire", ""),
            "breeder":  h.get("breeder", ""),
            "owner":    h.get("owner", ""),
            "recent_races": h.get("recent_races") or [],
            "weight_trend": h.get("weight_trend") or [],
        }

    # ── jockey stats: neutral ──
    def _fetch_jockey_stats(jockey_id):
        # Always return empty/neutral. This sets jockey_win_rate to 0,
        # which feature_store._parse_percentage maps to 0.0 — a constant
        # that doesn't vary by horse, so it doesn't change relative ranking.
        return {"jockey_id": jockey_id,
                "win_rate": "", "place_rate": "",
                "g1_wins": "", "single_recovery": ""}

    def _fetch_trainer_stats(trainer_id):
        return {"trainer_id": trainer_id,
                "win_rate": "", "place_rate": ""}

    # ── training times (race_id specific, pre-race) ──
    def _fetch_training_times(race_id):
        if race_id != race_id_target:
            return []
        return training_db

    # ── paddock reports ──
    def _fetch_paddock_reports(race_id, horse_names=None, race_name=""):
        if race_id != race_id_target:
            return {}
        return paddock_db

    # ── race info ──
    def _fetch_race_info_netkeiba(race_id):
        if race_id != race_id_target:
            return {}
        # Produce a dict with the fields live_pipeline expects.
        # Only pre-race safe values: grade (published ahead), surface/distance
        # (published ahead), weather/temperature/track_condition are EMPTY
        # to avoid using post-race-captured values.
        return {
            "grade":           race_info_stored.get("grade", "")
                               or snapshot.get("grade", ""),
            "race_name":       race_info_stored.get("race_name", "")
                               or snapshot.get("race_name", ""),
            "surface":         race_info_stored.get("surface", ""),
            "distance":        race_info_stored.get("distance", ""),
            # Empty = unknown. train.py falls to default branches.
            "track_condition": "",
            "weather":         "",
            "temperature":     "",
            "cushion_value":   "",
        }

    # ── result / live odds: hard-blocked ──
    def _fetch_result_netkeiba(race_id):
        raise SnapshotScraperError(
            f"fetch_result_netkeiba called during backtest for {race_id} — "
            "LEAK BLOCKED. Snapshots must not pull results."
        )

    def _fetch_odds_netkeiba(race_id):
        # Pre-race odds API — allowed to return "not published" to trigger
        # early-stage behavior. But we don't serve live odds either, so
        # we return a not-published placeholder.
        return {"status": "not-published",
                "http_status": 0, "response_url": "",
                "raw_reason": "snapshot-backed (no live odds)",
                "parse_error": "", "schema_version_guess": "snapshot",
                "fetched_at": "", "update_count": 0, "official_time": ""}

    def _fetch_jra_race_changes(race_id):
        return {}

    # ── cache loader (for enrich_race, enrich_stats, etc.) ──
    original_cache_load = scraper._cache_load

    def _cache_load(kind, key):
        # For the current race, return pre-filtered snapshot-backed data.
        if key != race_id_target:
            # Allow other keys (shouldn't normally be hit for a single-race predict)
            return original_cache_load(kind, key)
        if kind == "enrich_race":
            # Return raw entries shape — live_pipeline will re-enrich from
            # scraper patches, then cache. We serve None so enrich runs fresh.
            return None
        if kind == "paddock":
            return paddock_db or None
        if kind == "training":
            return training_db or None
        if kind == "horse":
            return None  # defer to _fetch_horse_detail via patches
        if kind == "jockey":
            return None
        if kind == "enrich_stats":
            return None
        return original_cache_load(kind, key)

    def _cache_save(kind, key, data):
        # No-op during backtest (we don't want to pollute the shared cache
        # with snapshot-backed enrichments).
        return

    return {
        "fetch_entries_netkeiba":    _fetch_entries,
        "fetch_horse_detail":        _fetch_horse_detail,
        "fetch_jockey_stats":        _fetch_jockey_stats,
        "fetch_trainer_stats":       _fetch_trainer_stats,
        "fetch_training_times":      _fetch_training_times,
        "fetch_paddock_reports":     _fetch_paddock_reports,
        "fetch_race_info_netkeiba":  _fetch_race_info_netkeiba,
        "fetch_result_netkeiba":     _fetch_result_netkeiba,
        "fetch_odds_netkeiba":       _fetch_odds_netkeiba,
        "fetch_jra_race_changes":    _fetch_jra_race_changes,
        "_cache_load":               _cache_load,
        "_cache_save":               _cache_save,
        "_cached_horse_detail":      _fetch_horse_detail,
        "_cached_jockey_stats":      _fetch_jockey_stats,
        "_cached_training_times":    _fetch_training_times,
        "_cached_paddock_reports":   _fetch_paddock_reports,
    }


# ═══════════════════════════════════════════════════════════
# Context manager
# ═══════════════════════════════════════════════════════════

def _blocked_collector(name):
    """Return a stub collector that emits no facts.

    Used to prevent live_pipeline from hitting keibalab / hochi / sanspo /
    daily (which are post-race articles for past races, irrelevant to
    snapshot-backed prediction) during backtest.
    """
    def _stub(*args, **kwargs):
        return {"source": name, "status": "skipped",
                "facts": [], "items_seen": 0,
                "error": "snapshot-backed backtest — external article sources blocked"}
    return _stub


@contextlib.contextmanager
def patch_scraper(snapshot: dict):
    """Monkey-patch scraper + fact_collectors to serve snapshot only.

    All network calls are blocked. All post-race fetches are blocked.
    Inside the context, live_pipeline.predict_live() reads purely from
    snapshot.

    Usage:
      with patch_scraper(snap):
          result = live_pipeline.predict_live(race_id, venue, race_name,
                                               race_date, auto_log=False)
    """
    replacements = make_snapshot_scraper(snapshot)
    originals = {}
    for name, fn in replacements.items():
        if hasattr(scraper, name):
            originals[name] = getattr(scraper, name)
            setattr(scraper, name, fn)

    # Also block external article collectors. These make network calls to
    # keibalab/hochi/sanspo/daily which are:
    #   (a) post-race articles for past dates → potentially leak-unsafe
    #   (b) return 404 for most race_ids → slow retries dominate runtime
    # Block them outright during backtest. collect_jra_facts is kept
    # because it re-uses our patched scraper for race_info/entries.
    import fact_collectors as fc
    fc_blocks = ["collect_keibalab_facts", "collect_hochi_facts",
                 "collect_sanspo_facts", "collect_daily_facts"]
    fc_originals = {}
    for name in fc_blocks:
        if hasattr(fc, name):
            fc_originals[name] = getattr(fc, name)
            setattr(fc, name, _blocked_collector(name.replace("collect_", "").replace("_facts", "")))

    try:
        yield
    finally:
        for name, orig in originals.items():
            setattr(scraper, name, orig)
        for name, orig in fc_originals.items():
            setattr(fc, name, orig)


# ═══════════════════════════════════════════════════════════
# Snapshot loader with leak re-validation
# ═══════════════════════════════════════════════════════════

def load_snapshot(race_id: str) -> dict:
    """Load snapshot from disk and re-run leak_check.

    Raises FileNotFoundError if snapshot missing.
    Raises build_snapshot.LeakError if leak is detected.
    """
    import json
    p = Path(__file__).resolve().parent.parent / "data" / "snapshot" / f"{race_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"No snapshot: {p}")
    with open(p, "r", encoding="utf-8") as f:
        snap = json.load(f)

    # Re-validate (defensive: snapshot on disk may have been edited)
    from tools.build_snapshot import leak_check
    leak_check(snap)

    return snap
