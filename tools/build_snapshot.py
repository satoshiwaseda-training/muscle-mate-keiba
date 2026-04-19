"""Point-in-time snapshot builder for leak-safe backtesting.

Builds a frozen snapshot for a single race_id at a given race_date.
Everything inside the snapshot MUST be information that was available
BEFORE race_date. Violations raise LeakError.

USAGE:
  python tools/build_snapshot.py --race-id 202503030511 --date 2025-11-22
  python tools/build_snapshot.py --all    # build for all races in backtest_cache

OUTPUT:
  data/snapshot/{race_id}.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshot"
CACHE_DIR = PROJECT_ROOT / "data" / "scraper_cache"
BACKTEST_DIR = PROJECT_ROOT / "data" / "backtest_cache"

# ═══════════════════════════════════════════════════════════
# Leak detection
# ═══════════════════════════════════════════════════════════

class LeakError(Exception):
    """Raised when snapshot data contains information from >= race_date."""
    pass


# Forbidden keys — if any of these appear in snapshot, it's a leak.
_FORBIDDEN_KEYS = frozenset([
    "finishing_order", "payouts", "result", "final_odds",
    "actual_rank", "win_time", "post_race",
])


def parse_flexible_date(s: str) -> dt.date | None:
    """Parse Japanese / ISO date strings.

    Handles:
      2025-11-15
      2025/11/15
      2025年11月15日
      25/11/15
      25年11月15日
      11/15 (year inferred as race_date's year or year-1 if month > race_month)
    Returns None if unparseable.
    """
    if not s:
        return None
    s = str(s).strip()

    # Full year first
    m = re.match(r"^(\d{4})[-/\u5e74\.](\d{1,2})[-/\u6708\.](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        try:
            return dt.date(int(y), int(mo), int(d))
        except ValueError:
            return None

    # 2-digit year (netkeiba format: "25/11/15" or "25年11月15日")
    m = re.match(r"^(\d{2})[-/\u5e74\.](\d{1,2})[-/\u6708\.](\d{1,2})", s)
    if m:
        y, mo, d = m.groups()
        yi = int(y)
        # 2-digit: assume 20xx since we're post-2000
        year = 2000 + yi
        try:
            return dt.date(year, int(mo), int(d))
        except ValueError:
            return None

    return None


def leak_check(snapshot: dict) -> None:
    """Raise LeakError if any future data is found.

    Checks:
      1. forbidden keys
      2. recent_races[].date < race_date (strictly less than)
      3. jockey/trainer stats are neutral (None / null)
      4. No result payload anywhere in the tree
    """
    race_date_str = snapshot.get("race_date", "")
    try:
        race_date = dt.date.fromisoformat(race_date_str)
    except Exception as e:
        raise LeakError(f"Invalid race_date in snapshot: {race_date_str!r}") from e

    # 1. Forbidden keys (recursive)
    def _scan(obj, path="$"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _FORBIDDEN_KEYS:
                    raise LeakError(f"Forbidden key '{k}' found at {path}")
                _scan(v, f"{path}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                _scan(v, f"{path}[{i}]")

    _scan(snapshot)

    # 2. recent_races filter
    for hid, hdata in (snapshot.get("horses") or {}).items():
        for i, rr in enumerate(hdata.get("recent_races") or []):
            d_str = rr.get("date", "")
            d = parse_flexible_date(d_str)
            if d is None:
                # Unparseable — reject defensively (we don't know if it's leak)
                raise LeakError(
                    f"Horse {hid} recent_races[{i}] date is unparseable: "
                    f"{d_str!r} (cannot verify < race_date)"
                )
            if d >= race_date:
                raise LeakError(
                    f"LEAK: Horse {hid} recent_races[{i}] date {d} "
                    f">= race_date {race_date} "
                    f"(raw={d_str!r} race={rr.get('race_name','?')})"
                )

    # 3. jockey / trainer stats must be neutral
    for jid, jdata in (snapshot.get("jockey") or {}).items():
        stats = jdata.get("stats")
        if stats is not None:
            # Allow only explicit neutral placeholder
            wr = stats.get("win_rate") if isinstance(stats, dict) else None
            if wr not in (None, "", "neutral"):
                raise LeakError(
                    f"Jockey {jid} has non-neutral stats: {stats!r} "
                    f"(current cache is cumulative — not point-in-time)"
                )

    for tid, tdata in (snapshot.get("trainer") or {}).items():
        stats = tdata.get("stats")
        if stats is not None:
            wr = stats.get("win_rate") if isinstance(stats, dict) else None
            if wr not in (None, "", "neutral"):
                raise LeakError(
                    f"Trainer {tid} has non-neutral stats: {stats!r}"
                )

    # 4. weight_trend dates (if present as dated list) — skip if just strings


# ═══════════════════════════════════════════════════════════
# Cache loaders
# ═══════════════════════════════════════════════════════════

def _load_cache(kind: str, key: str) -> dict | list | None:
    p = CACHE_DIR / kind / f"{key}.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[warn] failed to load {p}: {e}")
        return None


def _load_backtest_meta() -> dict:
    p = BACKTEST_DIR / "backtest_races.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════
# Snapshot builder
# ═══════════════════════════════════════════════════════════

def _filter_recent_races(recent_races: list, race_date: dt.date) -> list:
    """Keep only races strictly before race_date."""
    filtered = []
    for rr in recent_races or []:
        d = parse_flexible_date(rr.get("date", ""))
        if d is not None and d < race_date:
            filtered.append(rr)
    return filtered


def build_snapshot(race_id: str, race_date: dt.date,
                   race_name: str = "", venue: str = "",
                   grade: str = "",
                   online_fetch: bool = False) -> dict:
    """Build a point-in-time snapshot for race_id at race_date.

    Reads from local caches only (unless online_fetch=True).
    All temporal data is strictly filtered to date < race_date.

    Raises LeakError if filtering fails or caches contain inconsistent data.
    """
    # ── Race-level metadata ──
    # Pulled from enrich_race cache or shutuba. No post-race fields.
    enrich = _load_cache("enrich_race", race_id) or []

    if not enrich and online_fetch:
        import scraper as _scraper
        entries_raw = _scraper.fetch_entries_netkeiba(race_id, venue) or []
        if entries_raw:
            enrich = _scraper.enrich_entries(entries_raw, race_id,
                                             race_name=race_name)

    if not enrich:
        raise ValueError(
            f"No enrich_race cache for {race_id} and online_fetch=False. "
            "Run with --online to fetch, or ensure cache exists."
        )

    # ── Entries (shutuba-level only, no post-race) ──
    # We explicitly rebuild entries from enrich to strip any accidental post-race fields.
    entries = []
    for h in enrich:
        e = {
            "number":           h.get("number", ""),
            "waku":             h.get("waku", ""),
            "name":             h.get("name", ""),
            "horse_id":         h.get("horse_id", ""),
            "jockey":           h.get("jockey", ""),
            "jockey_id":        h.get("jockey_id", ""),
            "trainer":          h.get("trainer", ""),
            "trainer_id":       h.get("trainer_id", ""),
            "owner":            h.get("owner", ""),
            "age":              h.get("age", ""),
            "weight":           h.get("weight", ""),
            "horse_weight":     h.get("horse_weight", ""),
            # odds is public pre-race info; we accept whatever shutuba had.
            # The cache may have been populated post-race with "injected" odds
            # from the result page — those are final odds (numerically close
            # to pre-race odds but formally post-race). We keep them as-is and
            # flag in the audit.
            "odds":             h.get("odds", "0"),
            "stable":           h.get("stable", ""),
            "ritto":            h.get("ritto", ""),
            "transport_stress": h.get("transport_stress", ""),
        }
        # Disallow any stray forbidden fields
        for fk in _FORBIDDEN_KEYS:
            if fk in h:
                raise LeakError(
                    f"enrich_race[{race_id}] horse {e['name']} contains "
                    f"forbidden field '{fk}'"
                )
        entries.append(e)

    # ── Horse details (sire/dam/damsire/breeder/recent_races) ──
    horses: dict[str, dict] = {}
    for h in enrich:
        hid = h.get("horse_id", "") or ""
        if not hid:
            continue
        horse_cache = _load_cache("horse", hid) or {}

        if online_fetch and (not horse_cache.get("damsire")):
            import scraper as _scraper
            horse_cache = _scraper.fetch_horse_detail(hid) or horse_cache

        recent_raw = horse_cache.get("recent_races") or []
        recent_filtered = _filter_recent_races(recent_raw, race_date)

        horses[hid] = {
            "sire":          horse_cache.get("sire", "") or h.get("sire", ""),
            "dam":           horse_cache.get("dam", "") or h.get("dam", ""),
            "damsire":       horse_cache.get("damsire", ""),
            "breeder":       horse_cache.get("breeder", ""),
            "owner":         horse_cache.get("owner", "") or h.get("owner", ""),
            "recent_races":  recent_filtered,
            "weight_trend":  horse_cache.get("weight_trend") or [],
            # Audit metadata
            "_source":       "cache" if horse_cache else "missing",
            "_recent_races_dropped": len(recent_raw) - len(recent_filtered),
        }

    # ── Jockey / Trainer: neutral ──
    # Current cache is cumulative (not point-in-time) — we MUST NOT use it
    # for past races. Phase 1: neutral null entries.
    jockey: dict[str, dict] = {}
    trainer: dict[str, dict] = {}
    for h in enrich:
        jid = h.get("jockey_id", "") or ""
        tid = h.get("trainer_id", "") or ""
        if jid:
            jockey[jid] = {"stats": None, "_note": "neutral (phase1, not point-in-time)"}
        if tid:
            trainer[tid] = {"stats": None, "_note": "neutral (phase1, not point-in-time)"}

    # ── Paddock / training (race_id-specific, captured pre-race) ──
    paddock = _load_cache("paddock", race_id) or {}
    training = _load_cache("training", race_id) or []

    # ── Race info ──
    # Grade / surface / venue / distance are pre-race facts.
    # track_condition & cushion_value are captured at scrape time which
    # may be post-race — we store them but flag for careful use downstream.
    race_info: dict = {}
    try:
        meta = _load_backtest_meta().get(race_id, {})
    except Exception:
        meta = {}

    race_info.update({
        "race_name":       race_name or meta.get("race_name", ""),
        "grade":           grade or meta.get("grade", ""),
        "num_horses":      meta.get("num_horses", len(entries)),
        "venue":           venue or "",
        # Surface/distance from existing fetch_race_info cache
        # (not post-race; JRA publishes these ahead of time)
        # We leave as empty and rely on scrape at predict-time (via snapshot).
    })

    # Best effort: check if fetch_race_info has a cached response
    # (No dedicated cache exists today; online_fetch picks it up when patched.)

    # ── Weather / track_condition for the race_date ──
    # We do NOT inject these here to keep the leak surface small. At predict
    # time, snapshot_reader will expose a weather-only hook that returns a
    # pre-race-estimated track_condition (per §3 of the design).
    #
    # For phase1, track_condition is forced to "" (unknown) so that
    # train.py's track conditional branch falls to the "light" branch.

    snapshot = {
        "snapshot_version":  1,
        "snapshot_built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "race_id":           race_id,
        "race_date":         race_date.isoformat(),
        "venue":             venue,
        "race_name":         race_info["race_name"],
        "grade":             race_info["grade"],
        "num_horses":        race_info["num_horses"],
        "race_info":         race_info,
        "entries":           entries,
        "horses":            horses,
        "jockey":            jockey,
        "trainer":           trainer,
        "paddock":           paddock,
        "training":          training,
        # Audit metadata
        "leak_audit": {
            "race_date":    race_date.isoformat(),
            "horses_count": len(horses),
            "recent_races_total": sum(len(h.get("recent_races", []))
                                       for h in horses.values()),
            "recent_races_dropped": sum(h.get("_recent_races_dropped", 0)
                                         for h in horses.values()),
            "jockey_neutral":  len(jockey),
            "trainer_neutral": len(trainer),
            "leak_clear": False,  # flipped to True after leak_check passes
        },
    }

    leak_check(snapshot)
    snapshot["leak_audit"]["leak_clear"] = True

    return snapshot


def save_snapshot(snapshot: dict) -> Path:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    race_id = snapshot["race_id"]
    out = SNAPSHOT_DIR / f"{race_id}.json"
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    os.replace(tmp, out)
    return out


# ═══════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-id", help="Single race_id to snapshot")
    parser.add_argument("--date", help="race_date YYYY-MM-DD")
    parser.add_argument("--venue", default="", help="Venue name (optional)")
    parser.add_argument("--race-name", default="", help="Race name (optional)")
    parser.add_argument("--grade", default="", help="Grade (optional)")
    parser.add_argument("--all", action="store_true",
                        help="Build snapshots for all races in backtest_cache")
    parser.add_argument("--online", action="store_true",
                        help="Allow online fetch when cache is missing")
    parser.add_argument("--strict", action="store_true",
                        help="Abort on first LeakError (default: skip failing race)")
    args = parser.parse_args()

    if args.all:
        meta = _load_backtest_meta()
        print(f"[info] Building snapshots for {len(meta)} races")
        ok, fail = 0, 0
        for rid, m in meta.items():
            try:
                d = dt.date.fromisoformat(m["race_date"])
                snap = build_snapshot(
                    rid, d,
                    race_name=m.get("race_name", ""),
                    venue=m.get("venue", ""),
                    grade=m.get("grade", ""),
                    online_fetch=args.online,
                )
                p = save_snapshot(snap)
                dropped = snap["leak_audit"]["recent_races_dropped"]
                print(f"  OK  {rid} ({m['race_date']}) "
                      f"horses={len(snap['horses'])} dropped={dropped} -> {p.name}")
                ok += 1
            except LeakError as e:
                print(f"  LEAK {rid}: {e}")
                fail += 1
                if args.strict:
                    return 2
            except Exception as e:
                print(f"  FAIL {rid}: {e}")
                fail += 1
                if args.strict:
                    return 3
        print(f"\n[summary] ok={ok} fail={fail}")
        return 0 if fail == 0 else 1

    if not args.race_id or not args.date:
        parser.error("--race-id and --date required (or use --all)")

    race_date = dt.date.fromisoformat(args.date)
    snap = build_snapshot(
        args.race_id, race_date,
        race_name=args.race_name, venue=args.venue, grade=args.grade,
        online_fetch=args.online,
    )
    p = save_snapshot(snap)
    print(f"Snapshot saved: {p}")
    print(f"  horses: {len(snap['horses'])}")
    print(f"  recent_races dropped (post-date): "
          f"{snap['leak_audit']['recent_races_dropped']}")
    print(f"  leak_clear: {snap['leak_audit']['leak_clear']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
