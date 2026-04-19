"""Diagnose 単勝 odds for a specific race across all sources.

Usage:
  python tools/diagnose_odds.py --race-id 202604010811
  python tools/diagnose_odds.py --date 2026-04-19 --venue 中山 --race 11
  python tools/diagnose_odds.py --race-id 202604010811 --enable-yahoo
  python tools/diagnose_odds.py --race-id 202604010811 --with-horse-names

What it does:
  1. Calls netkeiba API, (optionally) JRA 公式, (optionally) Yahoo 競馬
  2. Prints per-source, per-horse odds side-by-side
  3. Flags any source whose values hit the sanity bound (rejects shown)
  4. Computes consensus primary + disagreements
  5. Optionally fetches shutuba entries so horse names are shown

This is the tool to reach for when you see an obviously-wrong odds on
the live panel. Run it once and you'll know whether it's:
  - a specific source parsing the wrong column (rejected list will light up)
  - source disagreement (consensus disagreement summary will show)
  - netkeiba API returning something unexpected (raw_reason will explain)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import odds_sources as osrc  # noqa: E402


VENUE_NAME_TO_CODE = {v: k for k, v in osrc.RACE_ID_VENUE_TO_NAME.items()}


def build_race_id(date_iso: str, venue_name: str, race_no: int) -> str:
    """Construct a netkeiba-style 12-digit race_id.

    Args:
      date_iso:    "YYYY-MM-DD"
      venue_name:  "中山" / "東京" / ...
      race_no:     1..12
    """
    y, m, d = date_iso.split("-")
    vcode = VENUE_NAME_TO_CODE.get(venue_name)
    if not vcode:
        raise ValueError(f"unknown venue: {venue_name}")
    # 注意: kaisai / kaisai_day はレース日カレンダに依存するため、
    # 素直に復元できない。ここでは "01/01" のダミー値を置き、
    # 明確に注意喚起するコメントを残す。
    # 本当の race_id を得るには netkeiba の race_list から引く必要がある。
    kaisai = "01"
    day = "01"
    return f"{y}{vcode}{kaisai}{day}{race_no:02d}"


def _fetch_horse_names(race_id: str) -> dict:
    """Return {馬番: 馬名} from shutuba, for display alignment."""
    try:
        import scraper
        entries = scraper.fetch_entries_netkeiba(race_id) or []
        out = {}
        for e in entries:
            try:
                num = int(str(e.get("number", "")).strip() or 0)
            except ValueError:
                continue
            nm = (e.get("name") or "").strip()
            if num and nm:
                out[num] = nm
        return out
    except Exception as e:
        print(f"[diagnose] shutuba fetch failed: {e}")
        return {}


def _fmt_odds(v) -> str:
    if v is None:
        return "   —  "
    try:
        return f"{float(v):>6.1f}"
    except (TypeError, ValueError):
        return "   ?  "


def print_report(consensus: dict, horse_names: dict) -> None:
    per_source = consensus.get("per_source") or {}
    primary = consensus.get("primary_source")
    sources = list(per_source.keys())

    print("")
    print("=" * 78)
    print(f"PRIMARY SOURCE: {primary}")
    print(f"fetched_at: {consensus.get('fetched_at')}")
    print(f"enabled sources: {', '.join(consensus.get('enabled_sources') or [])}")
    print("=" * 78)

    # Per-source meta
    print("\n── Per-source status ──")
    for src in sources:
        r = per_source[src]
        status = r.get("status")
        n = len(r.get("by_number") or {})
        rej = r.get("rejected") or {}
        flag = "" if not rej else f"  ⚠ rejected: {rej}"
        reason = r.get("raw_reason") or ""
        schema = r.get("schema_guess") or ""
        print(f"  {src:15s}  status={status:<16} n={n:>2}  schema={schema}{flag}")
        if reason:
            print(f"                   raw_reason: {reason}")

    # Per-horse comparison
    print("\n── Per-horse odds comparison ──")
    header = f"  {'#':>2}  {'name':<20s}  "
    for src in sources:
        header += f"{src:>14s}  "
    header += "disagree"
    print(header)

    # Union of all horse numbers across sources + entries
    nums = set()
    for r in per_source.values():
        nums.update((r.get("by_number") or {}).keys())
    nums.update(horse_names.keys())
    for num in sorted(nums):
        name = (horse_names.get(num) or "?").ljust(20)[:20]
        row = f"  {num:>2d}  {name}  "
        for src in sources:
            v = (per_source[src].get("by_number") or {}).get(num)
            row += f"{_fmt_odds(v)}         "[:16]
        dgr = (consensus.get("disagreements") or {}).get(num) or {}
        pct = dgr.get("max_pct") or 0.0
        fmt_flag = " ⚠" if dgr.get("flag") else "  "
        row += f"{pct*100:>5.1f}% {fmt_flag}"
        print(row)

    print("\n── Summary ──")
    print(f"  has_disagreement_any: {consensus.get('has_disagreement_any')}")
    print(f"  summary:              {osrc.summarize_disagreement(consensus)}")

    # If every source rejected values, tell the user loudly
    total_rejected = sum(len(r.get("rejected") or {}) for r in per_source.values())
    if total_rejected:
        print(f"\n  ⚠ Total rejected-by-sanity values: {total_rejected}")
        print(f"    これが 0 でない場合、そのソースは odds ではない値を")
        print(f"    返してきています（斤量・賞金・払戻金など）。")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--race-id", help="netkeiba 12-digit race_id")
    parser.add_argument("--date", help="race date YYYY-MM-DD (+ --venue + --race)")
    parser.add_argument("--venue", help="会場名 (東京/中山/阪神/etc)")
    parser.add_argument("--race", type=int, help="レース番号 1-12")
    parser.add_argument("--enable-yahoo", action="store_true",
                        help="Yahoo 競馬も叩く (default: off)")
    parser.add_argument("--disable-jra", action="store_true",
                        help="JRA 公式をスキップ")
    parser.add_argument("--with-horse-names", action="store_true", default=True,
                        help="shutuba から馬名を取ってきて横並び (default: on)")
    parser.add_argument("--no-horse-names", dest="with_horse_names",
                        action="store_false")
    parser.add_argument("--raw", action="store_true",
                        help="生 JSON を出力 (構造化レポートの代わり)")
    args = parser.parse_args()

    race_id = args.race_id
    if not race_id:
        if not (args.date and args.venue and args.race):
            parser.error("--race-id か (--date --venue --race) を指定してください")
        race_id = build_race_id(args.date, args.venue, args.race)
        print(f"[diagnose] constructed race_id={race_id} "
              f"(NOTE: kaisai/day は 01/01 のダミー — "
              f"正しい race_id が必要なら直接指定を)")

    print(f"[diagnose] fetching consensus for race_id={race_id}")
    consensus = osrc.fetch_odds_consensus(
        race_id=race_id,
        enable_jra=not args.disable_jra,
        enable_yahoo=args.enable_yahoo,
    )

    if args.raw:
        print(json.dumps(consensus, ensure_ascii=False, indent=2, default=str))
        return 0

    horse_names = _fetch_horse_names(race_id) if args.with_horse_names else {}
    print_report(consensus, horse_names)
    return 0


if __name__ == "__main__":
    sys.exit(main())
