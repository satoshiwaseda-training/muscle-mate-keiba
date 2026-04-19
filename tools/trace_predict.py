"""Full-pipeline trace for a single race.

One command that runs `predict_live` and dumps EVERY intermediate stage
of the odds flow so we can find exactly where 182/724/794 come from.

Usage:
  python tools/trace_predict.py --race-id 202606030811
  python tools/trace_predict.py --race-id 202606030811 --horse フォルテアンジェロ

Output sections:
  [A] scraper.fetch_entries_netkeiba raw values
  [B] consensus per-source (shutuba-direct, netkeiba-api, yahoo, jra)
  [C] consensus primary + disagreements
  [D] entries[*].odds after _inject_odds_if_missing
  [E] sf_horses[name].odds (feature_store output)
  [F] ranked[i].odds (final display value)
  [G] DIFF: any stage where the value changes unexpectedly

Paste this whole output to the bug report — it will pinpoint exactly
which stage corrupts the odds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _fmt_odds(v) -> str:
    try:
        return f"{float(v or 0):>7.2f}"
    except (TypeError, ValueError):
        return "     ??"


def _try_call(fn_desc, fn, *args, **kwargs):
    """Call fn and pretty-print any exception, returning None on failure."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"[trace] {fn_desc} raised: {e.__class__.__name__}: {e}")
        return None


def print_stage_diff(horse_name: str, stages: dict) -> None:
    """Show how a single horse's odds evolved through all pipeline stages."""
    print(f"\n── Pipeline trace for: {horse_name} ──")
    print(f"  [A] scraper.fetch_entries_netkeiba:  {stages.get('A')}")
    print(f"  [B.shutuba-direct] consensus source: {stages.get('B_shutuba')}")
    print(f"  [B.netkeiba-api]   consensus source: {stages.get('B_api')}")
    print(f"  [B.yahoo-keiba]    consensus source: {stages.get('B_yahoo')}")
    print(f"  [C] consensus primary value:         {stages.get('C')}")
    print(f"  [D] entries[*].odds post-inject:     {stages.get('D')}")
    print(f"  [E] sf_horses[name].odds:            {stages.get('E')}")
    print(f"  [F] ranked[i].odds (UI-facing):      {stages.get('F')}")

    # Diff highlights
    seq = [stages.get(k) for k in ("A", "C", "D", "E", "F")]
    distinct = sorted({s for s in seq if s is not None and s != ""})
    if len(distinct) > 1:
        print(f"  [G] ⚠ VALUE CHANGED ACROSS STAGES: {distinct}")
    else:
        print(f"  [G] ✓ consistent value across stages")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--venue", default="")
    parser.add_argument("--race-name", default="")
    parser.add_argument("--race-date", default="")
    parser.add_argument("--horse",
                        help="Focus on a single horse (partial name match)")
    parser.add_argument("--save-json",
                        help="Save full dump as JSON at this path")
    args = parser.parse_args()

    import scraper
    import feature_store as fs
    import odds_sources as osrc
    import live_pipeline as lp

    race_id = args.race_id
    print("")
    print(f"========= TRACE race_id={race_id} =========")

    # ── [A] Raw scraper output ──
    print(f"\n[A] Calling scraper.fetch_entries_netkeiba({race_id!r}) ...")
    entries = _try_call("fetch_entries_netkeiba",
                        scraper.fetch_entries_netkeiba, race_id, args.venue) or []
    print(f"    → {len(entries)} entries")
    for e in entries:
        nm = e.get("name", "?")
        if args.horse and args.horse not in nm:
            continue
        print(f"    #{e.get('number'):>2s} {nm:<18} odds={_fmt_odds(e.get('odds'))} "
              f"horse_id={e.get('horse_id','')}")

    # Snapshot A-stage per horse
    stage_by_horse: dict[str, dict] = {}
    for e in entries:
        nm = (e.get("name") or "").strip()
        if not nm:
            continue
        stage_by_horse[nm] = {"A": e.get("odds"), "number": e.get("number")}

    # ── [B] Consensus per-source ──
    print(f"\n[B] Fetching consensus (with entries as shutuba-direct source) ...")
    consensus = osrc.fetch_odds_consensus(
        race_id=race_id, race_date=args.race_date, venue=args.venue,
        entries=entries, enable_jra=True, enable_yahoo=True,
    )
    per_source = consensus.get("per_source") or {}
    print(f"    enabled_sources: {consensus.get('enabled_sources')}")
    for src_name, src_res in per_source.items():
        print(f"\n    ── [{src_name}] status={src_res.get('status')} "
              f"n={len(src_res.get('by_number') or {})}")
        print(f"        raw_reason: {src_res.get('raw_reason')}")
        print(f"        schema_guess: {src_res.get('schema_guess')}")
        if src_res.get("rejected"):
            print(f"        rejected (sanity out of bounds): {src_res['rejected']}")
        for e in entries:
            nm = (e.get("name") or "").strip()
            if args.horse and args.horse not in nm:
                continue
            try:
                num = int(str(e.get("number", "")).strip() or 0)
            except ValueError:
                num = 0
            v = (src_res.get("by_number") or {}).get(num)
            key = f"B_{src_name.split('-')[0]}"
            stage_by_horse.setdefault(nm, {})[key] = v
            print(f"          #{num:>2d} {nm:<18} → {_fmt_odds(v) if v is not None else '   ---'}")

    # ── [C] Consensus primary ──
    print(f"\n[C] consensus primary_source: {consensus.get('primary_source')!r}")
    print(f"    has_disagreement_any:     {consensus.get('has_disagreement_any')}")
    print(f"    summary:                  {osrc.summarize_disagreement(consensus)}")
    pbn = consensus.get("primary_by_number") or {}
    for e in entries:
        nm = (e.get("name") or "").strip()
        if args.horse and args.horse not in nm:
            continue
        try:
            num = int(str(e.get("number", "")).strip() or 0)
        except ValueError:
            num = 0
        v = pbn.get(num)
        stage_by_horse.setdefault(nm, {})["C"] = v
        print(f"    #{num:>2d} {nm:<18} → primary_by_number = {_fmt_odds(v)}")

    # ── [D] _inject_odds_if_missing (full live-pipeline behaviour) ──
    print(f"\n[D] Running _inject_odds_if_missing to finalize entries[*].odds ...")
    # Make a shallow copy to not affect the earlier state we captured
    entries_copy = [dict(e) for e in entries]
    status, meta = lp._inject_odds_if_missing(
        entries_copy, race_id,
        race_date=args.race_date, venue=args.venue,
    )
    print(f"    status: {status!r}")
    print(f"    pipeline_version: {meta.get('pipeline_version')}")
    print(f"    odds_source: {meta.get('odds_source')}")
    print(f"    consensus_primary: {meta.get('consensus_primary_source')}")
    for e in entries_copy:
        nm = (e.get("name") or "").strip()
        if args.horse and args.horse not in nm:
            continue
        stage_by_horse.setdefault(nm, {})["D"] = e.get("odds")
        print(f"    #{e.get('number'):>2s} {nm:<18} "
              f"odds={_fmt_odds(e.get('odds'))} src={e.get('odds_source')}")

    # ── [E] feature_store → sf_horses ──
    print(f"\n[E] Extracting sf_horses via feature_store ...")
    race_info = {"track_condition": "", "weather": "", "temperature": ""}
    sf = _try_call("feature_store.extract_structured_features",
                   fs.extract_structured_features,
                   entries=entries_copy, race_info=race_info,
                   track_condition="", weather="",
                   temperature="", cushion_value="", venue=args.venue)
    sf_horses = (sf or {}).get("horses") or {}
    for nm, h in sf_horses.items():
        if args.horse and args.horse not in nm:
            continue
        stage_by_horse.setdefault(nm, {})["E"] = h.get("odds")
        print(f"    {nm:<18} sf.odds = {_fmt_odds(h.get('odds'))}")

    # ── [F] Final predict_live — this is what UI shows ──
    print(f"\n[F] Running full predict_live() to capture ranked[i].odds ...")
    result = _try_call("predict_live",
                       lp.predict_live,
                       race_id=race_id, venue=args.venue,
                       race_name=args.race_name,
                       race_date=args.race_date,
                       auto_log=False)
    if result:
        for h in (result.get("ranked") or [])[:18]:
            nm = (h.get("name") or "").strip()
            if args.horse and args.horse not in nm:
                continue
            stage_by_horse.setdefault(nm, {})["F"] = h.get("odds")
            print(f"    ranked: {nm:<18} odds={_fmt_odds(h.get('odds'))} "
                  f"win_prob={h.get('win_prob', 0)*100:.1f}%")

    # ── [G] Pipeline diff per horse ──
    print("\n\n=====================================================")
    print("PIPELINE DIFF — どの段階でオッズが変わったかをハイライト")
    print("=====================================================")
    for nm, stages in stage_by_horse.items():
        if args.horse and args.horse not in nm:
            continue
        print_stage_diff(nm, stages)

    if args.save_json:
        p = Path(args.save_json)
        p.write_text(json.dumps({
            "race_id": race_id,
            "stage_by_horse": stage_by_horse,
            "status": status if 'status' in dir() else None,
            "meta": meta if 'meta' in dir() else None,
            "consensus_primary_source": consensus.get("primary_source"),
            "consensus_per_source_summary": {
                n: {k: r.get(k) for k in ("status", "schema_guess", "rejected", "raw_reason")}
                for n, r in per_source.items()
            },
        }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"\n[trace] full dump saved to: {p}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
