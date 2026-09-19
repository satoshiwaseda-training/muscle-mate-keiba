"""Weekend full analysis — one-shot script.

Runs all steps in sequence:
  1. Generate predictions for Saturday + Sunday
  2. Attach results for both days
  3. Generate weekly report
  4. Print detailed analysis

Usage:
  python tools/run_weekend_analysis.py
  python tools/run_weekend_analysis.py --date 2026-04-11   # specify Saturday date
"""

from __future__ import annotations

import datetime as dt
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools._autolog_utils import ensure_project_on_path, log, last_weekend

ensure_project_on_path()

import scraper
import live_pipeline as lp
import prediction_log as plog


def _predict_day(race_date: dt.date) -> dict:
    """Run predictions for all races on a given date."""
    log(f"=== Predicting {race_date.isoformat()} ===")
    races = scraper.fetch_race_list(race_date)
    log(f"  Found {len(races)} races")

    stats = {"total": len(races), "success": 0, "early": 0, "failed": 0, "errors": []}

    for i, race in enumerate(races):
        rid = race.get("race_id", "")
        rname = race.get("race_name", "?")
        venue = race.get("venue", "")
        log(f"  [{i+1}/{len(races)}] {rname} ({rid})")

        try:
            result = lp.predict_live(
                race_id=rid,
                venue=venue,
                race_name=rname,
                race_date=race_date.isoformat(),
                auto_log=True,
            )
            stage = result.get("prediction_stage", "?")
            odds_status = result.get("odds_status", "?")
            log(f"    -> stage={stage}, odds={odds_status}")
            if stage == "final":
                stats["success"] += 1
            else:
                stats["early"] += 1
        except Exception as e:
            log(f"    -> ERROR: {e}")
            stats["failed"] += 1
            stats["errors"].append(f"{rname}: {e}")
        time.sleep(0.5)

    return stats


def _attach_day(race_date: dt.date) -> dict:
    """Attach results for a given date."""
    log(f"=== Attaching results for {race_date.isoformat()} ===")
    all_preds = plog.list_predictions(only_live=True)
    target = [p for p in all_preds if p.get("race_date") == race_date.isoformat()]
    need = [p for p in target if not p.get("result")]
    log(f"  {len(target)} predictions, {len(need)} need results")

    stats = {"total": len(target), "need": len(need), "attached": 0, "failed": 0}

    for p in need:
        rid = p.get("race_id", "")
        try:
            res = scraper.fetch_result_netkeiba(rid)
            if res and res.get("finishing_order"):
                plog.attach_result(rid, res)
                stats["attached"] += 1
                log(f"  {p.get('race_name','?')}: attached")
            else:
                stats["failed"] += 1
                log(f"  {p.get('race_name','?')}: no result yet")
        except Exception as e:
            stats["failed"] += 1
            log(f"  {p.get('race_name','?')}: ERROR {e}")
        time.sleep(1.0)

    return stats


def _analyze() -> dict:
    """Full analysis of this weekend's predictions."""
    sat, sun = last_weekend(dt.date.today())
    sat_iso, sun_iso = sat.isoformat(), sun.isoformat()

    all_preds = plog.list_predictions(only_live=True)
    weekend = [p for p in all_preds if sat_iso <= (p.get("race_date") or "") <= sun_iso]
    with_result = [p for p in weekend if (p.get("result") or {}).get("finishing_order")]

    # Stage counts
    by_stage = {}
    for p in weekend:
        s = p.get("prediction_stage", "unknown")
        by_stage[s] = by_stage.get(s, 0) + 1

    # Basic metrics
    wins = 0
    cost = 0.0
    payout = 0.0
    market_fav_wins = 0
    model_fav_count = 0

    # Confidence/odds band analysis
    conf_bands = {"0-30": [0, 0], "30-50": [0, 0], "50-70": [0, 0], "70+": [0, 0]}
    odds_bands = {"1-3": [0, 0], "3-5": [0, 0], "5-10": [0, 0], "10-30": [0, 0], "30+": [0, 0]}

    # Stage-level metrics
    stage_wins = {"early": [0, 0], "final": [0, 0]}  # [wins, total]

    # Anomaly detection
    anomalies = []

    for p in with_result:
        ranked = p.get("ranked") or []
        if not ranked:
            continue

        top1 = ranked[0]
        top1_name = (top1.get("name") or "").strip()
        top1_odds = float(top1.get("odds") or 0)
        top1_conf = float(top1.get("win_prob", 0)) * 100  # percent

        res = p.get("result") or {}
        fo = res.get("finishing_order") or []

        # Find winner
        winner = None
        winner_odds = 0
        for h in fo:
            try:
                if int(h.get("rank", 0) or 0) == 1:
                    winner = (h.get("name") or "").strip()
                    try:
                        winner_odds = float(str(h.get("odds", 0)).replace(",", ""))
                    except ValueError:
                        pass
                    break
            except ValueError:
                pass
        if not winner:
            continue

        model_fav_count += 1
        is_win = (top1_name == winner)
        cost += 100.0

        stage = p.get("prediction_stage", "unknown")
        if stage in stage_wins:
            stage_wins[stage][1] += 1
            if is_win:
                stage_wins[stage][0] += 1

        if is_win:
            wins += 1
            try:
                pay = float(str((res.get("payouts") or {}).get("単勝", 0)).replace(",", ""))
                payout += pay
            except Exception:
                pass

        # Market favorite (lowest odds in finishing order = most popular)
        if winner_odds > 0:
            min_odds = 9999
            market_fav_name = ""
            for h in fo:
                try:
                    ho = float(str(h.get("odds", 9999)).replace(",", ""))
                    if 0 < ho < min_odds:
                        min_odds = ho
                        market_fav_name = (h.get("name") or "").strip()
                except (ValueError, TypeError):
                    pass
            if market_fav_name == winner:
                market_fav_wins += 1

        # Confidence band
        if top1_conf < 30:
            band = "0-30"
        elif top1_conf < 50:
            band = "30-50"
        elif top1_conf < 70:
            band = "50-70"
        else:
            band = "70+"
        conf_bands[band][1] += 1
        if is_win:
            conf_bands[band][0] += 1

        # Odds band
        if top1_odds <= 0:
            pass
        elif top1_odds < 3:
            ob = "1-3"
        elif top1_odds < 5:
            ob = "3-5"
        elif top1_odds < 10:
            ob = "5-10"
        elif top1_odds < 30:
            ob = "10-30"
        else:
            ob = "30+"
        if top1_odds > 0:
            odds_bands[ob][1] += 1
            if is_win:
                odds_bands[ob][0] += 1

        # Anomaly: high confidence miss
        if top1_conf > 70 and not is_win:
            anomalies.append({
                "type": "high_conf_miss",
                "race": p.get("race_name", "?"),
                "model_pick": top1_name,
                "confidence": f"{top1_conf:.1f}%",
                "winner": winner,
            })

        # Anomaly: very low odds (strong favorite) miss
        if top1_odds > 0 and top1_odds <= 2.0 and not is_win:
            anomalies.append({
                "type": "strong_fav_miss",
                "race": p.get("race_name", "?"),
                "model_pick": top1_name,
                "odds": f"{top1_odds:.1f}",
                "winner": winner,
            })

    roi = ((payout - cost) / cost * 100) if cost > 0 else 0
    hit_rate = (wins / model_fav_count * 100) if model_fav_count > 0 else 0
    market_rate = (market_fav_wins / model_fav_count * 100) if model_fav_count > 0 else 0

    return {
        "period": f"{sat_iso} ~ {sun_iso}",
        "volume": {
            "total_weekend": len(weekend),
            "saturday": sum(1 for p in weekend if p.get("race_date") == sat_iso),
            "sunday": sum(1 for p in weekend if p.get("race_date") == sun_iso),
            "with_result": len(with_result),
            "attach_rate_pct": (len(with_result) / len(weekend) * 100) if weekend else 0,
            "by_stage": by_stage,
        },
        "headline": {
            "win_hit_rate": f"{hit_rate:.1f}%",
            "roi": f"{roi:+.1f}%",
            "cost": f"{cost:.0f}",
            "payout": f"{payout:.0f}",
            "wins": wins,
            "evaluated": model_fav_count,
        },
        "stage_analysis": {
            k: {
                "total": v[1],
                "wins": v[0],
                "win_rate": f"{v[0]/v[1]*100:.1f}%" if v[1] > 0 else "N/A",
            }
            for k, v in stage_wins.items()
        },
        "market_comparison": {
            "model_win_rate": f"{hit_rate:.1f}%",
            "market_fav_win_rate": f"{market_rate:.1f}%",
            "diff": f"{hit_rate - market_rate:+.1f}pp",
        },
        "confidence_bands": {
            k: {"total": v[1], "wins": v[0], "rate": f"{v[0]/v[1]*100:.1f}%" if v[1] > 0 else "N/A"}
            for k, v in conf_bands.items()
        },
        "odds_bands": {
            k: {"total": v[1], "wins": v[0], "rate": f"{v[0]/v[1]*100:.1f}%" if v[1] > 0 else "N/A"}
            for k, v in odds_bands.items()
        },
        "anomalies": anomalies[:10],
    }


def _print_report(a: dict) -> None:
    """Print formatted analysis report."""
    print("\n" + "=" * 60)
    print("  WEEKEND ANALYSIS REPORT")
    print(f"  Period: {a['period']}")
    print("=" * 60)

    v = a["volume"]
    h = a["headline"]
    print(f"""
### 今週サマリ
- total races:    {v['total_weekend']}
  - Saturday:     {v['saturday']}
  - Sunday:       {v['sunday']}
- with_result:    {v['with_result']}
- attach_rate:    {v['attach_rate_pct']:.1f}%
- by_stage:       {v['by_stage']}
- win hit rate:   {h['win_hit_rate']}  ({h['wins']}/{h['evaluated']})
- ROI:            {h['roi']}  (cost={h['cost']}, payout={h['payout']})
""")

    sa = a["stage_analysis"]
    print("### ステージ分析")
    for stage, d in sa.items():
        print(f"  {stage:8s}: {d['wins']}/{d['total']} = {d['win_rate']}")

    mc = a["market_comparison"]
    print(f"""
### 市場比較
- model win rate:       {mc['model_win_rate']}
- market fav win rate:  {mc['market_fav_win_rate']}
- diff:                 {mc['diff']}
""")

    print("### confidence帯別成績")
    for band, d in a["confidence_bands"].items():
        print(f"  {band:8s}: {d['wins']}/{d['total']} = {d['rate']}")

    print("\n### odds帯別成績")
    for band, d in a["odds_bands"].items():
        print(f"  {band:8s}: {d['wins']}/{d['total']} = {d['rate']}")

    if a["anomalies"]:
        print(f"\n### 異常検知 ({len(a['anomalies'])} 件)")
        for an in a["anomalies"]:
            print(f"  [{an['type']}] {an['race']}: pick={an.get('model_pick','?')} -> winner={an.get('winner','?')}")
    else:
        print("\n### 異常検知: なし")

    # Verdict
    v_total = v["total_weekend"]
    attach = v["attach_rate_pct"]
    hit = float(h["win_hit_rate"].rstrip("%")) if h["win_hit_rate"] != "N/A" else 0
    roi_val = float(h["roi"].rstrip("%").lstrip("+")) if h["roi"] != "N/A" else 0

    print("\n### 結論")
    if v_total == 0:
        verdict = "評価不可"
        reason = "予測データがゼロ件。パイプラインが動いていない可能性。"
    elif attach < 50:
        verdict = "評価不可"
        reason = f"結果付与率が低すぎる ({attach:.0f}%)。レースが終了していないか、attach_results が未実行。"
    elif hit >= 25 and roi_val > -20:
        verdict = "機能している"
        reason = f"勝率 {hit:.1f}% (基準25%+), ROI {roi_val:+.1f}% は許容範囲内。"
    elif hit >= 15 or roi_val > -50:
        verdict = "部分的に機能"
        reason = f"勝率 {hit:.1f}%, ROI {roi_val:+.1f}%。改善余地あり。"
    else:
        verdict = "機能していない"
        reason = f"勝率 {hit:.1f}%, ROI {roi_val:+.1f}%。根本的な見直しが必要。"

    print(f"- verdict: {verdict}")
    print(f"- 理由: {reason}")
    print("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weekend full analysis")
    parser.add_argument("--date", help="Saturday date (YYYY-MM-DD)")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction step")
    parser.add_argument("--skip-attach", action="store_true", help="Skip result attachment")
    args = parser.parse_args()

    today = dt.date.today()
    sat, sun = last_weekend(today)

    if args.date:
        sat = dt.datetime.strptime(args.date, "%Y-%m-%d").date()
        sun = sat + dt.timedelta(days=1)

    log(f"Target weekend: {sat.isoformat()} (Sat) ~ {sun.isoformat()} (Sun)")

    # Step 1 & 2: Predict
    if not args.skip_predict:
        log("--- Step 1: Generating predictions ---")
        for d in [sat, sun]:
            stats = _predict_day(d)
            log(f"  {d}: total={stats['total']} success={stats['success']} early={stats['early']} failed={stats['failed']}")
    else:
        log("--- Step 1: SKIPPED (--skip-predict) ---")

    # Step 3: Attach results
    if not args.skip_attach:
        log("--- Step 2: Attaching results ---")
        for d in [sat, sun]:
            stats = _attach_day(d)
            log(f"  {d}: attached={stats['attached']}/{stats['need']}")
    else:
        log("--- Step 2: SKIPPED (--skip-attach) ---")

    # Step 4: Weekly report (if tool exists)
    log("--- Step 3: Weekly report ---")
    try:
        from tools import weekly_report
        # Just import, don't run - analysis below is more detailed
    except Exception:
        pass

    # Step 5: Analysis
    log("--- Step 4: Analysis ---")
    analysis = _analyze()

    # Save analysis
    out_dir = Path("data/weekend_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{sat.isoformat()}_to_{sun.isoformat()}.json"
    out_file.write_text(json.dumps(analysis, ensure_ascii=False, indent=2))
    log(f"Analysis saved to {out_file}")

    # Print report
    _print_report(analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
