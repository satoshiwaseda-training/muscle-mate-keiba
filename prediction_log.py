"""Live prediction persistence + trigger KPIs.

File: data/live_predictions.json — one entry per race keyed by race_id.

Each entry holds the full dual-mode prediction, the per-horse trigger
table, and (after the race runs) the attached result block with winner
and payouts. KPIs are computed on demand.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
LIVE_FILE = ROOT / "data" / "live_predictions.json"


def _load() -> dict:
    if not LIVE_FILE.exists():
        return {}
    try:
        return json.loads(LIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(data: dict) -> None:
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    LIVE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                         encoding="utf-8")


# ── Public API ────────────────────────────────────────

def store_prediction(prediction: dict) -> None:
    """Persist a prediction keyed by race_id. Does not overwrite the
    `result` block if one is already attached (so re-running prediction
    after result is known doesn't drop the ground truth)."""
    if not prediction or "race_id" not in prediction:
        return
    data = _load()
    rid = prediction["race_id"]
    existing_result = data.get(rid, {}).get("result")
    entry = dict(prediction)
    if existing_result:
        entry["result"] = existing_result
    else:
        entry.setdefault("result", None)
    data[rid] = entry
    _save(data)


def attach_result(race_id: str, result: dict) -> bool:
    """Attach the actual race outcome to a logged prediction.

    `result` shape (can be produced from scraper.fetch_result_netkeiba):
      {
        "finishing_order": [{"rank": 1, "name": "...", "odds": 2.5}, ...],
        "payouts": {"単勝": int, "馬連": int, ...}
      }
    """
    data = _load()
    if race_id not in data:
        return False
    entry = data[race_id]
    entry["result"] = result
    entry["result_attached_at"] = datetime.now().isoformat()
    data[race_id] = entry
    _save(data)
    return True


def list_predictions(
    only_with_result: bool = False,
    only_live: bool = False,
    since: Optional[str] = None,
) -> list[dict]:
    data = _load()
    out = []
    for rid, entry in data.items():
        if only_with_result and not entry.get("result"):
            continue
        if only_live and not entry.get("is_live"):
            continue
        if since and entry.get("race_date", "") < since:
            continue
        out.append(entry)
    out.sort(key=lambda e: e.get("race_date", ""), reverse=True)
    return out


def get_prediction(race_id: str) -> Optional[dict]:
    return _load().get(race_id)


# ── KPI computation ────────────────────────────────────

def _norm(n): return (n or "").strip()


def _parse_odds(raw) -> float:
    try:
        return float(str(raw).replace("---", "0").replace("--", "0"))
    except (ValueError, TypeError):
        return 0.0


def _race_winner(fo: list) -> Optional[str]:
    for h in fo or []:
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return _norm(h.get("name"))
        except ValueError:
            continue
    return None


def compute_kpis(preds: Optional[list[dict]] = None) -> dict:
    """Return the full KPI panel.

    Contains THREE completely independent blocks:
      • strict-trigger KPIs     — existing audit-grade metric
      • baseline ROI            — odds-favorite + model top-1 for
                                  comparison against any strategy
      • loose-trigger KPIs      — experimental betting rule

    Only predictions with an attached `result` contribute to the
    win-rate / ROI figures. Volume counts (total_triggers,
    loose_bets_total, etc.) use every prediction whether scored or not.
    """
    if preds is None:
        preds = list_predictions()

    n_races = 0
    n_result_races = 0

    # ── STRICT trigger accumulators (unchanged) ──
    total_triggers = 0
    triggers_with_result = 0
    trigger_wins = 0
    trigger_pnl = 0.0
    trigger_cost = 0.0
    trigger_count_histogram = defaultdict(int)

    # ── Baselines on the same pool ──
    odds_fav_wins = 0
    odds_fav_pnl = 0.0
    odds_fav_cost = 0.0
    model_top_wins = 0
    model_top_pnl = 0.0
    model_top_cost = 0.0

    # ── LOOSE bet accumulators (new, parallel) ──
    loose_bets_total = 0
    loose_bet_races_total = 0
    loose_bets_with_result = 0
    loose_bet_win_count = 0
    loose_bet_cost = 0.0
    loose_bet_payout = 0.0

    for e in preds:
        n_races += 1

        triggers = e.get("triggers") or []
        total_triggers += len(triggers)
        trigger_count_histogram[len(triggers)] += 1

        loose = e.get("loose_bets") or []
        if loose:
            loose_bet_races_total += 1
            loose_bets_total += len(loose)

        result = e.get("result") or {}
        fo = result.get("finishing_order") or []
        if not fo:
            continue
        winner = _race_winner(fo)
        if not winner:
            continue

        n_result_races += 1
        win_payout = float((result.get("payouts") or {}).get("単勝", 0) or 0)

        # Baseline: odds-favorite bet on this race
        odds_map = {}
        for h in fo:
            nm = _norm(h.get("name"))
            od = _parse_odds(h.get("odds"))
            if nm and od > 0 and nm not in odds_map:
                odds_map[nm] = od
        odds_fav = min(odds_map, key=lambda n: odds_map[n]) if odds_map else None
        odds_fav_cost += 100.0
        if odds_fav == winner:
            odds_fav_wins += 1
            odds_fav_pnl += win_payout

        # Baseline: flat model top-1 on this race
        ranked = e.get("ranked") or []
        model_top = ranked[0]["name"] if ranked else None
        model_top_cost += 100.0
        if model_top == winner:
            model_top_wins += 1
            model_top_pnl += win_payout

        # STRICT trigger outcomes
        for t in triggers:
            triggers_with_result += 1
            trigger_cost += 100.0
            if t.get("name") == winner:
                trigger_wins += 1
                trigger_pnl += win_payout

        # LOOSE bet outcomes (per-race, per-horse)
        for lb in loose:
            loose_bets_with_result += 1
            loose_bet_cost += 100.0
            if lb.get("name") == winner:
                loose_bet_win_count += 1
                loose_bet_payout += win_payout

    # ── Derived ratios ──
    def _safe_ratio(num, den):
        return (num / den) if den else 0.0

    trigger_rate_per_50 = (total_triggers / n_races * 50) if n_races else 0.0
    trigger_win_rate = _safe_ratio(trigger_wins, triggers_with_result)
    trigger_roi = _safe_ratio(trigger_pnl - trigger_cost, trigger_cost)

    odds_fav_roi = _safe_ratio(odds_fav_pnl - odds_fav_cost, odds_fav_cost)
    model_top_roi = _safe_ratio(model_top_pnl - model_top_cost, model_top_cost)

    loose_win_rate = _safe_ratio(loose_bet_win_count, loose_bets_with_result)
    loose_roi = _safe_ratio(loose_bet_payout - loose_bet_cost, loose_bet_cost)

    return {
        # --- Shared ---
        "n_races": n_races,
        "n_races_with_result": n_result_races,

        # --- STRICT trigger block (unchanged) ---
        "total_triggers": total_triggers,
        "triggers_per_50_races": round(trigger_rate_per_50, 2),
        "triggers_with_result": triggers_with_result,
        "trigger_wins": trigger_wins,
        "trigger_win_rate": round(trigger_win_rate, 4),
        "trigger_cost_yen": trigger_cost,
        "trigger_payout_yen": trigger_pnl,
        "trigger_pnl_yen": trigger_pnl - trigger_cost,
        "trigger_roi": round(trigger_roi, 4),
        "trigger_count_histogram": dict(sorted(trigger_count_histogram.items())),

        # --- Baselines ---
        "odds_favorite_win_rate": round(_safe_ratio(odds_fav_wins, n_result_races), 4),
        "odds_favorite_roi": round(odds_fav_roi, 4),
        "model_top_win_rate": round(_safe_ratio(model_top_wins, n_result_races), 4),
        "model_top_roi": round(model_top_roi, 4),

        # --- LOOSE bet block (new, experimental) ---
        "loose_bets_total": loose_bets_total,
        "loose_bet_races_total": loose_bet_races_total,
        "loose_bets_with_result": loose_bets_with_result,
        "loose_bet_win_count": loose_bet_win_count,
        "loose_bet_win_rate": round(loose_win_rate, 4),
        "loose_bet_cost_yen": loose_bet_cost,
        "loose_bet_payout_yen": loose_bet_payout,
        "loose_bet_pnl_yen": loose_bet_payout - loose_bet_cost,
        "loose_bet_roi": round(loose_roi, 4),
        "loose_vs_odds_roi_delta": round(loose_roi - odds_fav_roi, 4),
        "loose_vs_model_roi_delta": round(loose_roi - model_top_roi, 4),
        "loose_rule_version": "cons>=1_comp>=0.60_odds<=15_no_strongneg_v1",
    }


def recent_trigger_table(limit: int = 20) -> list[dict]:
    """Return the most recent STRICT trigger horses with their outcome
    if known."""
    rows = []
    for e in list_predictions()[:limit * 3]:
        triggers = e.get("triggers") or []
        if not triggers:
            continue
        result = e.get("result") or {}
        winner = _race_winner(result.get("finishing_order") or [])
        for t in triggers:
            rows.append({
                "race_id": e.get("race_id", ""),
                "race_name": e.get("race_name", ""),
                "race_date": e.get("race_date", ""),
                "horse": t["name"],
                "consensus_count": t["consensus_count"],
                "composite_condition": t["composite_condition"],
                "source_count": t["source_count"],
                "odds": t.get("odds", 0),
                "outcome": ("WIN" if winner and t["name"] == winner
                            else "LOSS" if winner else "PENDING"),
                "facts": t.get("facts_preview", []),
            })
    return rows[:limit]


def recent_loose_bets_table(limit: int = 20) -> list[dict]:
    """Return the most recent LOOSE bet horses with their outcome if known.

    Each row also carries payout / pnl so the UI can show 100-yen flat
    betting profitability per bet.
    """
    rows = []
    for e in list_predictions()[:limit * 3]:
        loose = e.get("loose_bets") or []
        if not loose:
            continue
        result = e.get("result") or {}
        fo = result.get("finishing_order") or []
        winner = _race_winner(fo)
        payouts = result.get("payouts") or {}
        win_payout = float(payouts.get("単勝", 0) or 0)
        for lb in loose:
            is_win = bool(winner and lb["name"] == winner)
            payout = win_payout if is_win else 0
            pnl = payout - 100
            rows.append({
                "race_date": e.get("race_date", ""),
                "race_id": e.get("race_id", ""),
                "race_name": e.get("race_name", ""),
                "horse": lb["name"],
                "odds": lb.get("odds", 0),
                "consensus_count": lb.get("consensus_count"),
                "composite_condition": lb.get("composite_condition"),
                "source_count": lb.get("source_count"),
                "loose_trigger_reason": lb.get("loose_trigger_reason", ""),
                "outcome": "WIN" if is_win else ("LOSS" if winner else "PENDING"),
                "payout": payout,
                "pnl": pnl if winner else 0,
            })
    return rows[:limit]
