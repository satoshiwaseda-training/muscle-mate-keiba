"""PDCA self-evolution engine.

Extended with odds-bias self-audit: did we over-favour favourites
and miss expected-value opportunities in longer-priced horses?
"""

from data_store import (
    get_prediction, get_result,
    load_weights, save_weights,
    compute_stats, get_recent_miss_summary,
    load_predictions, load_results,
    save_pdca_log, load_pdca_log,
)
import gemini_client

# 1レースあたりの重み変動上限（急激な過学習を防ぐ）
MAX_WEIGHT_DELTA = 0.05


def _apply_weight_damping(old_weights: dict, suggested: dict) -> dict:
    """
    重みの急変を防ぐダンピング処理。
    1レースあたり最大 MAX_WEIGHT_DELTA (5%) までしか変化させない。
    """
    damped = {}
    for k in old_weights:
        old_v = old_weights.get(k, 0.0)
        new_v = suggested.get(k, old_v)
        delta = new_v - old_v
        delta = max(-MAX_WEIGHT_DELTA, min(MAX_WEIGHT_DELTA, delta))
        damped[k] = round(old_v + delta, 4)
    total = sum(damped.values())
    if total > 0:
        damped = {k: round(v / total, 4) for k, v in damped.items()}
    return damped


def compare_and_evolve(race_id: str, api_key: str = "") -> dict:
    """
    Compare prediction vs result, run Gemini reflection (with odds-bias audit),
    and update weights.json.
    """
    prediction = get_prediction(race_id)
    result = get_result(race_id)

    if not prediction:
        return {"error": f"予想データが見つかりません: {race_id}"}
    if not result:
        return {"error": f"結果データが見つかりません: {race_id}"}

    finishing = {h["name"]: h["rank"] for h in result.get("finishing_order", [])}
    pred_horses = prediction.get("horses", [])
    top_pick = pred_horses[0]["name"] if pred_horses else ""

    hit_1st = finishing.get(top_pick) == 1
    hit_top3 = finishing.get(top_pick, 99) <= 3

    # Odds-bias audit: was top pick the lowest-odds (most popular)?
    odds_bias_flag = _check_odds_bias(pred_horses, result)

    old_weights = load_weights()
    reflection_data = {
        "reflection": "",
        "odds_bias_audit": "",
        "key_lessons": [],
        "suggested_weights": old_weights,
        "weight_reasoning": "",
    }

    if api_key:
        reflection_data = gemini_client.generate_reflection(
            api_key=api_key,
            race_name=prediction.get("race_name", race_id),
            prediction=prediction,
            result=result,
            current_weights=old_weights,
        )
        # ダンピング適用後に保存（急激な重み変動を防止）
        raw_suggested = reflection_data.get("suggested_weights", old_weights)
        new_weights = _apply_weight_damping(old_weights, raw_suggested)
        save_weights(new_weights)
        reflection_data["suggested_weights"] = new_weights
    else:
        new_weights = old_weights

    race_name = prediction.get("race_name", race_id)
    evolve_result = {
        "race_id": race_id,
        "race_name": race_name,
        "top_pick": top_pick,
        "hit_1st": hit_1st,
        "hit_top3": hit_top3,
        "odds_bias_flag": odds_bias_flag,
        "reflection": reflection_data.get("reflection", ""),
        "odds_bias_audit": reflection_data.get("odds_bias_audit", ""),
        "key_lessons": reflection_data.get("key_lessons", []),
        "miss_categories": reflection_data.get("miss_categories", {}),
        "weight_reasoning": reflection_data.get("weight_reasoning", ""),
        "old_weights": old_weights,
        "new_weights": new_weights,
    }

    # PDCAログ保存（ダッシュボードの重み推移に使用）
    save_pdca_log(race_id, {
        "race_name": race_name,
        "hit_1st": hit_1st,
        "hit_top3": hit_top3,
        "old_weights": old_weights,
        "new_weights": new_weights,
        "miss_categories": reflection_data.get("miss_categories", {}),
        "key_lessons": reflection_data.get("key_lessons", []),
    })

    return evolve_result


def _check_odds_bias(pred_horses: list[dict], result: dict) -> dict:
    """
    Detect whether prediction was biased toward low-odds (popular) horses.

    Returns:
        {
            "biased": bool,  # True if top pick had the lowest odds
            "top_pick_odds": float,
            "winner_odds": float,
            "missed_ev": bool,  # True if actual winner had higher odds (potential value missed)
        }
    """
    if not pred_horses:
        return {"biased": False, "top_pick_odds": 0, "winner_odds": 0, "missed_ev": False}

    def _parse_odds(s) -> float:
        try:
            return float(str(s).replace("倍", "").strip())
        except Exception:
            return 0.0

    top_pick_odds = _parse_odds(pred_horses[0].get("odds", 0))

    finishing = result.get("finishing_order", [])
    winner = next((h for h in finishing if h.get("rank") == 1), {})
    winner_odds = _parse_odds(winner.get("odds", 0))

    biased = top_pick_odds > 0 and top_pick_odds == min(
        _parse_odds(h.get("odds", 999)) for h in pred_horses if h.get("odds")
    )
    missed_ev = winner_odds > top_pick_odds * 1.5 if winner_odds > 0 and top_pick_odds > 0 else False

    return {
        "biased": biased,
        "top_pick_odds": top_pick_odds,
        "winner_odds": winner_odds,
        "missed_ev": missed_ev,
    }


def get_trend_analysis() -> dict:
    stats = compute_stats()
    misses = get_recent_miss_summary(10)

    miss_reasons = []
    for m in misses:
        pred = m.get("prediction", {})
        result = m.get("result", {})
        pred_top = pred.get("horses", [{}])[0].get("name", "?")
        actual_top = next(
            (h["name"] for h in result.get("finishing_order", []) if h["rank"] == 1), "?"
        )
        miss_reasons.append({
            "race": m.get("race_name", m["race_id"]),
            "predicted": pred_top,
            "actual_winner": actual_top,
        })

    return {
        "stats": stats,
        "recent_misses": miss_reasons,
        "weights": load_weights(),
    }


def get_hit_rate_history() -> list[dict]:
    predictions = load_predictions()
    results = load_results()

    history = []
    for race_id, pred in predictions.items():
        result = results.get(race_id)
        if not result:
            continue
        finishing = {h["name"]: h["rank"] for h in result.get("finishing_order", [])}
        pred_horses = pred.get("horses", [])
        top_pick = pred_horses[0]["name"] if pred_horses else ""
        bias = _check_odds_bias(pred_horses, result)
        history.append({
            "race_id": race_id,
            "race_name": pred.get("race_name", race_id),
            "timestamp": pred.get("timestamp", ""),
            "hit_1st": int(finishing.get(top_pick) == 1),
            "hit_top3": int(finishing.get(top_pick, 99) <= 3),
            "confidence": pred_horses[0].get("confidence", 0) if pred_horses else 0,
            "ev_gap": pred_horses[0].get("ev_gap", "?") if pred_horses else "?",
            "odds_biased": int(bias["biased"]),
            "missed_ev": int(bias["missed_ev"]),
        })

    history.sort(key=lambda x: x["timestamp"])
    return history
