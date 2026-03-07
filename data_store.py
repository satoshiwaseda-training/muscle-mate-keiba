"""Local JSON data management for predictions, results, and PDCA weights."""

import json
import os
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
PREDICTIONS_FILE = DATA_DIR / "predictions.json"
RESULTS_FILE = DATA_DIR / "results.json"
WEIGHTS_FILE = DATA_DIR / "weights.json"
HORSE_PROFILES_FILE = DATA_DIR / "horse_profiles.json"
JRA_GROUND_TRUTH_FILE = DATA_DIR / "jra_ground_truth.json"
PDCA_LOG_FILE = DATA_DIR / "pdca_log.json"

# 科学的黄金比: 生体40% / 環境30% / 人間20% / 背景10%
DEFAULT_WEIGHTS = {
    "bio_condition": 0.40,    # 生体・コンディション (調教ラップ/パドック/体重)
    "environment": 0.30,      # 環境・適性 (馬場/天気/輸送/クッション値)
    "human_skill": 0.20,      # 人間・相性 (騎手大舞台実績/コンビ成績)
    "background": 0.10,       # 背景・資本 (外厩密度/馬主資金力/血統)
}


def _ensure_data_dir():
    DATA_DIR.mkdir(exist_ok=True)


def _load_json(path: Path, default):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: Path, data):
    _ensure_data_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# --- Predictions ---

def load_predictions() -> dict:
    return _load_json(PREDICTIONS_FILE, {})


def save_prediction(race_id: str, prediction: dict):
    """Save a prediction for a race.

    prediction: {
        "race_name": str,
        "horses": [{"rank": int, "name": str, "confidence": float, "reason": str, "bet": str}],
        "gemini_comment": str,
        "timestamp": str,
    }
    """
    data = load_predictions()
    prediction["timestamp"] = datetime.now().isoformat()
    data[race_id] = prediction
    _save_json(PREDICTIONS_FILE, data)


def get_prediction(race_id: str) -> dict | None:
    data = load_predictions()
    return data.get(race_id)


# --- Results ---

def load_results() -> dict:
    return _load_json(RESULTS_FILE, {})


def save_result(race_id: str, result: dict):
    """Save confirmed race result.

    result: {
        "race_name": str,
        "finishing_order": [{"rank": int, "name": str, "time": str}],
        "payouts": {"win": int, "place": int, "exacta": int, ...},
        "timestamp": str,
    }
    """
    data = load_results()
    result["timestamp"] = datetime.now().isoformat()
    data[race_id] = result
    _save_json(RESULTS_FILE, data)


def get_result(race_id: str) -> dict | None:
    data = load_results()
    return data.get(race_id)


# --- Weights ---

def load_weights() -> dict:
    """ローカルファイル → GitHub Gist の順に重みを読み込む。"""
    if WEIGHTS_FILE.exists():
        return _load_json(WEIGHTS_FILE, DEFAULT_WEIGHTS.copy())
    # ローカルになければ Gist から復元を試みる
    try:
        import github_sync
        remote = github_sync.pull_weights()
        if remote:
            _save_json(WEIGHTS_FILE, remote)
            return remote
    except Exception:
        pass
    return DEFAULT_WEIGHTS.copy()


def save_weights(weights: dict):
    # Normalize so they sum to 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: round(v / total, 4) for k, v in weights.items()}
    _save_json(WEIGHTS_FILE, weights)
    # GitHub Gist にも同期（失敗しても無視）
    try:
        import github_sync
        github_sync.push_weights(weights)
    except Exception:
        pass


def reset_weights():
    save_weights(DEFAULT_WEIGHTS.copy())


# --- Horse Profiles (Best Weight & Transport/Weather History) ---

def load_horse_profiles() -> dict:
    return _load_json(HORSE_PROFILES_FILE, {})


def save_horse_profile(horse_id: str, profile: dict):
    data = load_horse_profiles()
    data[horse_id] = profile
    _save_json(HORSE_PROFILES_FILE, data)


def get_horse_profile(horse_id: str) -> dict | None:
    return load_horse_profiles().get(horse_id)


def upsert_best_weight_record(
    horse_id: str,
    race_date: str,
    rank: int,
    weight_kg: int,
    venue: str,
    weather_temp: str,
    transport_km: int,
):
    """
    Record a race result for best-weight and transport/weather analysis.

    best_weight_records  — only stores rank 1 or 2 finishes (ベスト体重DB)
    transport_weight_log — stores all races for environmental correlation
    """
    data = load_horse_profiles()
    profile = data.get(horse_id, {"best_weight_records": [], "transport_weight_log": []})

    entry = {
        "date": race_date,
        "rank": rank,
        "weight_kg": weight_kg,
        "venue": venue,
        "weather_temp": weather_temp,
        "transport_km": transport_km,
    }

    if rank <= 2:
        profile.setdefault("best_weight_records", []).append(entry)

    profile.setdefault("transport_weight_log", []).append(entry)

    data[horse_id] = profile
    _save_json(HORSE_PROFILES_FILE, data)


# --- JRA Ground Truth (最優先データソース) ---

def save_jra_ground_truth(race_id: str, data: dict):
    """JRA公式馬場情報をGround Truthとして保存。外部スクレイプより優先される。"""
    store = _load_json(JRA_GROUND_TRUTH_FILE, {})
    from datetime import datetime as _dt
    data["saved_at"] = _dt.now().isoformat()
    store[race_id] = data
    _save_json(JRA_GROUND_TRUTH_FILE, store)


def get_jra_ground_truth(race_id: str) -> dict:
    """JRA公式データを取得。存在しない場合は空dict。"""
    store = _load_json(JRA_GROUND_TRUTH_FILE, {})
    return store.get(race_id, {})


def merge_track_data(jra_data: dict, scraped_data: dict) -> dict:
    """
    JRA公式データを最優先(Ground Truth)として、外部スクレイプデータとマージ。
    JRA側に値がある場合は常にJRAを使用する。

    Priority: JRA公式 > netkeiba/OpenMeteo等
    """
    merged = dict(scraped_data)  # 外部データをベースに

    # JRAの値で上書き（空でない場合のみ）
    for key in ["cushion_value", "water_content_goal", "water_content_4c",
                "track_bias_text", "going", "inner_rail_moved", "turf_replaced"]:
        val = jra_data.get(key)
        if val not in (None, "", False):
            merged[key] = val
            merged[f"{key}_source"] = "JRA公式"

    # 取消馬は常に反映（安全のため）
    if jra_data.get("scratched"):
        merged["scratched"] = jra_data["scratched"]

    return merged


# --- Stats helpers ---

def compute_stats() -> dict:
    """Compute overall hit rate statistics from predictions vs results."""
    predictions = load_predictions()
    results = load_results()

    total = 0
    hit_1st = 0
    hit_top3 = 0

    for race_id, pred in predictions.items():
        result = results.get(race_id)
        if not result:
            continue
        total += 1

        finishing = {h["name"]: h["rank"] for h in result.get("finishing_order", [])}
        pred_horses = pred.get("horses", [])

        if pred_horses:
            top_pick = pred_horses[0]["name"]
            if finishing.get(top_pick) == 1:
                hit_1st += 1
            if finishing.get(top_pick, 99) <= 3:
                hit_top3 += 1

    return {
        "total_races": total,
        "hit_1st": hit_1st,
        "hit_top3": hit_top3,
        "rate_1st": round(hit_1st / total, 3) if total else 0,
        "rate_top3": round(hit_top3 / total, 3) if total else 0,
    }


def save_pdca_log(race_id: str, analysis: dict):
    """Append a PDCA analysis record. Deduplicates by race_id."""
    log = _load_json(PDCA_LOG_FILE, [])
    entry = dict(analysis)
    entry["race_id"] = race_id
    entry["saved_at"] = datetime.now().isoformat()
    log = [e for e in log if e.get("race_id") != race_id]
    log.append(entry)
    _save_json(PDCA_LOG_FILE, log)


def load_pdca_log() -> list[dict]:
    return _load_json(PDCA_LOG_FILE, [])


def get_weight_history() -> list[dict]:
    """Return per-race weight snapshots from PDCA log (for trend chart)."""
    log = load_pdca_log()
    history = []
    for e in log:
        if e.get("new_weights"):
            history.append({
                "race_id": e.get("race_id", ""),
                "race_name": e.get("race_name", e.get("race_id", "")),
                "saved_at": e.get("saved_at", ""),
                "new_weights": e["new_weights"],
                "miss_categories": e.get("miss_categories", {}),
            })
    history.sort(key=lambda x: x["saved_at"])
    return history


def get_recent_miss_summary(n: int = 5) -> list[dict]:
    """Return last N races where top pick missed top 3."""
    predictions = load_predictions()
    results = load_results()

    misses = []
    for race_id, pred in predictions.items():
        result = results.get(race_id)
        if not result:
            continue
        finishing = {h["name"]: h["rank"] for h in result.get("finishing_order", [])}
        pred_horses = pred.get("horses", [])
        if not pred_horses:
            continue
        top_pick = pred_horses[0]["name"]
        if finishing.get(top_pick, 99) > 3:
            misses.append({
                "race_id": race_id,
                "race_name": pred.get("race_name", race_id),
                "prediction": pred,
                "result": result,
            })

    return misses[-n:]
