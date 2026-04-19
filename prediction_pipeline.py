"""End-to-end theoretical prediction pipeline.

Pipeline stages
---------------
  1. fetch latest entries + race info + (optional) enrichment
  2. filter scratched / excluded horses (live: JRA changes page; backtest: result set)
  3. extract T-15-safe structured features
  4. score every horse independently with score_runner
  5. convert raw scores → calibrated win probabilities (softmax with temperature)
  6. select top-3 set maximizing α·P1 + β·P2
  7. persist prediction payload

The same pipeline powers `app_theoretical.py` (live) and `backtest.py`
(historical). The only stage that differs is how "scratched" is decided.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional

import scraper
import feature_store
import probability_engine as pe
from train import score_runner
from data_store import load_weights, save_prediction


# ── Feature coverage flags used by the override gate ─────────
COVERAGE_FIELDS = ("jockey_win_rate", "training_acceleration",
                   "paddock_vascularity", "horse_weight_delta")


@dataclass
class PredictionResult:
    race_id: str
    race_name: str
    grade: str
    venue: str
    race_date: str
    n_entries_raw: int
    n_scratched: int
    n_scored: int
    scratched: list[str] = field(default_factory=list)
    horses: list[dict] = field(default_factory=list)       # ranked, with win_prob
    selected_top3: list[dict] = field(default_factory=list)
    p1: float = 0.0
    p2: float = 0.0
    objective: float = 0.0
    feature_coverage: float = 0.0
    enriched: bool = False
    override_decision: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


# ── Scratch filtering ────────────────────────────────────

def filter_scratched(entries: list[dict], scratched_names: set[str]) -> tuple[list[dict], list[str]]:
    """Remove scratched horses. Returns (kept, scratched_names_found)."""
    kept, removed = [], []
    for e in entries:
        name = (e.get("name") or "").strip()
        # Explicit scratch list (JRA changes page or result fall-back)
        if name in scratched_names:
            removed.append(name)
            continue
        # Odds "---" / "0" / "取消" is also a scratch indicator on historical pages
        raw_odds = str(e.get("odds") or "").strip()
        if raw_odds in ("", "---", "--", "取消", "除外"):
            removed.append(name)
            continue
        kept.append(e)
    return kept, removed


# ── Feature coverage score ───────────────────────────────

def coverage_ratio(structured: dict) -> float:
    """Fraction of enrichment fields populated on the first horse. 0..1."""
    horses = (structured or {}).get("horses") or {}
    if not horses:
        return 0.0
    h = next(iter(horses.values()))
    hit = 0
    for k in COVERAGE_FIELDS:
        v = h.get(k)
        if isinstance(v, (int, float)) and v != 0:
            hit += 1
    return hit / len(COVERAGE_FIELDS)


# ── Score every horse ────────────────────────────────────

def score_all_horses(entries: list[dict], structured: dict, grade: str) -> list[dict]:
    """Score each horse by rotating it into the horses[0] slot.

    score_runner() was designed to score whichever horse is in position 0.
    To rank the full field we call it once per horse.
    """
    ctx = {"weights": load_weights()}
    scored = []
    for horse in entries:
        this_name = (horse.get("name") or "").strip()
        odds_val = feature_store._parse_odds(horse.get("odds", "0"))
        # Put `horse` first, then all others — score_runner reads horses[0]
        this_first = [{
            "name": this_name,
            "rank": 1,
            "odds": odds_val,
            "confidence": 0, "ev_gap": 0, "bet": "",
        }]
        for other in entries:
            if other is horse:
                continue
            this_first.append({
                "name": (other.get("name") or "").strip(),
                "rank": 2,
                "odds": feature_store._parse_odds(other.get("odds", "0")),
                "confidence": 0, "ev_gap": 0, "bet": "",
            })
        feat = {
            "grade": grade,
            "num_horses": len(entries),
            "horse_features": this_first,
            "structured_features": structured,
        }
        s = score_runner(feat, ctx).get("top_confidence", 0.0)
        scored.append({
            "name": this_name,
            "odds": odds_val,
            "score": float(s),
            "horse_id": horse.get("horse_id", ""),
            "jockey": horse.get("jockey", ""),
            "confirmed_running": True,
        })
    return scored


# ── Reasons for the selected horses ──────────────────────

def horse_reasons(name: str, structured: dict) -> list[str]:
    """Readable bullet points explaining why this horse scored well.

    Pulls top-contributing normalized signals from structured_features.
    Silently returns an empty list if no enrichment is present.
    """
    if not structured:
        return []
    h = (structured.get("horses") or {}).get(name, {})
    if not h:
        return []
    contribs: list[tuple[float, str]] = []
    jwr = h.get("jockey_win_rate", 0) or 0
    if jwr:
        contribs.append((jwr, f"騎手勝率 {jwr*100:.1f}%"))
    tacc = h.get("training_acceleration", 0) or 0
    if tacc:
        contribs.append((abs(tacc), f"調教加速度 {tacc:+.3f}"))
    wd = h.get("horse_weight_delta", 0) or 0
    if wd:
        contribs.append((-abs(wd) / 10.0, f"馬体重変動 {wd:+d}kg"))
    pad_v = h.get("paddock_vascularity", 0) or 0
    pad_h = h.get("paddock_hindquarter", 0) or 0
    pad_g = h.get("paddock_gait", 0) or 0
    bio = (pad_v + pad_h + pad_g) / 3.0
    if bio:
        contribs.append((bio, f"パドック合成 {bio:+.2f}"))
    cardio = h.get("training_cardio_index", 0) or 0
    if cardio:
        contribs.append((cardio, f"心肺指数 {cardio:.2f}"))
    contribs.sort(reverse=True)
    return [txt for _, txt in contribs[:3]]


# ── Main entry points ────────────────────────────────────

def predict_live(
    race_id: str,
    venue: str = "",
    enrich: bool = True,
    alpha: float = pe.DEFAULT_ALPHA,
    beta: float = pe.DEFAULT_BETA,
    temperature: Optional[float] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> PredictionResult:
    """Fetch latest data, score all horses, return full prediction."""
    def log(m):
        if progress_cb: progress_cb(m)

    cfg = pe.load_config()
    T = float(temperature if temperature is not None else cfg.get("temperature", pe.DEFAULT_TEMPERATURE))

    log("レース情報を取得中...")
    race_info = scraper.fetch_race_info(race_id) or {}
    log("出走表を取得中...")
    entries = scraper.fetch_entries(race_id, venue) or []
    n_raw = len(entries)

    log("取消・除外馬を確認中...")
    scratched_names: set[str] = set()
    try:
        changes = scraper.fetch_jra_race_changes(race_id) or {}
        scratched_names |= set(changes.get("scratched", []))
    except Exception:
        pass
    entries, removed = filter_scratched(entries, scratched_names)
    scratched_names |= set(removed)

    if enrich and entries:
        log(f"詳細情報を収集中 ({len(entries)}頭)...")
        try:
            entries = scraper.enrich_entries(entries, race_id)
        except Exception as e:
            log(f"詳細情報の取得に一部失敗: {e}")

    log("構造化特徴量を抽出中...")
    structured = feature_store.extract_structured_features(
        entries=entries,
        race_info=race_info,
        track_condition=race_info.get("track_condition", ""),
        weather=race_info.get("weather", ""),
        temperature=race_info.get("temperature", ""),
        cushion_value=race_info.get("cushion_value", ""),
        venue=venue,
    )
    coverage = coverage_ratio(structured)

    log("全頭スコア計算中...")
    scored = score_all_horses(entries, structured, race_info.get("grade", ""))
    ranked = pe.assign_win_probs(scored, temperature=T)

    log("Top-3 最適化中...")
    sel = pe.select_top3(ranked, alpha=alpha, beta=beta)

    # Override gate evaluation
    if ranked:
        odds_fav = min(
            (r for r in ranked if r["odds"] > 0),
            key=lambda r: r["odds"],
            default=ranked[0],
        )
        model_top = sel["selected"][0] if sel["selected"] else ranked[0]
        allowed, reason = pe.should_override_market(
            selected_top=model_top,
            odds_favorite=odds_fav,
            feature_coverage=coverage,
        )
        override = {"allow": allowed, "reason": reason,
                    "model_top": model_top["name"], "odds_fav": odds_fav["name"]}
    else:
        override = {"allow": False, "reason": "no-horses"}

    # Attach reasons
    for h in sel["selected"]:
        h["reasons"] = horse_reasons(h["name"], structured)

    result = PredictionResult(
        race_id=race_id,
        race_name=race_info.get("race_name", ""),
        grade=race_info.get("grade", ""),
        venue=venue,
        race_date=datetime.now().date().isoformat(),
        n_entries_raw=n_raw,
        n_scratched=len(scratched_names),
        n_scored=len(ranked),
        scratched=sorted(scratched_names),
        horses=ranked,
        selected_top3=sel["selected"],
        p1=sel["p1"],
        p2=sel["p2"],
        objective=sel["objective"],
        feature_coverage=coverage,
        enriched=enrich,
        override_decision=override,
    )

    # Persist
    payload = {
        "race_name": result.race_name,
        "grade": result.grade,
        "timestamp": datetime.now().isoformat(),
        "horses": [
            {"rank": i + 1, "name": h["name"], "odds": str(h["odds"]),
             "win_prob": h["win_prob"], "score": h["score"],
             "confidence": round(h["win_prob"] * 100, 1),
             "ev_gap": "0", "bet": ""}
            for i, h in enumerate(ranked[:8])
        ],
        "structured_features": structured,
        "theoretical": {
            "selected_top3": [h["name"] for h in sel["selected"]],
            "p1": sel["p1"],
            "p2": sel["p2"],
            "objective": sel["objective"],
            "temperature": T,
            "alpha": alpha,
            "beta": beta,
            "feature_coverage": coverage,
            "scratched": sorted(scratched_names),
            "override": override,
        },
        "_ranking_meta": {
            "odds_top": override.get("odds_fav", ""),
            "score_top": override.get("model_top", ""),
            "ranking_changed": override.get("model_top") != override.get("odds_fav"),
            "top_score": ranked[0]["score"] if ranked else 0,
        },
    }
    try:
        save_prediction(race_id, payload)
    except Exception as e:
        log(f"保存に失敗: {e}")

    return result


def predict_backtest(
    race_id: str,
    entries: list[dict],
    race_info: dict,
    result: dict,
    alpha: float = pe.DEFAULT_ALPHA,
    beta: float = pe.DEFAULT_BETA,
    temperature: Optional[float] = None,
) -> PredictionResult:
    """Same pipeline as live, but the scratch set comes from `result.finishing_order`.

    Used by the updated backtest. Never calls the network.
    """
    cfg = pe.load_config()
    T = float(temperature if temperature is not None else cfg.get("temperature", pe.DEFAULT_TEMPERATURE))

    fo = result.get("finishing_order") or []
    finishers = {(h.get("name") or "").strip() for h in fo}
    # In backtest, "scratched" = in entries but not in finishing_order
    scratched_names = {(e.get("name") or "").strip() for e in entries
                       if (e.get("name") or "").strip() not in finishers}
    entries, _ = filter_scratched(entries, scratched_names)

    structured = feature_store.extract_structured_features(
        entries=entries,
        race_info=race_info,
        track_condition=race_info.get("track_condition", ""),
        weather=race_info.get("weather", ""),
        temperature=race_info.get("temperature", ""),
        cushion_value=race_info.get("cushion_value", ""),
        venue=race_info.get("venue", ""),
    )
    coverage = coverage_ratio(structured)
    scored = score_all_horses(entries, structured, race_info.get("grade", ""))
    ranked = pe.assign_win_probs(scored, temperature=T)
    sel = pe.select_top3(ranked, alpha=alpha, beta=beta)

    for h in sel["selected"]:
        h["reasons"] = horse_reasons(h["name"], structured)

    return PredictionResult(
        race_id=race_id,
        race_name=result.get("race_name", ""),
        grade=race_info.get("grade", ""),
        venue=race_info.get("venue", ""),
        race_date=result.get("timestamp", "")[:10],
        n_entries_raw=len(entries) + len(scratched_names),
        n_scratched=len(scratched_names),
        n_scored=len(ranked),
        scratched=sorted(scratched_names),
        horses=ranked,
        selected_top3=sel["selected"],
        p1=sel["p1"],
        p2=sel["p2"],
        objective=sel["objective"],
        feature_coverage=coverage,
        enriched=any(coverage > 0 for _ in [0]),
    )


def to_dict(pr: PredictionResult) -> dict:
    return asdict(pr)
