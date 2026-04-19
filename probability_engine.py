"""Per-horse win probabilities, top-3 set metrics, selection objective.

Pure functions over `scored` — a list of
    {"name": str, "score": float, "odds": float}
items. No I/O, no framework dependencies.

FORMULAS
--------
  p_i = exp(score_i / T) / Σ_k exp(score_k / T)      (softmax with temperature T)

  P1 = Σ_{i∈S} p_i                                   (prob winner is in S)

  Plackett–Luce top-2 approximation:
  P2 = Σ_{i≠j ∈ S} p_i · p_j / (1 − p_i)              (prob both 1st and 2nd are in S)

  Selection objective:
  argmax_{|S|=3}  α·P1 + β·P2                         (α+β=1, tunable)
"""

from __future__ import annotations

import json
import math
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_TEMPERATURE = 5.0       # legacy: softmax-on-scores path (deprecated for live)
                                # kept for backward compat with select_top3 / backtest
                                # harnesses that use raw score_runner outputs
DEFAULT_ALPHA = 0.5             # weight on P1 (winner in set)   -- tuned 2026-04
DEFAULT_BETA = 0.5              # weight on P2 (top-2 pair in set) -- tuned 2026-04
DEFAULT_EDGE_THRESHOLD = 0.02   # override edge (was 0.04)        -- relaxed 2026-04
DEFAULT_COVERAGE_THRESHOLD = 0.5
DIVERSITY_MAX_ODDS_MULTIPLE = 6.0  # no horse in S may exceed 6x the favorite's odds

# ── Calibrated probability layer ─────────────────────
# Market-anchored: market-implied prob is the BASE, fact edge is a
# bounded multiplicative adjustment. This replaces softmax(score) for
# live display since softmax on score_runner's 0-100 output with any
# reasonable temperature produces 95%+ concentration on the top horse.
#
# Formula per horse i:
#   implied_i     = 1 / odds_i
#   base_prob_i   = implied_i / sum(implied_j)          (overround stripped)
#   fact_edge_i   = composite_condition_i - 0.5         (in roughly [-0.5, +0.5])
#   multiplier_i  = exp(CALIBRATION_K * fact_edge_i)
#   adjusted_i    = base_prob_i * multiplier_i
#   win_prob_i    = adjusted_i / sum(adjusted_j)        (renormalized)

DEFAULT_CALIBRATION_K = 4.0     # bumped from 1.5 (2.67× amplification) to give
                                # structured edge meaningful pull against market
                                # anchor. With STRUCTURED_GAIN=2.0 in score_runner,
                                # edge range is now [-0.10, +0.36] → multiplier
                                # swing ~[-33%, +328%]. Paired with loose rule
                                # (cons≥1, comp≥0.60, odds≤15) so outliers get
                                # filtered at the bet-selection layer.

DEFAULT_MARKET_OVERROUND = 1.20  # fallback if we can't derive from data


def assign_calibrated_probs(
    scored: list[dict],
    k: float = DEFAULT_CALIBRATION_K,
) -> list[dict]:
    """Market-anchored win probability calculation.

    Input: `scored` list with at minimum `name`, `odds`, and optionally
           `composite_condition` per horse.
    Output: new list, sorted by win_prob descending, with added fields:
              - base_market_prob   : overround-stripped implied prob
              - fact_edge          : composite_condition - 0.5
              - fact_multiplier    : exp(k × fact_edge)
              - win_prob           : final calibrated probability

    Horses with odds <= 1.0 (scratched or missing) get base_market_prob=0
    and do not receive any fact adjustment.
    """
    if not scored:
        return []

    # Step 1 — implied probs from odds (overround-stripped)
    running = []
    non_running = []
    for h in scored:
        odds = float(h.get("odds", 0) or 0)
        if odds > 1.0:
            running.append((h, odds))
        else:
            non_running.append(h)

    out: list[dict] = []

    if not running:
        # Degenerate: no valid odds — fall back to uniform across the field
        n = max(len(scored), 1)
        for h in scored:
            out.append({
                **h,
                "base_market_prob": 1.0 / n,
                "fact_edge": 0.0,
                "fact_multiplier": 1.0,
                "win_prob": 1.0 / n,
            })
        return out

    implied = [1.0 / odds for _, odds in running]
    total_implied = sum(implied)
    if total_implied <= 0:
        total_implied = 1.0
    base_probs = [i / total_implied for i in implied]

    # Step 2 — fact edge per horse.
    # PRIMARY source: structured_edge (score_runner's structured adjustment,
    # computed by core_model_bridge.structured_edge_from_score). This
    # routes the ORIGINAL normalized/nonlinear/interaction/conditional
    # model directly into the probability layer without double-counting
    # the odds base.
    # FALLBACK: composite_condition - 0.5 (used by backtest harnesses
    # that don't compute a structured edge).
    edges: list[float] = []
    edge_source = "structured_edge"
    for h, _ in running:
        if "structured_edge" in h:
            edges.append(float(h.get("structured_edge", 0) or 0))
        else:
            edge_source = "composite_condition"
            comp = float(h.get("composite_condition", 0.5) or 0.5)
            edges.append(comp - 0.5)

    # Step 3 — multiplicative adjustment
    multipliers = [math.exp(k * e) for e in edges]
    adjusted = [bp * m for bp, m in zip(base_probs, multipliers)]
    total_adj = sum(adjusted) or 1.0
    final = [a / total_adj for a in adjusted]

    # Step 4 — assemble output for running horses
    for (h, odds), base_p, edge, mult, win_p in zip(
        running, base_probs, edges, multipliers, final
    ):
        out.append({
            **h,
            "base_market_prob": round(base_p, 4),
            "fact_edge": round(edge, 4),
            "fact_multiplier": round(mult, 4),
            "win_prob": round(win_p, 4),
        })

    # Scratched / missing-odds horses — zero everything
    for h in non_running:
        out.append({
            **h,
            "base_market_prob": 0.0,
            "fact_edge": 0.0,
            "fact_multiplier": 1.0,
            "win_prob": 0.0,
        })

    out.sort(key=lambda r: r["win_prob"], reverse=True)
    return out


def calibration_warnings(ranked: list[dict]) -> list[str]:
    """Return a list of plain-Japanese warning strings when the
    calibrated probabilities look suspicious.

    Triggers:
      - Any horse with odds ≥ 1.5 ending up with win_prob > 0.85
      - In a full field (≥ 5 running), top horse > 0.90
    """
    warnings: list[str] = []
    running = [r for r in ranked if r.get("odds", 0) > 1.0]
    for r in running:
        odds = float(r["odds"])
        p = float(r.get("win_prob", 0))
        if odds >= 1.5 and p > 0.85:
            warnings.append(
                f"{r['name']} (odds {odds:.1f}) → win_prob {p*100:.1f}% "
                f"— オッズに対して高すぎる可能性"
            )
    if len(running) >= 5 and running:
        top = max(running, key=lambda r: r.get("win_prob", 0))
        top_p = float(top.get("win_prob", 0))
        if top_p > 0.90:
            warnings.append(
                f"トップ {top['name']} が {top_p*100:.1f}% "
                f"— フルフィールドで 90%+ は異常値の可能性"
            )
    return warnings

CONFIG_FILE = Path(__file__).parent / "data" / "probability_config.json"


# ── Softmax ────────────────────────────────────────────

def softmax(scores: Sequence[float], temperature: float = DEFAULT_TEMPERATURE) -> list[float]:
    """Numerically-stable softmax with temperature."""
    if not scores:
        return []
    T = max(temperature, 1e-6)
    m = max(scores)
    exps = [math.exp((s - m) / T) for s in scores]
    Z = sum(exps) or 1.0
    return [e / Z for e in exps]


# ── Probability assignment ─────────────────────────────

def assign_win_probs(
    scored: list[dict],
    temperature: float = DEFAULT_TEMPERATURE,
) -> list[dict]:
    """Return a new list with `win_prob` field added and rows sorted by prob desc."""
    if not scored:
        return []
    probs = softmax([r["score"] for r in scored], temperature)
    out = [{**r, "win_prob": p} for r, p in zip(scored, probs)]
    out.sort(key=lambda r: r["win_prob"], reverse=True)
    return out


# ── Set-level probabilities ────────────────────────────

def prob_winner_in_set(probs_in_set: Iterable[float]) -> float:
    """P1 — probability at least one horse in S wins."""
    return float(sum(probs_in_set))


def prob_top2_in_set(probs_in_set: Sequence[float]) -> float:
    """P2 — Plackett–Luce approximation for both 1st and 2nd coming from S.

    For each ordered pair (i, j) in S:
        P(i 1st) · P(j 2nd | i out) = p_i · p_j / (1 − p_i)
    Clamped so tiny numerical bleed doesn't push P2 > P1² bound.
    """
    k = len(probs_in_set)
    if k < 2:
        return 0.0
    total = 0.0
    for i in range(k):
        pi = probs_in_set[i]
        denom = 1.0 - pi
        if denom <= 1e-9:
            continue
        for j in range(k):
            if i == j:
                continue
            total += pi * probs_in_set[j] / denom
    return max(0.0, min(1.0, total))


# ── Selection objective ────────────────────────────────

def select_top3(
    ranked: list[dict],
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    candidate_pool: int = 8,
    enforce_diversity: bool = True,
    odds_favorite_name: str | None = None,
    max_odds_multiple: float = DIVERSITY_MAX_ODDS_MULTIPLE,
) -> dict:
    """Pick the 3-horse subset S that maximizes α·P1 + β·P2.

    Diversity constraint (enforce_diversity=True):
      - S must contain at least 1 horse whose name differs from
        `odds_favorite_name` (if provided and a valid odds fav exists)
      - No horse in S may have odds greater than `max_odds_multiple`
        times the favorite's odds (excludes extreme longshots)
      - If no triple satisfies constraints, falls back to unconstrained
        top-3 and notes the fallback in `diversity_fallback`
    """
    if not ranked:
        return {"selected": [], "p1": 0.0, "p2": 0.0, "objective": 0.0,
                "pool_size": 0, "diversity_fallback": False}

    # Small field: no selection to do
    if len(ranked) <= 3:
        probs = [h["win_prob"] for h in ranked]
        return {
            "selected": list(ranked),
            "p1": prob_winner_in_set(probs),
            "p2": prob_top2_in_set(probs),
            "objective": alpha * prob_winner_in_set(probs) + beta * prob_top2_in_set(probs),
            "pool_size": len(ranked),
            "diversity_fallback": False,
        }

    # Determine the odds favorite from the field if not passed in
    running = [h for h in ranked if (h.get("odds") or 0) > 0]
    if running:
        of_row = min(running, key=lambda r: r["odds"])
        odds_fav_name = odds_favorite_name or of_row["name"]
        fav_odds = of_row["odds"]
    else:
        odds_fav_name = odds_favorite_name
        fav_odds = None

    pool = ranked[: max(3, candidate_pool)]

    def triple_allowed(members: list[dict]) -> bool:
        if not enforce_diversity:
            return True
        # Rule 1: must contain at least one non-favorite
        if odds_fav_name is not None:
            if all(m["name"] == odds_fav_name for m in members):
                return False
        # Rule 2: no extreme longshot (odds > max_odds_multiple * fav_odds)
        if fav_odds is not None:
            cap = fav_odds * max_odds_multiple
            for m in members:
                if (m.get("odds") or 0) > cap:
                    return False
        return True

    def score_triple(indices: tuple[int, ...]) -> tuple[float, float, float]:
        probs = [pool[i]["win_prob"] for i in indices]
        p1 = prob_winner_in_set(probs)
        p2 = prob_top2_in_set(probs)
        return (alpha * p1 + beta * p2, p1, p2)

    best = None
    for triple in combinations(range(len(pool)), 3):
        members = [pool[i] for i in triple]
        if not triple_allowed(members):
            continue
        obj, p1, p2 = score_triple(triple)
        if best is None or obj > best["objective"]:
            best = {
                "selected": members,
                "p1": p1, "p2": p2, "objective": obj,
                "pool_size": len(pool),
                "diversity_fallback": False,
            }

    # Diversity fallback: relax if no triple satisfied constraints
    if best is None:
        for triple in combinations(range(len(pool)), 3):
            obj, p1, p2 = score_triple(triple)
            if best is None or obj > best["objective"]:
                best = {
                    "selected": [pool[i] for i in triple],
                    "p1": p1, "p2": p2, "objective": obj,
                    "pool_size": len(pool),
                    "diversity_fallback": True,
                }

    best["selected"].sort(key=lambda r: r["win_prob"], reverse=True)
    return best


# ── Override gate ─────────────────────────────────────

def should_override_market(
    selected_top: dict,
    odds_favorite: dict,
    feature_coverage: float,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
    edge_threshold: float = DEFAULT_EDGE_THRESHOLD,
) -> tuple[bool, str]:
    """Decide whether the selected model top pick should override the odds favorite.

    Returns (allow_override, reason). When coverage falls below the threshold
    the caller should also switch to CONSERVATIVE MODE: use odds-based ranking
    with the scorer contributing only as a tiebreaker. See `conservative_mode`.
    """
    if not selected_top or not odds_favorite:
        return (False, "missing-input")
    if selected_top.get("name") == odds_favorite.get("name"):
        return (False, "no-deviation")
    if not selected_top.get("confirmed_running", True):
        return (False, "candidate-not-confirmed-running")
    if feature_coverage < coverage_threshold:
        return (False, f"low-feature-coverage ({feature_coverage:.0%} < {coverage_threshold:.0%})")
    edge = selected_top.get("win_prob", 0.0) - odds_favorite.get("win_prob", 0.0)
    if edge < edge_threshold:
        return (False, f"insufficient-edge ({edge:+.3f} < {edge_threshold:+.3f})")
    return (True, f"override-ok (edge {edge:+.3f}, coverage {feature_coverage:.0%})")


def conservative_mode(ranked: list[dict]) -> list[dict]:
    """Fallback ranking when feature coverage is too low to trust the scorer.

    Re-sorts by odds ascending (implied probability descending). The model's
    softmax output is still attached but the order is dictated by the market.
    """
    if not ranked:
        return []
    running = [r for r in ranked if (r.get("odds") or 0) > 0]
    scratched = [r for r in ranked if not ((r.get("odds") or 0) > 0)]
    running.sort(key=lambda r: r["odds"])
    return running + scratched


# ── Temperature calibration ────────────────────────────

def calibrate_temperature(
    history: list[dict],
    grid: tuple[float, ...] = (4, 6, 8, 10, 12, 15, 18, 22, 28),
) -> tuple[float, dict]:
    """Grid-search the softmax temperature that minimizes mean negative log-likelihood
    of the actual winner over historical races.

    `history` rows: {"scores": [...], "names": [...], "winner_name": str}
    Returns (best_T, {T: nll}).
    """
    table: dict[float, float] = {}
    best_T = DEFAULT_TEMPERATURE
    best_nll = float("inf")
    for T in grid:
        nll_sum = 0.0
        n = 0
        for row in history:
            names = row.get("names") or []
            scores = row.get("scores") or []
            winner = row.get("winner_name")
            if not names or not scores or not winner or winner not in names:
                continue
            probs = softmax(scores, T)
            p_win = max(probs[names.index(winner)], 1e-12)
            nll_sum += -math.log(p_win)
            n += 1
        if n == 0:
            continue
        avg = nll_sum / n
        table[T] = avg
        if avg < best_nll:
            best_nll = avg
            best_T = T
    return best_T, table


# ── Config persistence ─────────────────────────────────

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"temperature": DEFAULT_TEMPERATURE, "alpha": DEFAULT_ALPHA, "beta": DEFAULT_BETA}


def save_config(cfg: dict) -> None:
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
