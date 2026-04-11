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


DEFAULT_TEMPERATURE = 5.0       # calibrated 2026-04 on 50-race enriched backtest
                                # T=12 floored every horse at ~2% and triggered
                                # longshot value bets; T=5 concentrates ~67% on
                                # the top horse and floors longshots at ~0.2%,
                                # removing the softmax-floor longshot bias.
DEFAULT_ALPHA = 0.5             # weight on P1 (winner in set)   -- tuned 2026-04
DEFAULT_BETA = 0.5              # weight on P2 (top-2 pair in set) -- tuned 2026-04
DEFAULT_EDGE_THRESHOLD = 0.02   # override edge (was 0.04)        -- relaxed 2026-04
DEFAULT_COVERAGE_THRESHOLD = 0.5
DIVERSITY_MAX_ODDS_MULTIPLE = 6.0  # no horse in S may exceed 6x the favorite's odds

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
