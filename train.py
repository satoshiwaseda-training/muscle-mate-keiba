"""Candidate scoring function — expressive structured-feature formula.

EDITABLE: Only score_runner() may be modified for candidate generation.
Do NOT change function signature, return format, or add imports.

MODEL STRUCTURE (v6 — camp z-score normalization):
  - W_CAMP unchanged at 0.028
  - camp contribution switched from absolute (n_camp - 0.5) to
    race-relative z-score (rel_camp_z) so weak-camp horses are pushed
    DOWN relative to the field, not just elite-camp horses pushed up.
  - rel_camp_z is clamped to ±3σ; std falls back to 1.0 if degenerate.

PRIOR STRUCTURE (v5 — odds-band-conditional pedigree, calibrated camp):
  - W_CAMP bumped 0.015 → 0.028 (camp shows real win-rate signal)
  - W_PEDIGREE kept 0.025 (elite pedigree alone doesn't predict wins)
  - pedigree contribution dampened by odds_band:
      odds < 3   → weight 0.0  (don't move favorites)
      3-15       → weight 1.0  (sweet spot)
      odds > 15  → weight 0.5  (avoid longshot bias)

PRIOR STRUCTURE (v4 — pedigree + camp extension):
  1. Normalized odds-implied base probability
  2. Normalized structured signals (jockey, training, weight, bio)
  2h-j. Pedigree & camp signals (sire/damsire tier, distance/surface fit,
        breeder/owner tier, external stable)
  3. Four interaction terms (jockey*grade, training*bio, pedigree*camp)
  4. One conditional (heavy track boosts cardio weight)
  5. Nonlinear weight-delta penalty (quadratic)

COMPLEXITY BUDGET:
  - Total terms: 15 (was 11)
  - Interaction terms: 3 (was 2)
  - Conditional branches: 2 (unchanged)
  - All inputs normalized to [0, 1] range
  - New terms use weak coefficients (0.01-0.025, centered)

GEMINI EXCLUSION:
  - "confidence": IGNORED
  - "ev_gap": IGNORED
  - "bet": IGNORED
"""


def score_runner(features: dict, context: dict) -> dict:
    """Score a race using normalized, interacting structured features.

    Returns {"top_confidence": float} in range [2, 95].
    """
    horses = features.get("horse_features", [])
    if not horses:
        return {"top_confidence": 50}

    num_horses = features.get("num_horses", 0) or len(horses)
    top = horses[0]
    top_name = top.get("name", "")
    sf = features.get("structured_features")

    # ── Resolve structured data ──
    if sf and isinstance(sf, dict):
        h = sf.get("horses", {}).get(top_name, {})
        race = sf.get("race", {})
    else:
        h = {}
        race = {}

    # ════════════════════════════════════════════════════════
    # STEP 1: Normalized base probability from odds
    # ════════════════════════════════════════════════════════
    odds = h.get("odds", 0.0) or top.get("odds", 0.0)
    if odds > 1.0:
        raw_prob = 1.0 / odds
        base = _clamp(raw_prob / 1.20, 0.02, 0.80)  # overround-corrected
    else:
        base = _clamp(1.0 / max(num_horses, 1), 0.02, 0.80)

    # ════════════════════════════════════════════════════════
    # STEP 2: Normalized individual signals (all → [0, 1])
    # ════════════════════════════════════════════════════════

    # 2a. Field size signal: fewer runners = higher base rate
    #     Normalized: 8 runners → 1.0, 18 runners → 0.0
    n_field = _clamp((18 - num_horses) / 10.0, 0.0, 1.0)

    # 2b. Grade signal: G1 = 1.0, G2 = 0.5, G3 = 0.0
    grade = features.get("grade", "")
    n_grade = 1.0 if grade == "G1" else (0.5 if grade == "G2" else 0.0)

    # 2c. Jockey quality: win_rate normalized to [0, 1]
    #     JRA range: ~2% (apprentice) to ~25% (top). Scale: 0% → 0, 25% → 1
    jockey_wr = h.get("jockey_win_rate", 0.0)
    n_jockey = _clamp(jockey_wr / 0.25, 0.0, 1.0)

    # 2d. Training acceleration: normalized [-1, 1] → [0, 1]
    #     Range: roughly -0.15 to +0.15. Scale to unit.
    training_acc = h.get("training_acceleration", 0.0)
    n_training = _clamp((training_acc + 0.15) / 0.30, 0.0, 1.0)

    # 2e. Weight delta: nonlinear penalty (quadratic)
    #     0-4 kg = normal (penalty ≈ 0). 4-20 kg = increasing penalty.
    #     Normalized penalty: 0.0 (stable) to 1.0 (extreme)
    abs_delta = abs(h.get("horse_weight_delta", 0.0))
    excess = _clamp(abs_delta - 4.0, 0.0, 16.0)  # excess beyond 4kg
    n_weight_penalty = (excess / 16.0) ** 2  # quadratic: 0→0, 16→1

    # 2f. Bio composite (paddock signals)
    #     Each paddock score is in [-1, 1]. Average then scale to [0, 1].
    pad_v = h.get("paddock_vascularity", 0.0)
    pad_h = h.get("paddock_hindquarter", 0.0)
    pad_g = h.get("paddock_gait", 0.0)
    bio_raw = (pad_v + pad_h + pad_g) / 3.0  # [-1, 1]
    n_bio = _clamp((bio_raw + 1.0) / 2.0, 0.0, 1.0)  # → [0, 1]

    # 2g. Consensus signal: how much lower is top pick's odds vs field
    other_odds = [x.get("odds", 0) for x in horses[1:] if x.get("odds", 0) > 1.0]
    if odds > 1.0 and other_odds:
        ratio = odds / (sum(other_odds) / len(other_odds))
        n_consensus = _clamp(1.0 - ratio, 0.0, 1.0)  # lower ratio = stronger
    else:
        n_consensus = 0.5  # neutral when unknown

    # ════════════════════════════════════════════════════════
    # STEP 2h-j: Pedigree & camp signals (v2)
    # ════════════════════════════════════════════════════════

    # 2h. Pedigree composite: [0, 1] from sire/damsire tier + distance/surface fit
    pedigree_comp = h.get("pedigree_composite", 0.5)
    n_pedigree = _clamp(pedigree_comp, 0.0, 1.0)

    # 2i. Camp composite: [0, 1] from breeder/owner tier + external stable
    camp_comp = h.get("camp_composite", 0.5)
    n_camp = _clamp(camp_comp, 0.0, 1.0)

    # ── v4: race-relative camp z-score ──
    # Backtest finding: absolute camp_composite shifts all horses similarly
    # (most G1/G2 horses have ノーザン tier=5 breeder), so it doesn't move
    # rankings. Switch to z-score within the race so weak-camp horses
    # are pushed DOWN relative to the field, even if no horse is "elite".
    camps_in_race = [
        hf.get("camp_composite", 0.5)
        for hf in (sf.get("horses", {}) if isinstance(sf, dict) else {}).values()
        if isinstance(hf, dict)
    ]
    if len(camps_in_race) >= 2:
        mean_camp = sum(camps_in_race) / len(camps_in_race)
        var_camp = sum((c - mean_camp) ** 2 for c in camps_in_race) / len(camps_in_race)
        std_camp = var_camp ** 0.5
        if std_camp <= 0:
            std_camp = 1.0
        rel_camp_z = (camp_comp - mean_camp) / std_camp
        # Clamp to ±3σ to defend against degenerate races
        rel_camp_z = _clamp(rel_camp_z, -3.0, 3.0)
    else:
        rel_camp_z = 0.0

    # 2j. Sire distance fit: [0, 1] — how well sire suits this race distance
    sire_dist = h.get("sire_distance_fit", 0.5)
    n_sire_dist = _clamp(sire_dist, 0.0, 1.0)

    # ════════════════════════════════════════════════════════
    # STEP 3: Interaction terms (max 4)
    # ════════════════════════════════════════════════════════

    # I1: Jockey × Grade — elite jockeys matter more in higher-grade races
    #     High-grade + high-jockey → strong positive; low-grade dilutes effect
    ix_jockey_grade = n_jockey * n_grade

    # I2: Training × Bio — bio-mechanical composite
    #     Good training acceleration + good paddock condition → synergy
    ix_training_bio = n_training * n_bio

    # I3: Pedigree × Camp — elite bloodline + strong camp → synergy
    ix_pedigree_camp = n_pedigree * n_camp

    # ════════════════════════════════════════════════════════
    # STEP 4: Conditional (max 1 condition, 2 branches)
    # ════════════════════════════════════════════════════════

    # C1: Heavy/yielding track → cardio (stamina) matters more
    track = race.get("track_condition", "")
    cardio = h.get("training_cardio_index", 0.0)
    n_cardio = _clamp(cardio, 0.0, 1.0)
    if track in ("重", "不良"):
        # Heavy track: cardio index becomes a significant signal
        cond_track = 0.04 * n_cardio
    else:
        cond_track = 0.01 * n_cardio  # light contribution on normal track

    # ════════════════════════════════════════════════════════
    # STEP 5: Weighted combination
    # ════════════════════════════════════════════════════════
    # Coefficients — the knobs for candidate optimization
    W_FIELD     = 0.048  # term 1 (was 0.06; reduced 20% — calibration)
    W_GRADE     = 0.02   # term 2
    W_JOCKEY    = 0.05   # term 3
    W_TRAINING  = 0.03   # term 4
    W_WEIGHT_P  = 0.04   # term 5 (penalty, subtracted)
    W_BIO       = 0.03   # term 6
    W_CONSENSUS = 0.04   # term 7
    W_IX_JG     = 0.03   # interaction 1
    W_IX_TB     = 0.02   # interaction 2
    # cond_track has its own internal weight (0.04 or 0.01)

    # ── Pedigree & camp coefficients (v3 — calibrated 2026-04 from n=121 backtest) ──
    # Backtest findings (n=121, 1810 horses):
    #   - camp_composite: strong 10.8% vs weak 5.0% win rate → REAL signal
    #     → bumped W_CAMP 0.015 → 0.028
    #   - pedigree_composite: elite 5.3% vs neutral 7.7% win rate
    #     → kept W_PEDIGREE at 0.025, but added odds-band dampener
    #   - sire_distance_fit: zero coverage (entity_tier table too sparse)
    #     → kept W_SIRE_DIST at 0.020, also subject to odds-band dampener
    W_PEDIGREE  = 0.025  # pedigree composite (centered)
    W_CAMP      = 0.028  # camp composite (centered) — bumped from 0.015
    W_SIRE_DIST = 0.020  # sire distance fit (centered)
    W_IX_PC     = 0.010  # interaction: pedigree × camp

    # ── Odds-band conditional pedigree dampener (v3) ──
    # Backtest showed pedigree was effective in 3-15 odds band, harmful
    # at >15 (over-promoting longshots), and unable to flip favorites <3.
    # Apply a multiplicative weight to pedigree contribution by odds band.
    if odds <= 0 or odds < 3.0:
        pedigree_weight = 0.0  # favorites: don't move; or no-odds (early)
    elif odds > 15.0:
        pedigree_weight = 0.5  # longshots: dampen
    else:
        pedigree_weight = 1.0  # sweet spot 3-15

    # Structured-signal gain: raw adjustment max ≈ 0.18 is narrow relative to
    # intra-race base gaps (median top1-top2 = 0.08), so structured terms rarely
    # flip rankings. Doubling them brings max swing to ≈0.36 and lets structured
    # influence ~10% of races without destabilizing the calibration layer.
    STRUCTURED_GAIN = 2.0

    adjustment = (
        W_FIELD     * n_field
        + W_GRADE   * n_grade
        + W_JOCKEY  * n_jockey
        + W_TRAINING * n_training
        - W_WEIGHT_P * n_weight_penalty
        + W_BIO     * n_bio
        + W_CONSENSUS * (n_consensus - 0.5)  # centered: neutral = 0
        + W_IX_JG   * ix_jockey_grade
        + W_IX_TB   * ix_training_bio
        + cond_track
        # ── v4: pedigree (dampened) + camp (z-score, race-relative) ──
        # pedigree-derived terms: dampened by odds-band weight (v3).
        # camp: switched to race-relative z-score so weak-camp horses
        #       are pushed down relative to the field (was: all moved together).
        + W_PEDIGREE  * pedigree_weight * (n_pedigree - 0.5)
        + W_CAMP      *                   rel_camp_z
        + W_SIRE_DIST * pedigree_weight * (n_sire_dist - 0.5)
        + W_IX_PC     * pedigree_weight * ix_pedigree_camp
    )

    final_prob = _clamp(base + STRUCTURED_GAIN * adjustment, 0.02, 0.95)
    return {"top_confidence": final_prob * 100.0}


def _clamp(val: float, lo: float, hi: float) -> float:
    """Clamp val to [lo, hi]. No imports needed."""
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val
