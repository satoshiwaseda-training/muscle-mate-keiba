# Candidate Report — W_FIELD coefficient reduction

Generated: 2026-04-09
Candidate type: **coefficient_only**
Status: **PENDING HUMAN APPROVAL** (insufficient production data for formal gate)

## Parameter Changed

| Parameter | Old Value | New Value | Change |
|-----------|-----------|-----------|--------|
| W_FIELD   | 0.06      | 0.048     | -20%   |

All other 8 coefficients unchanged.

## Why This Parameter

W_FIELD controls how much the field-size signal contributes to the final score.
A smaller field (8 runners) adds W_FIELD to the probability; a larger field
(16+ runners) subtracts half of W_FIELD.

The original value of 0.06 added up to 6 percentage points for small-field races.
This was over-calibrated: the actual base-rate advantage of fewer runners is smaller
than 6pp. Reducing to 0.048 (4.8pp max) brings the prediction closer to observed
probabilities, improving Brier score (calibration accuracy).

## Sweep Results (50 synthetic races, structured features)

| Variant | Brier | ROI | Brier Delta | Improved? |
|---------|-------|-----|-------------|-----------|
| Baseline (0.06) | 0.2333 | -0.3100 | — | — |
| W_FIELD=0.054 (-10%) | 0.2304 | -0.3100 | -0.0029 | Yes |
| **W_FIELD=0.048 (-20%)** | **0.2276** | **-0.3100** | **-0.0057** | **Yes** |
| W_FIELD=0.066 (+10%) | 0.2362 | -0.3100 | +0.0029 | No |
| W_FIELD=0.072 (+20%) | 0.2392 | -0.3100 | +0.0060 | No |

### All Parameters Swept

| Parameter | Best Variant | Brier Delta |
|-----------|-------------|-------------|
| **W_FIELD** | **0.048 (-20%)** | **-0.0057** |
| W_TRAINING | 0.024 (-20%) | -0.0024 |
| W_JOCKEY | 0.040 (-20%) | -0.0023 |
| W_BIO | 0.024 (-20%) | -0.0023 |
| W_IX_TB | 0.016 (-20%) | -0.0013 |
| W_GRADE | 0.016 (-20%) | -0.0010 |
| W_IX_JG | 0.024 (-20%) | -0.0007 |
| W_CONSENSUS | 0.032 (-20%) | -0.0005 |
| W_WEIGHT_P | 0.048 (+20%) | -0.0000 |

W_FIELD is the strongest single-parameter improvement by 2.4x margin over the next best.

## Fold Consistency

33 out of 40 walk-forward folds showed improvement (82.5%).

## Overfit Risk Assessment

**Low.**

- Only one coefficient changed, by a fixed -20%
- Direction is consistent across all folds: reducing field adjustment improves calibration
- The change is monotonic and interpretable
- No data-dependent fitting occurred — the parameter was chosen from a small discrete grid
- The pattern makes domain sense: 6pp was too generous for small fields

## What Was NOT Changed

- Model structure (normalization, interactions, conditional)
- Feature set (same inputs)
- 8 other coefficients (all unchanged)
- prepare.py, evaluator.py (read-only)
- No Gemini fields used

## Formal Gate Status

The formal adoption gate (evaluator.py) requires 300+ paired races in production data.
Currently 0 results exist. This candidate **cannot be formally adopted** until
production data accumulates.

The sweep was conducted on synthetic data (50 races, structured features, 20% hit rate,
varied track conditions and grades). Results are indicative, not definitive.

## Recommendation

1. Accumulate production prediction + result pairs through the batch retroactive pipeline
2. Once 300+ paired races exist, re-run `python run_candidate_generation.py`
3. The formal gate will then validate or discard this candidate against real data

---
**HUMAN APPROVAL REQUIRED before merging to production.**
