# 2026-05-09 京都新聞杯 G2 Post-race Review

## Result

- race_id: `202608030511`
- race: 京都新聞杯 (G2)
- venue/course: 京都 芝2200m
- source: netkeiba result page / local `scraper.fetch_result_netkeiba`

| Finish | Horse | Odds | Jockey | Horse Weight |
|---:|---|---:|---|---|
| 1 | コンジェスタス | 13.7 | 西村淳 | 518(-6) |
| 2 | ベレシート | 2.1 | 北村友 | 480(-2) |
| 3 | ラディアントスター | 31.6 | 池添 | 514(-10) |

Payouts:

| Bet | Payout |
|---|---:|
| 単勝 | 1,370 |
| 馬連 | 1,540 |
| 馬単 | 4,310 |
| 3連複 | 13,940 |
| 3連単 | 74,860 |

## Prediction-log status

No saved live prediction for `202608030511` was found in the local
`data/live_predictions.json` or `data/prediction_archive`.

This means the exact "predicted top3 vs result" comparison cannot be
audited from the current local log. Treat this as an operational miss:
future post-race review is only useful if `prediction_variants.primary`
and `prediction_variants.experimental` are persisted before the race.

## What the result teaches

The result fits the current G2 market-bucket thesis:

| Horse | Result | Odds | Market Bucket |
|---|---:|---:|---|
| ベレシート | 2 | 2.1 | 1-3人気帯 |
| コンジェスタス | 1 | 13.7 | 4-7人気帯 |
| ラディアントスター | 3 | 31.6 | 8人気以下 |

So the main lesson is not "abandon G2 diversification." The winning
馬連 required one horse from the top market bucket and one from the
middle bucket. The design is directionally right; the risk is choosing
the wrong horse inside the 4-7人気帯.

## Failure modes to check against the saved UI prediction

When the actual saved prediction is available, evaluate both candidates:

1. If 第一候補 contained `ベレシート` but not `コンジェスタス`, the miss was
   intra-bucket selection inside G2's 4-7人気帯.
2. If 実験候補 also missed `コンジェスタス`, jockey/trainer-only scoring did
   not capture the winner profile strongly enough.
3. If either candidate contained both `コンジェスタス` and `ベレシート`, the
   馬連BOX 3点 would have hit: payout 1,540 yen against 600 yen cost,
   profit +940 yen, ROI +156.7% for that candidate.
4. If neither candidate included `ベレシート`, the model overrode a very
   strong market signal too aggressively.

## Actionable learning

- Keep G2 bucket diversification. Today validated the need for the
  `1-3 / 4-7 / 8+` spread.
- Do not change LOOSE trigger thresholds from one race. This remains
  governed by `docs/trading_constitution.md`.
- Add review pressure on the 4-7人気帯 selector:
  - recent win streak / unbeaten signal
  - positive 2200m stamina profile
  - jockey/trainer compatibility
  - negative filter for large body-weight drop only when backed by a
    public negative source
- Treat missing prediction logs as a blocking review error. A result
  without a saved `prediction_variants` snapshot cannot produce a clean
  ROI learning loop.

## Next Review Requirement

Before changing scoring weights, collect at least 20 G2 races with:

- first candidate top3
- experimental candidate top3
- final result
- 馬連BOX hit/miss for each candidate
- whether winner was in 1-3, 4-7, or 8+ market bucket

Only then decide whether the 4-7人気帯 selector should add a new
non-threshold feature or just improve display/persistence.
