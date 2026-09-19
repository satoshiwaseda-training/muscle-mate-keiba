# 2026-05-10 NHKマイル G1 Post-race Review

## Result

- race_id: `202605020611`
- race: NHKマイル (G1)
- venue/course: 東京 芝1600m
- source: netkeiba result page / local `scraper.fetch_result_netkeiba`

| Finish | Horse | Odds | Market Rank | Jockey | Horse Weight |
|---:|---|---:|---:|---|---|
| 1 | ロデオドライブ | 4.6 | 1 | レーン | 476(0) |
| 2 | アスクイキゴミ | 8.4 | 4 | 戸崎圭 | 488(-4) |
| 3 | アドマイヤクワッズ | 9.7 | 6 | 坂井 | 478(-4) |

Payouts:

| Bet | Payout |
|---|---:|
| 単勝 | 460 |
| 馬連 | 1,710 |
| 馬単 | 2,990 |
| 3連複 | 6,160 |
| 3連単 | 26,150 |

## Prediction-log status

No saved live prediction for `202605020611` was found in the local
`data/live_predictions.json` or `data/prediction_archive`.

This is the biggest operational failure for today. The exact comparison
between 第一候補 and 実験候補 cannot be audited without a pre-race
`prediction_variants` snapshot. A result-only review can teach market
structure, but it cannot prove whether the model improved ROI.

## Betting Simulation From Result Shape

If either saved candidate top3 contained both `ロデオドライブ` and
`アスクイキゴミ`, the 3-horse 馬連BOX would have hit:

| Strategy | Cost | Payout | Profit | ROI |
|---|---:|---:|---:|---:|
| 馬連BOX 3点 | 300 | 1,710 | +1,410 | +470.0% |
| 単勝3点 + 馬連BOX3点 | 600 | 2,170 | +1,570 | +261.7% |

The second row assumes `ロデオドライブ` was also one of the three win
targets, so the 460 yen win payout is added.

## What the result teaches

Today's top three were market ranks 1, 4, and 6. That means two things
at the same time:

- A blind 4-7人気 emphasis is not a standalone strategy. The winner was
  still the shortest price horse in the final result table.
- A pure market-top3 candidate would likely include `ロデオドライブ`,
  `エコロアルバ`, and `ダイヤモンドノット`, but it would miss the 馬連
  partner `アスクイキゴミ`.

So the useful lesson is not "always buy 4-7人気." It is:

> G1/G2 馬連 profitability depends on whether the second slot can find a
> justified non-top3 market horse without abandoning the strongest anchor.

## Improvement Applied

The app now performs a visible prediction-log coverage audit for the
selected race date:

- counts target G1/G2 races from the same `scraper.LIVE_GRADE_FILTER`
  used by prediction generation
- counts saved live prediction logs for that date
- counts result-attached predictions
- raises a red warning listing missing race IDs when a target race has
  no saved prediction

This prevents today's failure mode from being silent: if NHKマイル is in
the target list but `202605020611` is not logged, the app now says so
before any post-race ROI review is attempted.

## Next Action

Do not change STRICT or LOOSE trigger thresholds from this one race.
For the experimental candidate, keep testing whether the non-top3
market companion is selected by public facts such as jockey fit,
trainer form, course suitability, weight delta, and fact consensus.

The minimum useful next sample is 20 G1/G2 races with:

- 第一候補 top3
- 実験候補 top3
- final result
- 馬連BOX hit/miss for each candidate
- market rank of the first and second finishers

Only that sample can answer whether the mid-market companion rule is
ROI-positive rather than a story fitted to a small number of races.
