# G2 Market Bucket Strategy Recheck — 2026-05-09

## Question

Does the fixed G2 strategy `1-3 / 4-7 / 8+` actually help, or was it
overfit to the idea that a 4-7人気 horse will always be useful?

## Data

- Source: local backtest predictions + results
- Scope: G2 only
- Sample: 63 races
- Bet assumptions:
  - 馬連 only: 3 BOX tickets, 300 yen per race
  - legacy mixed: 単勝3 + 馬連3, 600 yen per race

## G2 Market Reality

Winner market bucket:

| Bucket | Wins | Share |
|---|---:|---:|
| 1-3人気 | 33 | 52.4% |
| 4-7人気 | 20 | 31.7% |
| 8人気以下 | 10 | 15.9% |

Top-2 pair market buckets:

| Pair Bucket | Count | Share |
|---|---:|---:|
| 1-3 + 4-7 | 28 | 44.4% |
| 1-3 + 1-3 | 14 | 22.2% |
| 1-3 + 8+ | 13 | 20.6% |
| 4-7 + 8+ | 4 | 6.3% |
| 4-7 + 4-7 | 4 | 6.3% |

The middle bucket matters, but it is not guaranteed. More importantly,
choosing the correct horse inside that bucket is the hard part.

## 馬連 Only Result

| Strategy | Hit | ROI | ROI ex-top2 payouts |
|---|---:|---:|---:|
| win_prob | 13/63 | -56.9% | -67.0% |
| diversified_1-3_4-7_8+ | 4/63 | +3.0% | -84.2% |
| tight_1-2_3-5_6+ | 9/63 | -35.7% | -65.1% |
| loose_1-4_5-9_10+ | 5/63 | -39.2% | -86.8% |
| mid_heavy_1-2_3-6_7+ | 8/63 | -40.4% | -72.2% |
| wide_1-3_4-8_9+ | 4/63 | -18.2% | -90.4% |

`diversified_1-3_4-7_8+` is not robust for 馬連. It is positive only
because of two large payouts.

## Mixed 600 Yen Result

For 単勝3 + 馬連3, diversified looked strong:

| Strategy | ROI | 単勝 Hit | 馬連 Hit |
|---|---:|---:|---:|
| diversified_1-3_4-7_8+ | +20.8% | 28/63 | 4/63 |
| win_prob | -48.2% | 30/63 | 13/63 |

This explains the earlier false confidence: the strategy was helped by
単勝 payouts, not by stable 馬連 performance.

## Decision

Do not promote fixed `1-3 / 4-7 / 8+` to G2 first candidate for 馬連.

Implementation decision:

- 第一候補: baseline `win_prob` top3
- 実験候補: market-bucket + jockey/trainer candidate
- Save both under `prediction_variants` and evaluate after results

This preserves the hypothesis without pretending it is already proven.

LOOSE trigger thresholds are unchanged.
