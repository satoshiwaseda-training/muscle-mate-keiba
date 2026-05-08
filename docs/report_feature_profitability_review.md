# 収益性改善レポートの実装候補レビュー

## 結論

提示された改善案のうち、現行データで ROI 改善が確認できたのは「騎手・調教師の成績を市場分散 top3 の選定に強く使う」方向だった。

G1/G2 の 102 レース、買い方は単勝3点 + 馬連BOX3点 = 600円/R で評価した。

| 候補 | ROI | 現行差分 | PnL | 備考 |
|---|---:|---:|---:|---|
| 現行 baseline | +12.9% | - | +7,880円 | G1/G2 市場分散 |
| training_condition | +15.8% | +3.0pp | +9,690円 | G1には有効、G2は変化なし |
| feature_only_jockey_trainer_combo | +25.9% | +13.0pp | +15,860円 | 最良。ただし実験扱い |
| relative_weight | +12.3% | -0.6pp | +7,530円 | 改善せず |
| stable_body_delta | +8.2% | -4.7pp | +5,000円 | 悪化 |
| combined_conservative | +8.9% | -4.0pp | +5,430円 | 悪化 |

## グレード別

### G1

- baseline: ROI +0.1%
- training_condition: ROI +7.8%
- feature_only_jockey_trainer_combo: ROI +22.8%

### G2

- baseline: ROI +20.8%
- feature_only_relative_weight: ROI +25.8%
- feature_only_jockey_trainer_combo: ROI +27.8%
- strong_combined_conservative: ROI +21.4%

## 注意点

- `feature_only_jockey_trainer_combo` は過去データでは強いが、上位2本の高配当を除いた ex-big2 ROI は -8.8%。現行 baseline の -22.1% より改善しているが、まだ頑健とは言い切れない。
- パドック全文抽出は ROI を悪化させたため、本番の予測順位に入れない。
- 相対斤量・馬体重変化・血統/外厩の単純加点は、今回の評価では改善しない。

## 実装方針

本番の既定ロジックはすぐには置き換えない。まず以下を実装候補とする。

1. G1/G2 の市場分散 top3 に対し、各人気帯の候補選定で `jockey_win_rate` と `trainer_win_rate` を優先する実験戦略を追加する。
2. UI では「実験: 騎手/調教師優先」として現行 top3 と並べて表示し、実買い推奨にはしない。
3. 100レース相当の paper trading で ROI、馬連的中、ex-big2、最大ドローダウンを確認してから既定化を判断する。
