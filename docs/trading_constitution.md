# 🏁 Horse Racing AI 運用憲法（Live Trading Constitution）

> **「ルールを守れるかどうかが、勝てるかどうか」**

このドキュメントは本システムの運用上の最上位規範である。
実装の詳細よりも優先される。運用中に本憲法と実装が矛盾した場合、
実装を修正する。


## 0. PURPOSE（最優先）

本システムの目的は：

> **競馬予想ではなく、期待値（ROI）を最大化する投資戦略の構築である**

的中率の向上ではなく、**長期的な収益性（ROI）** を最優先とする。

- 短期の的中率は目的ではない
- トリガー発火頻度も目的ではない
- "勝った気持ち" は指標ではない
- **ROI、それだけが指標である**


## 1. FUNDAMENTAL PRINCIPLE（基本原則）

### 1.1 市場は強い
- オッズは最も強力なベースラインである
- 市場を超えるには「ズレ」が必要
- オッズどおりに賭けても **takeout（−20%）** に負けるだけ

### 1.2 ズレはコストを伴う
- ズレるほど短期的な精度は下がる
- ROI は一時的に悪化する可能性がある
- 勝率低下は **期待値改善の副作用** として許容される

### 1.3 勝てるのは「条件付き」
- 全レースで勝つことは不可能
- 勝てるレースのみを選択する
- 条件を満たさないレースは **ベットしない** が正解


## 2. SYSTEM STRUCTURE（システム構造）

本システムは以下の二層構造とする：

### 2.1 STRICT（監査用）

- 高信頼シグナル
- 説明責任用途
- rare trigger（50レースに数頭しか発火しない）
- **投資判断には使わない**
- UI 表示は `fact-mode confident override` ラベル

**目的**: 「なぜ勝てた/負けた」を後から追跡可能にする。

### 2.2 LOOSE（運用用）

- 実際の投資候補
- 統計検証対象
- ROI 評価対象
- UI 表示は `experimental betting rule` ラベル

**目的**: 実際に賭ける。

### 2.3 二層は混ぜない
- STRICT と LOOSE の KPI は完全に独立管理
- LOOSE の検証結果で STRICT の閾値を動かさない
- STRICT の発火数を理由に LOOSE を変更しない


## 3. LOOSE TRIGGER RULE（固定ルール）

以下の条件を **すべて** 満たす馬のみを「投資候補」とする：

```python
def trigger_loose_capped(horse: dict) -> bool:
    return (
        horse["consensus_count"] >= 1
        and horse["composite_condition"] >= 0.60
        and horse["odds"] is not None
        and horse["odds"] <= 15.0
        and not horse.get("strong_negative_present", False)
    )
```

### 各条件の意味

| 条件 | 意味 | 理由 |
|---|---|---|
| `consensus_count >= 1` | 2ソース以上から同じカテゴリ・極性のファクトが取得できている | 単一情報源のノイズを排除 |
| `composite_condition >= 0.60` | 正のファクトが負を上回っている | 明確なポジティブシグナルのみ |
| `odds <= 15.0` | オッズ15倍以下 | softmax floor による過剰longshotベットを防止 |
| `strong_negative_present == False` | 高信頼な負ファクト（conf > 0.6）が存在しない | 矛盾ある馬は賭けない |

### Rule Version

**`cons>=1_comp>=0.60_odds<=15_no_strongneg_v1`**

このバージョン文字列を毎 loose bet のログに保存する。
ルールを変更した場合、必ずバージョン文字列も更新する。


## 4. OPERATIONAL DISCIPLINE（運用規律）

### 4.1 賭けの実行ルール
- 1 loose bet = 100 yen flat
- 条件を満たす馬すべてにベット
- 条件を満たさない馬にベットしない
- **「今日は特別だから」は禁止**

### 4.2 介入禁止
- 途中経過で閾値を動かさない
- 「たまたま負けている」を理由に撤退しない
- ドローダウン中に戦略変更しない
- n=100 を超えるまで戦略の有効性判断を保留する

### 4.3 記録義務
毎 loose bet は以下をログに残す：

- `race_id`, `race_date`, `race_name`
- `horse` name, `odds`
- `consensus_count`, `composite_condition`
- `source_count`, 各 state score
- `strong_negative_present`
- `loose_trigger_reason`
- `loose_rule_version`
- 結果（`WIN` / `LOSS` / `PENDING`）
- `payout`, `pnl`


## 5. SUCCESS METRICS（成功指標）

### 5.1 プライマリ指標（最重要）

```
loose_bet_roi  ≥  −20.0%   （JRA 単勝 takeout floor）
```

これを **100 bets 以上** のサンプルで安定して達成できたら、
このルールは有効と判定する。

### 5.2 サブ指標

| 指標 | 目標 | 理由 |
|---|---|---|
| `loose_vs_odds_roi_delta` | ≥ 0 | オッズベースラインを超える |
| `loose_vs_model_roi_delta` | ≥ 0 | 自モデル top-1 フラット比で劣化しない |
| `loose_bet_win_count` | ≥ 1 per 50 races | 発火が実在し得る |
| `n_with_result` | ≥ 100 | 統計的に語れる |

### 5.3 評価しない指標
- 日次 ROI（ノイズすぎる）
- 1 レース単位 ROI（1 回の大穴で歪む）
- 的中率単体（目的関数ではない）
- トリガー発火頻度（量ではなく質）


## 6. TERMINATION CRITERIA（撤退条件）

以下に該当する場合、このルールを **停止** する：

- 200 bets 到達時点で ROI が `−25%` を下回っている（takeout より 5pp 悪い）
- 連続 50 bets で `0/50` ヒット（完全な信号喪失）
- 依存するデータソースが 2 つ以上同時にダウン
- `loose_rule_version` を変更する正当な根拠が発生した場合


## 7. CHANGE CONTROL（変更管理）

### 7.1 変更できるもの
- ソース追加（収集を広げる）
- ファクト抽出辞書の拡張
- バグ修正

### 7.2 変更できないもの（このドキュメントの更新なしには）
- 4 つの loose 条件の数値
- 賭け額（100 yen flat）
- 成功指標の定義
- 撤退条件

### 7.3 ルール変更時の義務
- `LOOSE_RULE_VERSION` 文字列を必ずインクリメント
- 古いバージョンのログを archive に移動
- 新バージョンは最低 100 bets までは結果を信じない


## 8. THE ONE SENTENCE（一行で）

> **「ルールを守れるかどうかが、勝てるかどうか」**

的中率ではなく ROI、
発火数ではなく品質、
"気持ち" ではなくルール、
"今日は特別" ではなく **毎日同じ規律**。

---

## Appendix: 関連ファイル

| ファイル | 役割 |
|---|---|
| `dual_mode_scoring.py` | `trigger_loose_capped` 関数 + `LOOSE_RULE_VERSION` |
| `live_pipeline.py` | 毎 live 予想時に loose 評価を実行 |
| `prediction_log.py` | loose bet の永続化と KPI 計算 |
| `app_live.py` | Loose Trigger KPI ダッシュボード |
| `trigger_evaluation.py` | オフラインでの loose 戦略検証 |
| `docs/trading_constitution.md` | **この文書（運用憲法）** |
