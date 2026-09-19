# Corrosion Depth Estimation — 深さ推定計画書 v1

**作成日**: 2026-04-29
**プロジェクト**: Corrosion Detector の拡張機能 (Phase 7+)
**前提**: v8 Final で確定した SAM2 検出パイプラインを上流とし、検出された腐食領域に対して「深さ」を推定する
**作成者**: サトシ (オーナー単独承認モード)

---

## 1. 目的とスコープ

### 1.1 やりたいこと

検出された腐食領域に対して、以下のいずれかの形で **深さ** を出力する:

| 出力形式 | 内容 | 難易度 | 実現可能性 |
|---------|------|--------|-----------|
| (A) **クラス分類** | 浅 / 中 / 深 の 3 段階 | 低 | 🟢 高 |
| (B) **相対深さマップ** | ピクセル単位の相対値 (0-1) | 中 | 🟡 中 |
| (C) **絶対深さ (mm)** | 健全表面からの mm 単位距離 | 高 | 🔴 補助情報必須 |

### 1.2 スコープの確定 (v1)

**運用優先・現実重視で次の段階を採る**:

- **Phase 7.1** (まず): クラス分類 (A) — 検査員の意思決定支援
- **Phase 7.2** (次): 相対深さマップ (B) — 詳細レビュー支援
- **Phase 7.3** (条件付き): 絶対深さ (C) — ArUco マーカーが使える場面のみ
- **Phase 7.4** (将来): 腐食特化モデル fine-tune

絶対深さ (C) を最初から目指さない理由は §2 で詳述。

### 1.3 v1 の主指標

| Phase | 主指標 | 採用ゲート |
|-------|-------|----------|
| 7.1 (クラス) | macro F1 | F1 ≥ 0.70 |
| 7.2 (相対) | Spearman 相関 (順序保存) | ρ ≥ 0.6 |
| 7.3 (絶対) | MAE (mm) | 浅腐食 ≤ 1mm、深腐食 ≤ 3mm |
| 7.4 (fine-tune) | 上記の改善幅 | Δ95% CI 下限 > 0 |

---

## 2. 単眼画像の物理的限界 — 計画の出発点

腐食深さ推定の難しさは、技術選定の前に**前提として共有しておく必要がある**:

### 2.1 1 枚の画像から絶対深さは原理的に決まらない

- 画像の各ピクセルは「どの方向の光が来たか」しか記録しない
- カメラから物体までの距離 (Z) と物体サイズの積が一定であれば、画像は同じになる
- → **スケールを与える追加情報なしには絶対 mm は出せない**

### 2.2 必要な補助情報の選択肢

| 補助情報 | 精度 | 実用性 (検査現場) | コメント |
|---------|------|-----------------|---------|
| ArUco マーカー (既知サイズ) | 高 | 中 (撮影時に置く) | 平面腐食なら直接 mm 換算可能 |
| 超音波厚さ計 (UT) | 最高 | 高 (既に普及) | 画像と組み合わせて校正データに |
| カメラ + 距離計 (LiDAR スマホ等) | 中 | 中 | iPhone Pro / iPad Pro 限定 |
| 多視点撮影 (NeRF/Gaussian Splatting) | 高 | 低 (撮影手順複雑) | 完全 3D 復元、長期計画 |
| 同一場所の検査履歴 (画像差分) | 中 | 低 (履歴必須) | 減肉「進行量」のみ、初期値不明 |
| **何もなし (画像 1 枚のみ)** | **相対のみ** | 高 | 順序関係しか出ない |

### 2.3 v1 の戦略

**画像 1 枚のみで出せる範囲** (クラス分類 + 相対深さ) を主軸にし、ArUco マーカーや UT データが使える場面では絶対深さに拡張する **二段運用**。

---

## 3. 段階的アプローチ

### 3.1 Phase 7.1 — クラス分類 (浅 / 中 / 深)

#### 狙い

腐食領域の「深さレベル」を 3 値で分類。検査員が「優先順位を付けるための材料」として使う。

#### 入力

- 既存 SAM2 パイプラインが出した polygon
- polygon でクロップした画像領域
- (任意) 周辺の健全表面のテクスチャ

#### 出力

```json
{
  "depth_class": "shallow" | "moderate" | "deep" | "unknown",
  "confidence": 0.85
}
```

#### 候補手法

| 手法 | 説明 | 評価 |
|------|------|------|
| **(A1) 軽量 CNN (ResNet18 / MobileNetV3 base)** | 腐食クロップを ImageNet 事前学習済みモデルで分類 | 🟢 ベースライン |
| (A2) ViT-small base + LoRA fine-tune | Transformer ベース、データ少なくても効く | 🟡 データ揃ってから |
| (A3) 影・ハイライト統計 + 古典 ML (LightGBM) | 学習データ少ない時の手段 | 🟢 並行で試す |
| (A4) GPT-4V / Claude Sonnet vision | LLM の常識的判断 | 🔴 v8 Final で LLM 排除した方針と矛盾、却下 |

**v1 採用**: A1 (軽量 CNN) と A3 (古典 ML) を並行で評価、F1 が高い方を採用。

#### 必要データ

- 画像 + polygon + 深さクラスラベル: **300+ サンプル** (浅 100 / 中 100 / 深 100、クラス均衡)
- ラベル付与は検査員が UT データや経験で決定 (ラベリングルールは §6 で詳述)

#### 期待精度

- F1 ≥ 0.70 をゲート
- ベースライン (ランダム): F1 ≈ 0.33
- 経験則: 軽量 CNN + 300 枚で F1 0.65-0.80 が一般的

---

### 3.2 Phase 7.2 — 相対深さマップ (単眼深度モデル流用)

#### 狙い

腐食領域内のピクセルごとに「どこが深いか」の相対マップを出す。
**絶対値ではなく、領域内の最深箇所がどこかを可視化**する。

#### 候補手法

| 手法 | モデル | 商用ライセンス | 評価 |
|------|--------|--------------|------|
| **(B1) Depth Anything V2 (small)** | LiheYoung/Depth-Anything-V2 | Apache-2.0 | 🟢 採用候補 |
| (B2) MiDaS v3.1 (DPT-Hybrid) | intel-isl/MiDaS | MIT | 🟢 採用候補 (代替) |
| (B3) ZoeDepth | isl-org/ZoeDepth | MIT | 🟡 metric depth 出すが屋内/屋外特化 |
| (B4) Marigold | prs-eth/marigold | Apache-2.0 | 🟡 高精度だが推論重 |

**v1 採用**: B1 (Depth Anything V2 small) — 軽量、Apache-2.0、ゼロショット高精度。

#### 処理フロー

```
[画像]
   ↓ SAM2
[polygon]
   ↓ Depth Anything V2 (画像全体に推論)
[depth_map (相対)]
   ↓ polygon でマスク
[腐食領域のみの depth_map]
   ↓ 統計
[depth_min / depth_max / depth_mean / depth_p95]
```

#### 出力

```json
{
  "depth_map_b64": "data:image/png;base64,...",   // 可視化用 PNG
  "relative_depth": {
    "min":  0.12,
    "max":  0.87,
    "mean": 0.43,
    "p95":  0.78
  },
  "model": "depth_anything_v2_small"
}
```

`relative_depth` は 0-1 の相対値 (画像内の最も近い点が 0、最も遠い点が 1)。

#### 評価

- val 80 枚に「目視で深さ順位」を付けてもらう (5 段階)
- モデル出力との **Spearman 順位相関** を計算
- ゲート: ρ ≥ 0.6

#### 注意点

- Depth Anything は屋外シーン主体に学習されている → 配管・構造物の近接撮影はドメインミスマッチの可能性
- 並行で MiDaS も試して比較
- 結果が悪ければ Phase 7.4 (fine-tune) で改善

---

### 3.3 Phase 7.3 — 絶対深さ (ArUco / UT 併用、条件付き)

#### 前提

検査員が以下のいずれかを実施できる現場のみ対象:

- 画像内に **ArUco マーカー** (既知サイズ、5cm × 5cm 推奨) を貼って撮影
- 同じ場所の **超音波厚さ計 (UT)** データを添付できる
- 既知サイズの物体 (定規、ケーブル直径等) が写っている

#### 手法

##### B1. ArUco スケール法 (画像のみで完結)

```
1. ArUco 検出 (OpenCV aruco.detectMarkers)
2. マーカーピクセルサイズから mm/px 換算を計算
3. 平面仮定で polygon の 横方向の物理サイズを推定
4. 相対深さマップ (Phase 7.2) を mm スケールに換算
   → 仮定: 画像の depth 軸の Z 範囲はマーカー平面 ± 数 mm の腐食凹凸
5. 出力: 腐食領域の最大深さ推定 (mm)
```

##### B2. UT 校正法 (実測値とのフィッティング)

```
1. 同じ画像 + UT データ (例: 6 点で測定) のペアを蓄積 (50+ ペア)
2. 各 UT 点を画像座標に対応付け (検査員が画像にマーカー)
3. 相対深さ → 実測 mm の回帰モデル (LightGBM / 線形回帰) を学習
4. 推論時: 相対深さ → 推定 mm に変換
5. 信頼区間も併記 (校正データのばらつき)
```

##### B3. 多視点 (将来検討)

LiDAR + iPhone Pro / NeRF 多視点撮影で 3D メッシュを作る方法。撮影手順の負担が大きいので v1 では非採用、v2+ で検討。

#### 評価指標

- ペア (画像 + UT) を test に分けて MAE 計算
- 浅腐食 (< 2mm): MAE ≤ 1mm が目標
- 深腐食 (> 5mm): MAE ≤ 3mm が目標
- 信頼区間 95% で予測値が UT 実測値を含む割合 ≥ 0.80

---

### 3.4 Phase 7.4 — 腐食特化 fine-tune (将来課題)

データが集まった段階 (UT ペア 200+) で:

- Depth Anything V2 small に LoRA fine-tune
- 入力: 腐食画像、出力: depth_map (mm 単位)
- 期待: ドメインミスマッチ解消、絶対 MAE 改善
- コスト: GPU 1 時間 ($2)

ただし **Phase 7.1 / 7.2 / 7.3 で十分な精度が出ていればスキップ可能**。

---

## 4. アーキテクチャ統合

既存の SAM2 検出パイプラインに **「深さ推定モジュール」** を追加する。

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1: SAM2 検出 (既存、v8 Final)                          │
│  → polygon + sam2_predicted_iou + color_score               │
└───────────────────┬──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│  Stage 2: 後処理 (既存、NMS / マージ / 最小面積)              │
└───────────────────┬──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│  Stage 3 (NEW): 深さ推定                                       │
│  - depth_class: ResNet18 で polygon クロップ分類              │
│  - relative_depth: Depth Anything V2 で全体推論 → polygon マスク │
│  - (任意) absolute_depth: ArUco / UT で校正                  │
└───────────────────┬──────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────────┐
│  Stage 4: 出力 (既存 + 深さ情報を追加)                        │
└──────────────────────────────────────────────────────────────┘
```

### 実装ファイル (予定)

```
api/depth/
├── __init__.py
├── classifier.py         # Phase 7.1: 軽量 CNN クラス分類
├── monocular_depth.py    # Phase 7.2: Depth Anything V2 ラッパー
├── aruco_scale.py        # Phase 7.3: ArUco マーカー検出 + mm 換算
└── ut_calibration.py     # Phase 7.3: UT データとの回帰モデル

api/detectors/hybrid_detector.py
  └─ Stage 3 で api.depth.* を呼び出す
```

### API レスポンス拡張 (例)

```json
{
  "detections": [
    {
      "polygon": [[y, x], ...],
      "confidence": 0.92,
      "subtype": "pitting",
      "sam2_predicted_iou": 0.95,
      "color_score": 0.78,
      "depth": {
        "class":            "moderate",
        "class_confidence": 0.81,
        "relative":         {"min": 0.12, "max": 0.87, "mean": 0.43},
        "absolute_mm":      {"value": 2.3, "ci_low": 1.8, "ci_high": 2.9, "method": "aruco"}
      }
    }
  ]
}
```

`absolute_mm` は ArUco / UT が使える場合のみ、それ以外は `null`。

---

## 5. データ要件

### 5.1 ラベル形式

各画像 + polygon に対して、以下のいずれかを付与:

```json
{
  "image_id": "ccs_train_42",
  "polygon": [[y, x], ...],

  "depth_class": "moderate",          // Phase 7.1 用 (必須)
  "depth_class_source": "ut_measurement" | "expert_judgment",

  "depth_rank_within_image": 2,        // Phase 7.2 用 (任意、画像内の順位 1-N)

  "depth_mm": 2.3,                     // Phase 7.3 用 (ArUco/UT 利用時のみ)
  "depth_mm_source": "ut_5pt_avg" | "aruco_estimated",
  "depth_mm_uncertainty_mm": 0.4
}
```

### 5.2 データ量目標

| Phase | ラベル種類 | 必要数 | 取得方法 |
|-------|----------|--------|---------|
| 7.1 | depth_class | 300+ (各クラス 100) | 検査員 / 経験則 |
| 7.2 | depth_rank_within_image | 80+ (val 全部) | 検査員目視 |
| 7.3 | depth_mm | 50+ ペア | ArUco 撮影 or UT 検査 |

### 5.3 データ取得経路

1. **既存 CCS データセットを再ラベル**: 440 枚に depth_class を付与 (検査員 + AI 補助)
2. **公開 UT データセット調査** (要 §11 の文献調査)
3. **運用画像から蓄積**: HF Space で処理した画像を opt-in で保存
4. **ラボでの基準腐食サンプル撮影** (深さ既知の参照データ作成、長期計画)

### 5.4 ラベリングルール (Phase 7.1)

| クラス | 基準 (UT 実測がある場合) | 経験則 (UT なし) |
|------|------------------------|----------------|
| **shallow** | 減肉 ≤ 1.0 mm or 元肉厚の ≤ 10% | 表面変色のみ、点食痕 < 1mm |
| **moderate** | 減肉 1.0-3.0 mm or 10-30% | 明らかな凹凸、層状剥離の初期 |
| **deep** | 減肉 ≥ 3.0 mm or ≥ 30% | 地金露出、大きな凹みあり |
| **unknown** | 判定不能 (光量不足等) | (同) |

検査員 2 名で合議、Cohen's κ ≥ 0.6 を維持。

---

## 6. 評価ハーネス拡張

既存 `tests/eval/run_eval.py` に深さ評価を追加する。

### 6.1 新指標

```python
# tests/eval/depth_metrics.py (新規)

def class_macro_f1(y_true, y_pred) -> float:
    """Phase 7.1: 3 クラス分類の macro F1"""

def spearman_within_image(y_true_ranks, y_pred_scores, image_groups) -> float:
    """Phase 7.2: 画像ごとに順位相関 → 平均"""

def metric_depth_mae(y_true_mm, y_pred_mm) -> tuple[float, float]:
    """Phase 7.3: MAE と 95% CI"""
```

### 6.2 採否ゲート (CODEX レビュー対応)

各 phase で前 variant 比 95% CI 下限 > 0 を採否条件とする (既存 v8 Final と同じ枠組み)。

`docs/baseline.md` に depth-* メトリクスを追記する。

### 6.3 release_test 凍結ルールの継承

- val: 開発中の比較に使う
- release_test: Phase 7 完了時 1 回のみ評価
- monitoring_set: 運用後の継続監視

---

## 7. 想定リスクと回避策

### 7.1 単眼の物理的限界

| リスク | 確率 | 影響 | 回避策 |
|-------|------|------|------|
| 絶対深さの精度が要件を満たさない | 高 | 大 | Phase 7.3 を「ArUco 必須」に縛り、無理な無補助推定はしない |
| 屋外シーン特化の Depth Anything が腐食でドメインミスマッチ | 中 | 中 | Phase 7.4 で fine-tune、または MiDaS と比較 |
| 照明条件でクラス分類精度が大きく変動 | 中 | 中 | 照明スライス評価、データ拡張 (各種輝度補正) |

### 7.2 データ取得困難

| リスク | 確率 | 影響 | 回避策 |
|-------|------|------|------|
| UT 実測データが集まらない (運用前) | 高 | 大 | クラス分類 (Phase 7.1) は経験則ラベルで進める、絶対深さは後回し |
| 検査員 2 名合議のコスト | 中 | 中 | 1 名でラベル → 別 1 名がスポットチェック (10%) で代用 |
| 公開データセットに depth がない | 高 | 大 | 既存 CCS は corrosion mask のみ、新規ラベリングが必要 |

### 7.3 アーキテクチャ統合

| リスク | 確率 | 影響 | 回避策 |
|-------|------|------|------|
| 推論時間が長くなりすぎる (CPU 30s → 90s) | 中 | 中 | Depth Anything V2 small (~50ms/画像) を使う、CPU でも軽い |
| メモリ不足 (HF Space 16GB) | 低 | 中 | small モデル前提、必要なら lazy load |
| API レスポンス互換性 | 低 | 小 | depth フィールドは optional に追加 |

---

## 8. マイルストーン (3-4 ヶ月の段階計画)

### Month 1 — Phase 7.1 (クラス分類 + データ準備)

| 週 | アクション |
|----|---------|
| W1 | 文献調査 (Depth Anything / MiDaS の腐食適用例)、ライブラリ動作確認 |
| W2 | CCS train 余り 200 枚に depth_class ラベル付与 (経験則) |
| W3 | ResNet18 + MobileNetV3 ベースライン実装、学習スクリプト |
| W4 | val で F1 評価、`baseline.md` に追記、採否判定 |

### Month 2 — Phase 7.2 (相対深さマップ)

| 週 | アクション |
|----|---------|
| W5 | Depth Anything V2 small / MiDaS の動作確認 (ローカル) |
| W6 | `api/depth/monocular_depth.py` 実装、Stage 3 統合 |
| W7 | val 80 枚に depth_rank ラベル付与、Spearman 相関評価 |
| W8 | UI に相対深さマップ表示 (canvas overlay)、運用採否判定 |

### Month 3 — Phase 7.3 (絶対深さ、ArUco)

| 週 | アクション |
|----|---------|
| W9 | ArUco マーカー (5cm) を発注 / 印刷、`api/depth/aruco_scale.py` 実装 |
| W10 | 検査現場に協力依頼、ArUco 同梱で 30+ ペア撮影 |
| W11 | UT データを取得できる場合は併用、`ut_calibration.py` 実装 |
| W12 | release_test で MAE 評価、信頼区間検証 |

### Month 4 — Phase 7.4 (任意、fine-tune)

| 週 | アクション |
|----|---------|
| W13 | Phase 7.1-7.3 の評価結果次第で実施判断 |
| W14-15 | LoRA fine-tune (GPU 環境) |
| W16 | release_test で性能改善確認、運用反映 |

### 失敗時の判断基準

- **Phase 7.1 で F1 < 0.50** → ラベル品質確認 + データ追加 → それでもダメなら手法変更
- **Phase 7.2 で Spearman < 0.30** → ドメインミスマッチが致命的、Phase 7.4 を前倒し
- **Phase 7.3 で MAE > 5mm** → ArUco 平面仮定が無理、UT 校正のみに切替

---

## 9. v8 Final との関係 (既存システムへの影響)

### 9.1 影響なし (依存関係)

- SAM2 検出パイプライン → そのまま使用
- 評価ハーネス (run_eval.py) → 深さ評価を追加するだけ
- `tests/eval/scoring_rule.md` → 凍結のまま、深さは別 scoring_rule v2 として並列

### 9.2 影響あり (拡張)

| ファイル | 変更内容 |
|---------|---------|
| `api/detectors/hybrid_detector.py` | Stage 3 で depth モジュール呼び出し |
| `api/detectors/base.py` | `Detection.meta` に `depth_class`, `depth_relative_*`, `depth_absolute_mm` 追加 |
| `public/index.html` | 検出カードに深さ表示、(任意) depth_map ヒートマップ |
| `tests/eval/datasets/labels_*.json` | `depth_*` フィールド追加 |
| `Dockerfile.hfspace` | Depth Anything V2 small ckpt の事前ダウンロード |

### 9.3 オプトイン設計

`DEPTH_ESTIMATION_ENABLED=false` 環境変数で機能を完全 OFF できるようにする (既存運用への影響ゼロ)。

---

## 10. コスト見積

| 項目 | コスト | 備考 |
|------|--------|------|
| Depth Anything V2 small ckpt (~50MB) | $0 | 公開、ダウンロード追加 |
| HF Space CPU 推論 | $0 | Free tier、推論時間 +1-2 秒 |
| ArUco マーカー印刷 (5cm × 10 枚) | ~¥500 | 一回限り |
| 学習用 GPU (Phase 7.4) | $4-8 | A100 1-2 時間 (Runpod) |
| ラベリング工数 (300 枚 × 1 分) | ~5 時間 | サトシさん本人 or 検査員 |
| **合計** | **~¥1,500 + 5 時間** | (LLM コスト無し、v8 Final 路線継続) |

---

## 11. 文献調査タスク (Phase 7.1 着手前に実施)

以下を調査して `docs/depth_estimation_literature.md` にまとめる:

- [ ] Depth Anything V2 の論文 (Yang et al. 2024) — 屋外シーン以外の精度
- [ ] MiDaS v3.1 の論文 (Birkl et al. 2023) — 近接撮影での挙動
- [ ] 腐食特化の depth 推定論文の有無 (ASCE / NDT&E International ジャーナル)
- [ ] ArUco マーカー + 平面仮定の精度限界文献
- [ ] 配管 UT 検査標準 (JIS Z 2300 / ASME B31.3) — 深さ要求精度の規格値

---

## 12. 完了の定義 (v1)

- [ ] Phase 7.1: クラス分類 (val) で macro F1 ≥ 0.70、`baseline.md` 記録
- [ ] Phase 7.2: 相対深さで Spearman 相関 ≥ 0.6
- [ ] Phase 7.3: ArUco 経由の絶対深さで浅腐食 MAE ≤ 1mm、深腐食 MAE ≤ 3mm
- [ ] UI に深さ情報が表示される
- [ ] `DEPTH_ESTIMATION_ENABLED=false` で完全 OFF できる
- [ ] release_test 1 回で最終評価
- [ ] monitoring_set 累積 50+ で 1 ヶ月安定稼働
- [ ] CODEX レビュー (本計画書 + 実装) 全項目採用済

---

## 13. v8 Final との 1 行サマリー比較

| 観点 | v8 Final (検出) | **v1 (深さ推定)** |
|------|----------------|------------------|
| 主要モデル | SAM2 (Apache-2.0) | Depth Anything V2 small (Apache-2.0) + 軽量 CNN |
| LLM 利用 | なし (v8 Final で恒久排除) | **なし (路線継続)** |
| 主指標 | critical_recall (画像単位) | macro F1 (クラス分類) / Spearman (相対) / MAE (絶対) |
| API コスト | $0 | $0 (LLM 不使用、ckpt は事前 DL) |
| 推論時間 | ~30 秒/画像 (CPU) | ~32 秒/画像 (CPU、+1-2s) |
| 物理的限界 | per_gt_recall 集合体問題 | **絶対深さは補助情報必須** (本質的) |
| 採用ゲート | val Δ95% CI 下限 > 0 + manual_review_rate ≤ 0.50 | (同じ枠組みを継承) |

---

## 14. 1 行で言うと

> **「単眼画像から絶対深さは原理的に出ないので最初から欲張らない。クラス分類 → 相対マップ → ArUco/UT 補助で絶対値、の段階で進む。LLM は今回も使わない。」**

---

## 改訂履歴

| Ver. | 日付 | 変更内容 |
|------|------|--------|
| v1 | 2026-04-29 | 初版 — 4 段階アプローチ、v8 Final アーキテクチャと統合、CODEX レビュー想定で採用ゲート明示 |
