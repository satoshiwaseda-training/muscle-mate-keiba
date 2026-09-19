# Corrosion Depth Estimation — 深さ推定計画書 v3

**作成日**: 2026-04-29 (v1 初版)
**改訂日**: 2026-04-29 (**v3 — CODEX 第 2 次レビュー全 7 項目反映**)
**プロジェクト**: Corrosion Detector の拡張機能 (Phase 7+)
**前提**: v8 Final で確定した SAM2 検出パイプラインを上流とし、検出された腐食領域に対して「外観上の深さ傾向」または **UT 校正データがある場合は減肉量 (loss_mm)** を推定する
**作成者**: サトシ (オーナー単独承認モード)

---

## 改訂履歴

| Ver. | 日付 | 変更内容 |
|------|------|--------|
| v1 | 2026-04-29 | 初版 — 4 段階アプローチ、v8 Final と統合 |
| v2 | 2026-04-29 | CODEX 第 1 次レビュー全 8 項目反映: 深さ定義分離、Depth Anything 降格、ArUco 単独 mm 削除、古典 ML 優先、API 独立、評価分離、ライセンス精密化 |
| **v3** | 2026-04-29 | **CODEX 第 2 次レビュー全 7 項目反映**: `depth_mm` を `loss_mm` に再定義 (UT は残存肉厚を測る)、top1 pixel hit を半径ベース `top_region_hit_rate` に緩和、measured 50 件は探索ゲート / 100+ で本採用、`DepthEstimator.annotate()` を dict ベースに統一、ラベルスキーマを `region_id` 参照型に、CPU 推論時間を「要実測」に格下げ、UI 文言ルールを E2E テスト可能に |

---

## v3 の核心 — v2 から変わった 4 つの本質追加

1. **`depth_mm` を `loss_mm` に再定義 (P0-1)**: UT が測るのは「残存肉厚」であり、画像で見える「孔の深さ」ではない。`loss_mm = reference_thickness_mm - remaining_thickness_mm` を主値とし、ラベルには `measurement_device` / `probe_diameter_mm` を必須化。
2. **`top_region_hit_rate` への緩和 (P0-2)**: UT プローブ径 (6-13mm) + ピン打ち誤差を考慮し、半径 r 以内のヒットを評価する。r = max(probe_diameter_px, 15px)。
3. **measured subset の段階ゲート (P0-3)**: 50 件は **探索ゲート** (CI 確認用)、100+ で **本採用ゲート** (`macro F1 ≥ 0.70` + `bootstrap CI lower ≥ 0.55` + `各クラス recall ≥ 0.50`)。
4. **dict ベース API 統合 (P1-1)**: `DepthEstimator.annotate(image, detections: list[dict]) -> list[dict]` で `api/index.py:1991` の `to_dict()` 後に追記する。`Detection`/`base.py` には触らない。

## v2 の核心 (継続) — v1 から変わった 4 つの本質

1. **「深さ」を 2 系統に分離**: `appearance_depth_class` (見た目、UT なし可) と `measured_depth_class` (UT/実測由来のみ、学習・評価の主軸)
2. **単眼深度モデルを「腐食深さ」とは呼ばない**: 出力名は `visual_depth_hint` に降格
3. **ArUco 単独で絶対 mm 換算しない**: Phase 7.3 は UT 校正主軸、ArUco は補助
4. **モデル順序の入れ替え**: 古典 ML → CNN

---

## 1. 目的とスコープ

### 1.1 やりたいこと

検出された腐食領域に対して、以下を出力する:

| 出力 | 内容 | UT 実測の必須度 |
|------|------|---------------|
| (A) **`appearance_depth_class`** | 浅そう / 中ぐらい / 深そう / 不明 (見た目) | 不要 (経験則 OK) |
| (B) **`measured_depth_class`** | 浅 / 中 / 深 (実測由来) | 必須 (UT/プロファイル/試験片) |
| (C) **`visual_depth_hint`** | 領域内の相対深度ヒント (0-1 マップ + 統計) | 不要 (補助情報) |
| (D) **`estimated_loss_mm`** | 減肉量推定 (mm + CI + method) | 必須 (UT 校正データ) |

### 1.2 スコープの確定 (v2)

| Phase | 出力 | 必須補助情報 |
|-------|------|-------------|
| **7.0** | 深さラベル定義 + scoring rule + ラベルスキーマ凍結 | (準備) |
| **7.1a** | A: appearance_depth_class (古典 ML、ベースライン) | なし |
| **7.1b** | A: appearance_depth_class (軽量 CNN、7.1a 超えた場合のみ) | なし |
| **7.2** | C: visual_depth_hint (Depth Anything V2 small) | なし |
| **7.3** | D: estimated_loss_mm (UT 校正主、ArUco 補助) | UT 実測 ペア |
| **7.4** | B: measured_depth_class (LoRA fine-tune、将来) | UT 実測 200+ ペア |

**重要**: A (appearance) と B (measured) は **別ラベル系**として扱う。A の良い精度は B の保証にならない。

### 1.3 v2 の主指標

| Phase | 主指標 | 採用ゲート | 注意 |
|-------|-------|----------|------|
| 7.1a/b | `appearance_depth_class` の **measured subset** での macro F1 | F1 ≥ 0.70 | 経験則 only ラベルの F1 は補助 |
| 7.2 | `deepest_point_top1_hit_rate` (UT 最深点が hint top-K に入る率) | top1 ≥ 0.50、top3 ≥ 0.80 | 「目視 5 段階」は v1 で却下 |
| 7.2 | `within_region_rank_corr` (Spearman、UT 複数点で順位) | ρ ≥ 0.5 | 補助指標 |
| 7.3 | `estimated_loss_mm` の MAE (UT 実測 vs 推定) | 浅腐食 ≤ 1mm、深腐食 ≤ 3mm | 信頼区間カバレッジ ≥ 0.80 |
| 7.4 | 上記の改善幅 | Δ95% CI 下限 > 0 | (将来) |

すべての指標は **同一構造物 / 同一撮影系列 / 同一腐食箇所単位で split** したホールドアウトで評価。

---

## 2. 単眼画像の物理的限界 — 計画の出発点 (v1 から拡張)

### 2.1 「シーン奥行き」と「腐食凹み深さ」は別物

| 量 | 典型スケール | 推定可能性 |
|----|------------|----------|
| シーン奥行き (Depth Anything 等が出すもの) | 数 cm 〜 数百 m | 単眼で可 (相対値) |
| 腐食孔の凹凸深さ | **0.1 mm 〜 数 cm** | 単眼ほぼ不能 |

Depth Anything V2 は KITTI / NYU 等の屋外・屋内シーンで学習。腐食孔の数 mm 凹凸を表現する訓練信号は含まれない。
→ **「Depth Anything が出す値を mm に換算する」発想を捨てる**。出せるのは「同一画像内で **どこが手前 / 奥か** の相対順位」のみ。

### 2.2 ArUco の役割を限定する

| ArUco から得られる | 得られない |
|-----------------|----------|
| 横方向 mm/px スケール | Z 軸 (奥行き / 凹み) 深さ |
| カメラに対する平面姿勢 | 凹凸量 |
| 撮影距離の概算 | 局所形状 |

→ **ArUco を mm 換算に直接使うのは不可**。Phase 7.3 の用途は:
- 腐食領域の **横方向サイズ (mm²)** を確定
- 撮影姿勢補正で UT 校正データとの照合精度を上げる
- Z 深さは **必ず UT 実測** から推定

### 2.3 結論 (v2 の戦略)

- 単眼画像のみ: A (appearance) と C (hint) のみ提供、UI で「外観上の傾向」と明記
- UT 実測併用時のみ: B (measured) と D (estimated_loss_mm) を提供、 `method` フィールドで根拠を明示

---

## 3. 段階的アプローチ (v2 改訂版)

### 3.0 Phase 7.0 — 深さラベル定義 + 評価規則の凍結 (新規、CODEX P0-1 反映)

#### 成果物

1. **`docs/depth_scoring_rule.md`** (新規凍結文書)
   - `appearance_depth_class` と `measured_depth_class` の定義を明確分離
   - 評価指標の計算規則 (深さ専用)
   - split 規則 (構造物・撮影系列・腐食箇所単位)

2. **`tests/eval/datasets/labels_depth_v0.schema.json`** (v3 改訂)

```json
{
  // ─── 既存 GT への参照 (P1-2: polygon を主キーにしない) ─────────────────
  "source_label_file": "labels_v0.json",
  "image_id":          "ccs_train_42",
  "region_id":         "ccs_train_42_r0",      // 既存 labels_v0 の region と安定リンク
  "region_index":      0,                       // 同一画像内の順序 (補助参照)
  "polygon_audit_copy": [[y, x], ...],          // 監査用コピー、ズレ検知のため

  // ─── 外観ベース (UT 不要、補助・弱教師) ────────────────────────────────
  "appearance_depth_class":  "shallow_looking" | "moderate_looking" | "deep_looking" | "unknown",
  "appearance_label_source": "expert_judgment_v1" | "user_judgment",

  // ─── 実測ベース (UT/プロファイル/試験片必須、主指標) ──────────────────
  "measured_depth_class":   "shallow" | "moderate" | "deep" | "unknown",
  "measured_label_source":  "ut_5pt_avg" | "profile_meter" | "reference_specimen",

  // ─── 減肉量 (P0-1: depth_mm を loss_mm に再定義) ──────────────────────
  "loss_mm":                  2.3,              // 主値、reference - remaining
  "reference_thickness_mm":   6.0,              // 元肉厚 (設計値 or 健全部 UT)
  "remaining_thickness_mm":   3.7,              // UT 実測値
  "loss_mm_method":           "ut_5pt_avg" | "ut_grid_8x8" | "profile_meter",
  "loss_mm_uncertainty_mm":   0.4,              // 標準偏差
  "loss_mm_ci95":             [1.6, 3.0],
  "measurement_device":       "Olympus 38DL Plus",   // P0-1: 機器名必須
  "probe_diameter_mm":        9.5,              // P0-1: プローブ径必須

  // ─── UT 測定点 (P0-2: 不確実性情報を追加) ─────────────────────────────
  "ut_points": [
    {
      "px_y":                  245,
      "px_x":                  312,
      "px_uncertainty_radius": 12,              // P0-2: ピン打ち誤差 (px)
      "remaining_thickness_mm": 4.7,            // 残存肉厚
      "loss_mm":               1.3,             // 派生値 (reference - remaining)
      "probe_diameter_mm":     9.5,
      "surface_condition":     "clean" | "wet" | "scaled" | "painted",
      "remaining_pct":         78.3
    }
  ],

  // ─── Split リーク防止用 ID 階層 ───────────────────────────────────────
  "structure_id":           "site_A_pipe_3",
  "capture_session_id":     "session_2026_03_15_morning",
  "corrosion_site_id":      "site_A_pipe_3_loc_north_2"
}
```

3. **重要ルール (凍結)**:
   - **`depth_mm` フィールドは廃止** (P0-1)、新規ラベルでは `loss_mm` を使う
   - UT/実測なしの `loss_mm` 出力は **禁止**
   - UI 文言: UT なしでは「外観上の深さ傾向」「要確認度」、UT ありでのみ「推定減肉量 (mm)」
   - 学習・評価の主ラベルは `measured_depth_class` と `loss_mm`
   - `appearance_depth_class` は補助学習・弱教師・事前学習に限定
   - polygon は **`region_id` 経由で `labels_v0.json` を参照**、`polygon_audit_copy` はズレ検知用 (主キーにしない、P1-2)

#### 採用ゲート (Phase 7.0)

- ラベルスキーマ + scoring rule のレビュー完了
- 既存 CCS 440 枚から 200+ 枚に `appearance_depth_class` を付与 (経験則)
- UT 実測ペアを 30+ ペア収集開始 (Phase 7.3 の前提作り)

---

### 3.1 Phase 7.1a — 非深層ベースライン (古典 ML、CODEX P1-5 反映)

#### 狙い

CNN を作る前に、**解釈可能な古典 ML ベースライン**で:
- 何が効くかを把握 (特徴量重要度)
- 過学習を発見しやすくする (モデルが小さい)
- ラベル品質の問題を切り分ける

#### 入力特徴量 (画像 + polygon → ベクトル)

| カテゴリ | 特徴量 |
|---------|-------|
| 色統計 | polygon 内の RGB / HSV 平均・分散・歪度 |
| 輝度勾配 | Sobel / Scharr の magnitude 統計 |
| 影・ハイライト | 局所 min/max、Lab L チャンネル分布 |
| テクスチャ | LBP (Local Binary Pattern) ヒストグラム、GLCM (contrast/homogeneity/energy) |
| SAM2 メタ | `sam2_predicted_iou`, `sam2_area_norm`, `sam2_stability_score` |
| 形状 | 周長 / 面積比、円形度、bbox アスペクト比 |
| **周辺健全面差分** | polygon 外側のリング (5-10 px) との色・輝度差 |

#### モデル候補

| モデル | 評価 |
|------|------|
| **Logistic Regression (multinomial)** | 線形ベースライン、解釈しやすい |
| **Random Forest** | 非線形性 + 重要度可視化 |
| **LightGBM** | 通常最高性能、過学習に強い |

3 つを並行訓練し、measured subset でクロスバリデーション → 最良を採用。

#### 評価ゲート (v3 改訂、CODEX P0-3 反映)

| ステージ | 母数 | ゲート条件 | 用途 |
|---------|------|----------|------|
| **探索ゲート** | measured 50+ | macro F1 point ≥ 0.70 | 進路確認のみ、本採用不可 |
| **本採用ゲート** | **measured 100+** | macro F1 ≥ 0.70 **AND** bootstrap CI lower ≥ 0.55 **AND** 各クラス recall ≥ 0.50 | UI 反映 / 運用採用 |
| 補助メトリクス | appearance 200+ | F1 (参考値) | 上振れ可能性ありなので参考のみ |

**注意**: 50 件で F1=0.70 を達成しても、3 クラス × ~17 件/クラスでは bootstrap CI が広い (典型 [0.50, 0.85])。
本採用は必ず 100+ 件で再評価し、CI 下限と各クラス recall を確認してから。

split: **構造物 → 撮影系列 → 腐食箇所単位** (画像単位はリークするため不採用)

#### 期待成果物

- `api/depth/classical_baseline.py`: 特徴量抽出 + LR/RF/LightGBM
- `tests/eval/depth_metrics.py`: macro F1, クラス別 recall, unknown 率
- `docs/baseline.md` に「Phase 7.1a」エントリ追記

---

### 3.1b Phase 7.1b — 軽量 CNN (条件付き、7.1a を超えた場合のみ)

#### 採用条件

- Phase 7.1a の measured subset F1 を **明確に上回る** (Δ95% CI 下限 > 0)

#### 入力設計 (CODEX P1-5 反映、過学習対策)

腐食 crop 単体ではなく、3 チャンネル合成:
1. **腐食 crop** (RGB)
2. **周辺リング** (polygon 外側 10-20 px、健全面参照用)
3. **マスク channel** (polygon 内 = 1、外 = 0)

#### モデル

- ResNet18 or MobileNetV3 small base
- ImageNet 事前学習済 → fine-tune
- 出力 head: 3 クラス (shallow / moderate / deep) + softmax

#### 過学習対策 (CODEX P0-2 反映)

- **構造物単位 split** (画像単位はリーク)
- **照明スライス** で性能ばらつき測定 (normal/low/backlight)
- **撮影距離スライス** (近接/中距離) で domain shift 確認
- データ拡張: 輝度補正、軽度の回転・反転 (上下反転は意味変わるので慎重)

---

### 3.2 Phase 7.2 — Visual Depth Hint (CODEX P0-3 反映、命名と評価を再定義)

#### v1 からの変更

- 名称: 「腐食深さマップ」 → **「visual_depth_hint」**
- 主指標: 「目視 5 段階順位」 → **「UT 最深点との top-K 一致率」**

#### モデル (CODEX P1-8 反映、ライセンス精密化)

| モデル | ライセンス | 採用 |
|------|----------|------|
| **Depth Anything V2 small** | **Apache-2.0** | ✅ 採用 (公開デモ可) |
| Depth Anything V2 Base | **CC-BY-NC-4.0** | ❌ 商用 / 公開デモ不可 |
| Depth Anything V2 Large/Giant | **CC-BY-NC-4.0** | ❌ 同上 |
| MiDaS v3.1 (DPT-Hybrid) | MIT | 🟡 比較対照のみ |

**v2 採用**: Depth Anything V2 **small** 固定、HF Space (公開デモ) で安全に使える。
参照: [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2)、[HF Base モデルカード](https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf) (CC-BY-NC-4.0 表示)。

#### 処理フロー

```
[画像]
  ↓ Depth Anything V2 small (画像全体)
[scene_depth_map (相対、シーン奥行き)]
  ↓ polygon でマスク
[polygon 内 depth 値]
  ↓ 周辺健全面 (リング) を基準面として平坦化 (差分マップ化)
[corrosion_relative_hint (健全面からの「凹みっぽさ」相対値)]
  ↓ 統計
[hint_min, hint_max, hint_mean, hint_p95, deepest_point (px)]
```

**「平坦化」が重要**: シーン奥行きそのままでなく、**周辺健全面 (リング) を基準にして相対化**することで「カメラ距離の影響」を抑え、腐食領域内の凹凸傾向を強調する。

#### 出力

```json
{
  "visual_depth_hint": {
    "min":  0.12,
    "max":  0.87,
    "mean": 0.43,
    "p95":  0.78,
    "deepest_point_px": [245, 312],
    "model": "depth_anything_v2_small",
    "_disclaimer": "シーン奥行きベースの相対値、絶対 mm ではない"
  }
}
```

#### 評価指標 (v3 改訂、CODEX P0-2 反映で半径ベースに緩和)

**top1 pixel 一致は厳しすぎる** (UT プローブ径 6-13mm + ピン打ち誤差数 px のため、ほぼ常に外れる)。
半径 r の領域に入ればヒットとする `top_region_hit_rate` を採用:

```
hit_radius_px = max(probe_diameter_px, 15)
  where probe_diameter_px = probe_diameter_mm / mm_per_px (ArUco 検出時) or 15 (ArUco 不使用時)
```

| 指標 | 計算法 | ゲート |
|------|-------|------|
| `top_region_hit_rate` | UT 最深点が hint top1 ピクセルから半径 `hit_radius_px` 内に入る率 | ≥ 0.50 |
| `top3_region_hit_rate` | UT 最深点が hint top3 ピクセル群のいずれかから半径 `hit_radius_px` 内に入る率 | ≥ 0.80 |
| `rank_corr_on_ut_points` | UT 複数点 (5+) と hint 値の Spearman 相関 | ρ ≥ 0.5 |
| `failure_rate` | rank_corr が中央値 - 2σ を下回る画像の割合 | ≤ 0.20 |
| `latency_p95` | 推論時間 95 パーセンタイル | **要実測 (Phase 7.2 W7)** |

UT データなしで「目視 5 段階」をラベルしても、それは「見た目の深そう判定」であり Depth Anything の出力と相関するのは当然。**UT 実測との一致だけが意味を持つ**ので、目視 5 段階評価は v2 で却下。

#### ゲート未達時

- UI 表示しない (誤導防止)
- Phase 7.4 で fine-tune 検討

---

### 3.3 Phase 7.3 — UT 校正による絶対深さ (CODEX P0-4 反映、ArUco を補助に)

#### v1 からの変更

- ArUco **単独**で mm 換算する案を **削除**
- **UT 校正法を主軸**に据え、ArUco は面積・姿勢補正の補助に限定
- 出力名を `absolute_mm` → **`estimated_loss_mm`** に変更 (推定値であることを明示)
- `method` / `n_calibration_points` / `ci` を必須フィールド化

#### 必要データ

50+ ペア (画像 + UT 実測):
- 各画像に UT 5+ 点で測定 (画像座標と実測 mm の対応付け)
- 検査員が画像にピン (画像座標) を打って UT 実測値を入力
- 同じ腐食箇所の連続複数枚も価値あり (時系列、同一構造)

#### 校正モデル

入力特徴量 (Phase 7.1a と同じセット) + visual_depth_hint 統計 → 出力: 推定 mm。

| モデル | 評価 |
|------|------|
| **線形回帰** | 解釈最重視、CI を出しやすい |
| **LightGBM 回帰** | 非線形対応、SHAP で重要度可視化 |
| ガウス過程回帰 | CI 自然に出る、データ少なめ向き |

採用基準: leave-one-corrosion-site-out CV で MAE 最小 + CI カバレッジ ≥ 0.80。

#### ArUco の役割 (補助のみ)

- 画像内 ArUco マーカー検出 → mm/px と平面姿勢を確定
- 腐食領域の **横方向サイズ (mm²)** を計算 (これは特徴量の 1 つに使える)
- 撮影距離 / 撮影角度の正規化 (UT データの照合精度向上)
- **Z 深さ換算には ArUco 単独では使わない**

#### 出力 (v3 改訂、loss_mm 名称に統一、P0-1 反映)

```json
{
  "estimated_loss_mm": {
    "value":                2.3,                      // 推定減肉量
    "ci95":                 [1.8, 2.9],
    "method":               "ut_calibrated_lightgbm_v1",
    "n_calibration_points": 47,
    "reference_thickness_mm": 6.0,                    // 元肉厚の前提値 (設計値 / 健全部 UT)
    "reference_source":     "design_spec" | "ut_healthy_area" | "specimen",
    "aruco_scale_used":     true,
    "probe_diameter_mm":    9.5,
    "warning":              null
  }
}
```

`warning` には「UT データ不足」「ArUco 検出失敗」「reference 不明」等を入れる。

#### 評価指標 (v3 改訂、CODEX 提案を反映)

| 指標 | 計算法 | ゲート |
|------|-------|------|
| `MAE` | 平均絶対誤差 (mm) | 浅腐食 ≤ 1.0、深腐食 ≤ 3.0 |
| `RMSE` | 二乗平均平方根誤差 | 補助 |
| `bias` | 系統的誤差 (推定 - 実測の平均) | \|bias\| ≤ 0.5 mm |
| `coverage` | CI95 が UT 実測値を含む割合 | ≥ 0.80 |
| `MAE_by_loss_bucket` | 浅 (<2mm) / 中 (2-5mm) / 深 (>5mm) で別計算 | 各バケットで上記 |

ペア (画像 + UT) は **構造物単位で train/test split** (画像単位リーク防止)。

---

### 3.4 Phase 7.4 — measured_depth_class の特化学習 (将来課題)

#### 採用条件

- UT 実測ペアが 200+ 蓄積
- Phase 7.1a/b の measured subset F1 が伸びない (頭打ち)

#### 候補

- Depth Anything V2 small に LoRA fine-tune (画像 → depth 系統)
- 軽量 CNN を UT ラベルで学習 (画像 → measured class)

詳細は v3 で書き起こし。

---

## 4. アーキテクチャ統合 (CODEX P1-6 反映、独立モジュール化)

### 4.1 v1 からの変更

- v1: `hybrid_detector.py` の Stage 3 に深さ推定を埋め込む
- **v2: 独立 `DepthEstimator` モジュール、`api/index.py` の v5/v8 パイプライン後段で注釈**

### 4.2 ファイル構成

```
api/depth/
├── __init__.py
├── depth_estimator.py        # オーケストレータ (DepthEstimator クラス)
├── classical_baseline.py     # Phase 7.1a 古典 ML
├── cnn_classifier.py         # Phase 7.1b CNN (条件付き採用)
├── visual_hint.py            # Phase 7.2 Depth Anything V2 small ラッパー
├── ut_calibration.py         # Phase 7.3 UT 校正回帰
├── aruco_helper.py           # Phase 7.3 ArUco 補助 (mm/px、姿勢)
└── feature_extractor.py      # 共通: polygon → 特徴量ベクトル

api/index.py
  └─ _run_v5_detector_pipeline 内で
     detections = [d.to_dict() for d in result.detections]   # 既存の dict 化 (line ~1991)
     if DEPTH_ESTIMATION_ENABLED:
         detections = DepthEstimator().annotate(image, detections)
         # 検出結果を **削除・並べ替え・review_flag に影響させない**
```

### 4.3 責務分離 + 型シグネチャ (v3、CODEX P1-1 反映)

```python
class DepthEstimator:
    def annotate(
        self,
        image:      "PIL.Image",
        detections: list[dict],   # to_dict() 後の検出 list
    ) -> list[dict]:
        """
        各 detection に optional な 'depth' キーを追加して返す。

        以下を絶対に変えない:
          - 検出順 (detections の並び)
          - 検出数 (len(detections))
          - confidence / polygon / bbox / visual_label / review フィールド
        """
```

責務:
- `Detection` クラスや `api/detectors/base.py` には触らない
- `api/index.py` の dict 化後に追記のみ
- `DEPTH_ESTIMATION_ENABLED=false` で **完全に呼ばれない** (snapshot test で検証、§6.3)

### 4.4 環境変数

```
DEPTH_ESTIMATION_ENABLED=false       # default: off (オプトイン設計)
DEPTH_CLASSIFIER_MODEL=lightgbm      # lightgbm / cnn / off
DEPTH_VISUAL_HINT_MODEL=depth_anything_v2_small
DEPTH_UT_CALIBRATION_PATH=models/ut_calibration_v1.pkl
DEPTH_ARUCO_ENABLED=false
```

### 4.5 API レスポンス拡張 (互換性維持)

```json
{
  "detections": [
    {
      "polygon": [[y, x], ...],
      "confidence": 0.92,
      "subtype": "pitting",
      "sam2_predicted_iou": 0.95,
      "color_score": 0.78,
      "depth": {                          // optional、enabled 時のみ
        "appearance_class":      "moderate_looking",
        "appearance_confidence": 0.81,
        "visual_hint":           {"mean": 0.43, "p95": 0.78, "deepest_point_px": [245, 312]},
        "estimated_loss_mm":     null,    // UT 校正データなし → null
        "_disclaimer":           "外観上の深さ傾向のみ、減肉量推定は UT/実測必須"
      }
    }
  ]
}
```

`DEPTH_ESTIMATION_ENABLED=false` なら `depth` フィールド自体が含まれない (既存検出 API と完全互換)。

---

## 5. データ要件 (CODEX P0-2 反映、split リーク防止)

### 5.1 ラベル必要数

| Phase | ラベル種類 | 必要数 | 取得方法 |
|-------|----------|--------|---------|
| 7.0 | スキーマ + scoring rule 凍結 | — | docs |
| 7.1a/b | `appearance_depth_class` | 200+ | 検査員 / オーナー経験則 |
| 7.1a/b | `measured_depth_class` (主ゲート用) | **50+** | UT 実測 |
| 7.2 | UT 最深点 (画像座標 + mm) | 50+ ペア | UT + 検査員ピン |
| 7.3 | UT 5+ 点 (画像座標 + mm) | 50+ ペア | UT |
| 7.4 | UT 5+ 点 | 200+ ペア | (将来) |

### 5.2 split 規則 (リーク防止、CODEX P0-2 反映)

**画像単位 split は禁止**。以下の階層で分離:

```
1. 構造物 (structure_id): site_A_pipe_3 等
   ↓
2. 撮影系列 (capture_session_id): session_2026_03_15_morning 等
   ↓
3. 腐食箇所 (corrosion_site_id): site_A_pipe_3_loc_north_2 等
   ↓
4. 個別画像
```

train/val/test を **構造物 (or 腐食箇所) 単位**で分離。同じ corrosion_site_id の複数画像が train/val 両方に入らないようにする。

### 5.3 データソース

| ソース | 想定枚数 | 備考 |
|------|---------|------|
| 既存 CCS 再ラベル | 200+ (appearance のみ) | 構造物 ID は再構築必要 |
| 検査現場 (UT 実測併用) | 50+ ペア | 検査員協力必須 |
| ラボ基準試験片 | 20-30 ペア | 深さ既知の参照 |

---

## 6. 評価ハーネス (CODEX P1-7 反映、検出評価と分離)

### 6.1 新規ファイル

```
tests/eval/depth_metrics.py        # 深さ専用指標
tests/eval/run_depth_eval.py       # 深さ専用 CLI
tests/eval/scoring_rule_depth.md   # 凍結文書 (Phase 7.0 で作成)
```

### 6.2 評価の二段構成 (CODEX P1-7)

**段階 1: GT polygon ベース評価** (検出ミスを切り離す)
- 入力: 既知 polygon + 画像 + (任意) UT データ
- 評価: 純粋な深さ推定精度

**段階 2: end-to-end 評価** (実運用相当)
- 入力: 画像のみ (SAM2 が polygon を出す)
- 評価: 「検出できた領域だけの深さ精度」を別指標として

混ぜると検出ミスと深さミスが見分けられないので分離。

### 6.3 互換性テスト + UI 文言テスト (v3 改訂、P1-1 / P2-1 反映)

#### 6.3.1 API スナップショットテスト (P1-1)

`DEPTH_ESTIMATION_ENABLED=false` で:
- 既存 `/api/analyze` レスポンスに `depth` フィールドが含まれない (フィールド存在チェック)
- 既存検出メトリクス (critical_recall, per_gt_recall 等) が変わらない
- レスポンス JSON が enabled=false 時と完全一致 (snapshot diff)

```python
# tests/eval/test_depth_compatibility.py
def test_depth_disabled_preserves_response():
    os.environ['DEPTH_ESTIMATION_ENABLED'] = 'false'
    resp_before = call_analyze(sample_image)
    # depth field がない確認
    for det in resp_before['detections']:
        assert 'depth' not in det
    # 既存 metrics 変化なし
    assert detection_metrics(resp_before) == reference_metrics
```

#### 6.3.2 UI 文言テスト (P2-1、E2E で検証可能化)

「UT なし時は mm 表示しない」を **テストで検出可能なルール**として実装:

```javascript
// tests/ui/test_no_mm_without_ut.spec.js (Playwright)
test('estimated_loss_mm が null のとき UI に「mm」を表示しない', async ({ page }) => {
  await page.route('**/api/analyze', route => route.fulfill({
    json: { detections: [{ /* ... */, depth: { estimated_loss_mm: null, /* ... */ } }] }
  }));
  await uploadImage(page, 'test.jpg');
  const html = await page.locator('#detection-list').innerHTML();
  expect(html).not.toContain('mm');
  expect(html).not.toContain('減肉量');
  expect(html).not.toContain('深さmm');
  // 代わりに「外観上の深さ傾向」表示があること
  expect(html).toMatch(/外観上の深さ傾向|要確認度/);
});
```

このテストが落ちる = 文言ルール違反、CI で防止可能。

---

## 7. リスクと回避策 (v2 改訂版)

### 7.1 単眼の物理的限界 (再掲、v1 から強化)

| リスク | 確率 | 影響 | 回避策 |
|-------|------|------|------|
| `measured_depth_class` の F1 が伸びない (経験則ラベル混入) | 高 | 大 | UT サブセットで主評価、経験則は弱教師に格下げ |
| Depth Anything がドメインミスマッチで failure_rate 高 | 高 | 中 | 周辺健全面を基準とする平坦化処理を入れる、ゲート未達なら UI 非表示 |
| ArUco 単独で mm 換算する誘惑 | 中 | 大 | v2 で **削除済**、UT 校正必須を凍結 |
| 公開デモで CC-BY-NC モデルを誤って使用 | 低 | 大 (法務) | Depth Anything Small (Apache-2.0) 固定、CI でバージョン検証 |

### 7.2 データ取得困難

| リスク | 確率 | 影響 | 回避策 |
|-------|------|------|------|
| UT 実測ペア 50+ が集まらない | 高 | 大 | Phase 7.1a/b の主ゲートを「measured 50 件」に下げて段階的に拡大、Phase 7.3/7.4 を後ろにずらす |
| 構造物単位 split で sample 不足 | 高 | 中 | 異なる構造物の最低 5-10 種を確保、足りなければラボ基準試験片で補完 |
| ラベル品質ばらつき | 中 | 中 | 2 名合議の Cohen's κ ≥ 0.6 を維持、不一致サンプルは unknown 扱い |

### 7.3 アーキテクチャ統合

| リスク | 確率 | 影響 | 回避策 |
|-------|------|------|------|
| 推論時間が伸びる (要実測、CODEX P1-3) | 中 | 中 | Depth Anything V2 small + SAM2 同時の latency_p95 を Phase 7.2 W7 で計測、目標 +5s 以内 |
| HF Space CPU で OOM (torch + SAM2 + Depth Anything 同時) | 中 | 大 | 16GB free tier でメモリ実測、超過時は遅延ロード or model unload |
| `depth` フィールドが UI を破壊 | 低 | 中 | `DEPTH_ESTIMATION_ENABLED=false` で完全 OFF、snapshot 回帰テスト |
| UI で「mm」と誤表示 | 低 | 大 | UT なし時は **mm 表示自体を出さない**、E2E テストで自動検証 (§6.3.2) |
| Dockerfile で不要な ckpt DL | 低 | 小 | `DEPTH_ESTIMATION_ENABLED=false` のビルドでは Depth Anything ckpt をスキップ (条件付き RUN) |

---

## 8. マイルストーン (v2、Phase 7.0 を冒頭に追加)

### Month 1 — Phase 7.0 + 7.1a (ラベル定義 + 古典 ML ベースライン)

| 週 | アクション |
|----|---------|
| W1 | `docs/depth_scoring_rule.md` ドラフト、`labels_depth_v0.schema.json` 作成、文献調査 |
| W2 | CCS train 余り 200+ 枚に `appearance_depth_class` 経験則ラベル付与、構造物 ID 整備 |
| W3 | UT 実測ペア 30+ 件の収集開始 (検査現場連携) |
| W4 | Phase 7.1a 古典 ML 実装、measured subset で評価、`baseline.md` 追記 |

### Month 2 — Phase 7.1b + 7.2

| 週 | アクション |
|----|---------|
| W5 | UT 実測ペア 50+ 達成、Phase 7.1a の measured F1 確定 |
| W6 | (条件付き) Phase 7.1b CNN 実装、構造物単位 split で 7.1a と比較 |
| W7 | Phase 7.2 visual_hint 実装 (Depth Anything V2 small + 平坦化) |
| W8 | UT 最深点 top-K 評価、failure_rate 計測 |

### Month 3 — Phase 7.3 (UT 校正)

| 週 | アクション |
|----|---------|
| W9 | UT データ + 画像のペアリングツール作成 (検査員が画像にピン打ち) |
| W10 | LightGBM/線形回帰の校正モデル学習 (leave-one-site-out CV) |
| W11 | ArUco 補助実装 (mm/px、姿勢)、特徴量に追加 |
| W12 | `estimated_loss_mm` 出力 + UI 表示 (mm 表記は UT 校正時のみ) |

### Month 4+ — Phase 7.4 (条件付き、fine-tune)

UT 実測ペア 200+ + Phase 7.1-7.3 が頭打ちの場合のみ。詳細は v3 で。

---

## 9. v8 Final との関係 (再整理、CODEX P1-6 反映)

### 9.1 影響なし (依存)

- SAM2 検出パイプライン、評価ハーネス、scoring_rule v1 → そのまま使用
- `Detection` クラスは meta 経由で読むのみ

### 9.2 影響あり (拡張、独立モジュール化)

| ファイル | 変更内容 |
|---------|---------|
| `api/depth/*` | 新規 (独立モジュール群) |
| `api/index.py` | `_run_v5_detector_pipeline` の **後段** で `DepthEstimator.annotate()` を呼ぶ (`DEPTH_ESTIMATION_ENABLED=true` 時のみ) |
| `api/detectors/*` | **変更なし** (検出パイプラインに深さは入らない) |
| `public/index.html` | 検出カードに深さ情報を optional 表示、UT なし時は「外観上の傾向」と明記 |
| `tests/eval/datasets/labels_depth_v0.json` | 新規 |
| `tests/eval/depth_metrics.py` / `run_depth_eval.py` | 新規 |
| `Dockerfile.hfspace` | Depth Anything V2 small ckpt の事前 DL 追加 (`DEPTH_ESTIMATION_ENABLED=false` でも DL するか選択可) |

### 9.3 オプトイン (CODEX P1-6)

`DEPTH_ESTIMATION_ENABLED=false` がデフォルト。
既存運用を一切壊さない。HF Space でも当面 OFF で稼働、ローカル PC 検証で十分品質を確認してから有効化。

---

## 10. コスト見積 (v2 修正版)

| 項目 | コスト | 備考 |
|------|--------|------|
| Depth Anything V2 **small** ckpt (~50MB) | $0 | Apache-2.0 |
| HF Space CPU 推論 | $0 | Free tier、+1-2 秒/画像 |
| ArUco マーカー (5cm) | ~¥500 | 一回限り |
| UT 実測ペアの取得 (検査員協力) | 工数依存 | 50+ 件で 10-20 時間 |
| 学習 GPU (Phase 7.4 のみ) | $4-8 | A100 1-2 時間、条件発生時 |
| ラベリング工数 | 8-15 時間 | 200+ 枚 × 経験則 |
| **合計** | **~¥500 + 工数** | (LLM コスト 0、v8 Final 路線継続) |

---

## 11. 文献調査タスク (Phase 7.0 着手前、CODEX P0-3 反映)

`docs/depth_estimation_literature.md` にまとめる:

- [ ] Depth Anything V2 (Yang et al. 2024) — 屋外シーン以外、近接被写体での挙動
- [ ] MiDaS v3.1 (Birkl et al. 2023) — 比較対照用
- [ ] **腐食孔深さ推定の SOTA** — ASCE / NDT&E International / Corrosion Science 検索
- [ ] **配管 UT 検査標準** — JIS Z 2300 / ASME B31.3 の深さ要求精度
- [ ] **photometric stereo / shape-from-shading** の腐食適用例
- [ ] ArUco 平面仮定の精度限界 (曲面配管での誤差)

---

## 12. 完了の定義 (v2)

### Phase 7.0
- [ ] `docs/depth_scoring_rule.md` 凍結
- [ ] `labels_depth_v0.schema.json` 凍結
- [ ] 200+ 枚に appearance ラベル付与、UT ペア 30+ 件収集

### Phase 7.1a/b
- [ ] **measured subset で macro F1 ≥ 0.70** (主ゲート)
- [ ] 構造物単位 split でリークなし確認
- [ ] 7.1b は 7.1a を超えた場合のみ採用

### Phase 7.2
- [ ] `deepest_point_top1_hit_rate ≥ 0.50`、`top3 ≥ 0.80`
- [ ] failure_rate ≤ 0.20
- [ ] 平坦化処理 (健全面リング基準) の効果確認

### Phase 7.3
- [ ] UT 50+ ペアで MAE ≤ 1mm (浅) / ≤ 3mm (深)
- [ ] CI95 カバレッジ ≥ 0.80
- [ ] UI で `estimated_loss_mm` 表示は UT 校正データありの時のみ

### 全体
- [ ] `DEPTH_ESTIMATION_ENABLED=false` で既存 API/UI が無変化
- [ ] release_test 1 回で最終評価
- [ ] CODEX 第 2 次レビュー全項目クリア
- [ ] Depth Anything Small (Apache-2.0) のみ使用、CC-BY-NC モデル混入ゼロ

---

## 13. v8 Final との 1 行比較 (v2 改訂)

| 観点 | v8 Final (検出) | **v2 (深さ推定)** |
|------|----------------|------------------|
| 主要モデル | SAM2 | Depth Anything V2 **small** + 古典 ML/軽量 CNN |
| LLM 利用 | なし | **なし** (路線継続) |
| 主指標 | critical_recall (画像) | macro F1 (`measured` subset) / top-K hit rate / MAE |
| 物理的限界 | per_gt_recall 集合体問題 | **絶対深さは UT 実測必須** (本質、誤魔化さない) |
| 統合方式 | 検出パイプライン本体 | **独立モジュール、後段注釈、検出に影響なし** |
| 採用ゲート | val Δ95% CI 下限 > 0 + 運用ゲート | (同じ枠組み継承、深さ専用 scoring rule v2) |

---

## 14. v3 の 1 行サマリー

> **「UT は孔深さでなく残存肉厚を測る、top1 pixel hit は厳しすぎ、50 件は探索ゲート。だから `loss_mm` を主値とし、半径ベース hit_rate で評価し、100+ で本採用する。`DepthEstimator` は dict を入出力する後段注釈で、検出に絶対影響しない。」**

---

## 付録: ライセンス記述 (CODEX 第 2 次レビュー P1-3 確認済)

| モデル | ライセンス | 商用 / 公開デモ |
|------|----------|---------------|
| **Depth Anything V2 Small** | **Apache-2.0** | ✅ 採用 |
| Depth Anything V2 Base | CC-BY-NC-4.0 | ❌ 禁止 |
| Depth Anything V2 Large | CC-BY-NC-4.0 | ❌ 禁止 |
| Depth Anything V2 Giant | CC-BY-NC-4.0 | ❌ 禁止 |

ソース:
- [Depth Anything V2 GitHub](https://github.com/DepthAnything/Depth-Anything-V2): Small は Apache-2.0、Base/Large/Giant は CC-BY-NC-4.0
- [Depth Anything V2 Small HF](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf): apache-2.0
- [depth-anything-v2 PyPI](https://pypi.org/project/depth-anything-v2/): 同ライセンス区分

CI で `Depth-Anything-V2-Small` 以外の checkpoint が読み込まれた場合エラーにするバリデーションを追加 (Phase 7.2 W6 で実装)。
