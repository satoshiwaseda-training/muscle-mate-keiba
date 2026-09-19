# Depth Scoring Rule — 深さ評価指標の計算規則 (凍結文書)

**バージョン**: v1 (2026-04-29)
**ステータス**: ✅ 採択 (オーナー単独承認モード)
**変更ルール**: この文書の変更は **過去の全評価数値に注釈が必要** になる。
**親計画書**: `docs/depth_estimation_plan_v1.md` (v3) と整合

---

## 1. 目的

深さ推定 (Phase 7.X) で使う指標の **計算規則を 1 箇所に固定** し、実装担当によるブレを防ぐ。
検出評価 (`tests/eval/scoring_rule.md` v1) とは独立して凍結する。

---

## 2. 用語定義

| 用語 | 定義 |
|------|------|
| `appearance_depth_class` | 見た目の深さ傾向 (UT なし可)、shallow_looking / moderate_looking / deep_looking / unknown |
| `measured_depth_class` | UT/プロファイル/試験片由来の実測クラス、shallow / moderate / deep / unknown |
| `loss_mm` | 減肉量 = `reference_thickness_mm − remaining_thickness_mm` |
| `reference_thickness_mm` | 元肉厚 (設計値 or 健全部 UT 実測) |
| `remaining_thickness_mm` | UT で測定した残存肉厚 |
| `measured subset` | `measured_depth_class` または `loss_mm` が UT 由来で付いたサンプル群 |
| `appearance subset` | `appearance_depth_class` のみが付いたサンプル群 (経験則ラベル) |

`depth_mm` フィールドは **廃止** (v3 で `loss_mm` に統合)。

---

## 3. 評価指標

### 3.1 Phase 7.1 (クラス分類) — `appearance_depth_class` 推定

| 指標 | 定義 |
|------|-----|
| `macro_f1` | 3 クラス (shallow / moderate / deep) の per-class F1 の単純平均、unknown は除外 |
| `macro_f1_ci_lower` | bootstrap (B=1000) による macro_f1 95% CI 下限 |
| `class_recall_min` | 3 クラス中の最小 recall |
| `unknown_rate` | 予測が unknown だった画像の割合 |
| `confusion_matrix` | 3x3 (unknown は別途) |

### 3.2 Phase 7.2 (visual_depth_hint) — Depth Anything 出力評価

UT 最深点と hint の一致を半径ベースで判定:

```
hit_radius_px = max(probe_diameter_px, 15)
  where probe_diameter_px = probe_diameter_mm / mm_per_px (ArUco 検出時)
                         or 15 (ArUco 不使用時)
```

| 指標 | 定義 |
|------|-----|
| `top_region_hit_rate` | UT 最深点が hint top1 ピクセルから半径 `hit_radius_px` 内に入る画像の割合 |
| `top3_region_hit_rate` | UT 最深点が hint top3 ピクセル群のいずれかから半径 `hit_radius_px` 内に入る割合 |
| `rank_corr_on_ut_points` | UT 複数点 (5+) と hint 値の Spearman 相関 (画像ごと → 平均) |
| `failure_rate` | rank_corr が中央値 - 2σ を下回る画像の割合 |
| `latency_p95` | 推論時間 95 パーセンタイル (秒、画像 1 枚あたり) |

### 3.3 Phase 7.3 (estimated_loss_mm) — UT 校正

| 指標 | 定義 |
|------|-----|
| `MAE` | 平均絶対誤差 (推定 mm − UT 実測 mm) |
| `RMSE` | 二乗平均平方根誤差 |
| `bias` | 系統的誤差 = mean(推定 − 実測) |
| `coverage` | CI95 が UT 実測値を含む割合 |
| `MAE_by_loss_bucket` | 浅 (<2mm) / 中 (2-5mm) / 深 (>5mm) でのバケット別 MAE |

---

## 4. 採用ゲート (recall 探索 vs 運用採用)

`docs/depth_estimation_plan_v1.md` v3 と整合:

### 4.1 Phase 7.1 (P0-3 反映、段階ゲート)

| ステージ | 母数 | ゲート | 用途 |
|---------|------|------|------|
| **探索ゲート** | measured 50+ | macro_f1 ≥ 0.70 | 進路確認、本採用不可 |
| **本採用ゲート** | **measured 100+** | macro_f1 ≥ 0.70 **AND** macro_f1_ci_lower ≥ 0.55 **AND** class_recall_min ≥ 0.50 | UI 反映、運用採用 |

### 4.2 Phase 7.2

| 指標 | ゲート |
|------|------|
| top_region_hit_rate | ≥ 0.50 |
| top3_region_hit_rate | ≥ 0.80 |
| rank_corr_on_ut_points | ρ ≥ 0.5 |
| failure_rate | ≤ 0.20 |
| latency_p95 | 要実測 (Phase 7.2 W7 で計測) |

### 4.3 Phase 7.3

| 指標 | ゲート |
|------|------|
| MAE (浅腐食 < 2mm) | ≤ 1.0 mm |
| MAE (深腐食 > 5mm) | ≤ 3.0 mm |
| \|bias\| | ≤ 0.5 mm |
| coverage | ≥ 0.80 |

---

## 5. Split 規則 (リーク防止、CODEX P0-2 反映)

### 5.1 階層

```
1. structure_id        (例: site_A_pipe_3)
   ↓
2. capture_session_id  (例: session_2026_03_15_morning)
   ↓
3. corrosion_site_id   (例: site_A_pipe_3_loc_north_2)
   ↓
4. image_id            (例: ccs_train_42)
   ↓
5. region_id           (例: ccs_train_42_r0)
```

### 5.2 split は最低 `corrosion_site_id` 単位

- 画像単位 split は **禁止** (同じ腐食箇所の複数角度撮影が train/val 両方に入りリーク)
- 推奨: `structure_id` 単位 (最も保守的)
- 最低: `corrosion_site_id` 単位

### 5.3 leave-one-X-out クロスバリデーション

UT ペアが少ない (50-100 件) ため:
- Phase 7.1: stratified k-fold CV (k=5)、構造物単位
- Phase 7.3: leave-one-corrosion-site-out CV

---

## 6. 信頼区間とサンプル数

すべての主指標は **点推定 + 95% bootstrap CI (B=1000)** を併記。

### 6.1 母数別の運用

| 母数 | CI 信頼性 | 運用 |
|------|----------|------|
| < 50 | 低 | 探索のみ、論文・運用採用不可 |
| 50-99 | 中 | 探索ゲート判定可、本採用不可 |
| **100-199** | **そこそこ** | **本採用ゲート判定可** |
| 200+ | 高 | スライス評価可 |

---

## 7. スライス評価

### 7.1 必須スライス

母数 ≥ 100 で以下を別途計算:

| 軸 | カテゴリ |
|----|--------|
| 照明 | normal / low_light / backlight |
| 撮影距離 | close (< 30cm) / mid (30-100cm) / far (> 100cm) |
| 表面状態 | clean / wet / scaled / painted |
| 腐食タイプ | pitting / uniform / flaking |

### 7.2 各セルで母数 ≥ 20

母数不足セルは `n/a` 表示、評価対象外。

---

## 8. UT データの扱い

### 8.1 必須フィールド (CODEX P0-1 反映)

UT 由来ラベルを使う場合、以下が **すべて必須**:

- `reference_thickness_mm` (元肉厚)
- `remaining_thickness_mm` (UT 実測)
- `loss_mm` (派生、整合性チェック対象)
- `measurement_device` (機器名)
- `probe_diameter_mm` (プローブ径、半径計算に使用)

### 8.2 UT 点の画像座標 (CODEX P0-2 反映)

各 UT 点に以下を付与:

- `px_y`, `px_x` (ピン打ち位置)
- `px_uncertainty_radius` (ピン打ち誤差半径、典型 5-15 px)
- `probe_diameter_mm` (UT プローブ径、半径計算用)
- `surface_condition` (clean / wet / scaled / painted)

### 8.3 整合性チェック

評価ハーネスは以下を起動時に検証:
- `loss_mm == reference_thickness_mm − remaining_thickness_mm` (誤差 < 0.05 mm)
- `probe_diameter_mm > 0`
- `0 ≤ remaining_thickness_mm ≤ reference_thickness_mm`

---

## 9. エッジケース

### 9.1 UT 点なし画像

`measured_depth_class` のみ付与、UT 点座標は空配列。
Phase 7.2 の `top_region_hit_rate` 評価対象外、Phase 7.1 の measured subset には含む。

### 9.2 ArUco 不検出 + UT データなし

`mm_per_px` が確定しないため `hit_radius_px = 15` (固定値) で評価。
`estimated_loss_mm` は出力しない。

### 9.3 polygon 内に UT 点が無い

UT が腐食箇所外を測ったケース。データクレンジングで除外。
評価ハーネスで warn ログを出す。

### 9.4 reference_thickness_mm 不明

`loss_mm` が計算不可。`measured_depth_class` のみ採用、`loss_mm` は null。
Phase 7.3 のテストセットからは除外。

---

## 10. release_test 凍結ルール

- `val_depth`: 開発中の比較に使う
- `release_test_depth`: Phase 7 完了時 1 回のみ評価
- `monitoring_set_depth`: 運用後の継続監視

検出 scoring_rule v1 と同じポリシーを継承。

---

## 11. 変更履歴と署名

| バージョン | 日付 | 変更内容 | 承認者 |
|-----------|------|--------|--------|
| v1 | 2026-04-29 | 初版採択、CODEX 第 2 次レビュー全 7 項目反映 | サトシ (satoshi.waseda@gmail.com) |
