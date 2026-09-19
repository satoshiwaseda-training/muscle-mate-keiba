# Corrosion Detector v8 — 実績まとめ & 精度向上 次期計画書

**作成日**: 2026-04-25 (v6 初版)
**改訂履歴**:
- v6 初版 — Phase 5.1〜5.5 ロードマップ
- v6.1 — CODEX レビュー P0/P1/P2/P3 反映 (Step 0 追加)
- v7 — Gemini 段階投入戦略への転換 (5.1/5.2 OFF、5.3 で MAX_CLASSIFY=3 投入)
- **v8 (2026-04-25 夜)** — **CODEX 第 2 次レビュー全 6 項目反映、Gemini を主経路から完全排除**:
  1. **検出経路から Gemini を完全切り離し** — `hybrid_detector.py` の P0 バグ 2 件 (line 217 削除条件、line 204 失敗時 continue) を修正
  2. **Gemini は注釈器に格下げ** — subtype/severity/reasoning のみ提供、検出有無・confidence・review_flag には関与しない
  3. **閾値校正の入力を SAM2 純粋スコアに変更** — predicted_iou + mask area + color_score (LLM 由来不安定性を排除)
  4. **採否ゲートに precision@K と manual_review_rate の下限を追加** — recall 探索モードと運用採用モードを分離
  5. **完了条件を統一** — release_test CI 下限 > 0.85、monitoring_set CI 下限 > 0.90

**作成者**: サトシ (オーナー単独承認モード)
**目的**:
1. 計画 v1 → v8 の実装実績を 1 ファイルに集約
2. 「critical_recall ≥ 0.95」を狙う次期アクションを優先順位付きで提示
3. **Gemini を検出経路から完全排除し、SAM2 とその下流の決定的特徴量に専念**

**v8 の核心 (アーキテクチャ責務)**:
```
SAM2 / fine-tuned SAM2     = 検出 (絶対に削除しない)
SAM2 predicted_iou + 色 +
面積 + テクスチャ特徴      = スコアリング (Phase 5.3 で軽量教師ありに発展可)
画像単位の非 LLM スコア    = 閾値校正
Gemini                     = 後段の説明・subtype 補助・レポート文面のみ
                             (検出結果は「絶対に消せない」)
```

> 「Gemini を **精度改善の構成要素として数えない**。検出 / 候補削除 / review_flag 決定 / 閾値校正 から完全排除。残すなら 説明生成・subtype 補助・レポート文面 のみ。」

**関連ドキュメント (深掘り用)**:
- `docs/accuracy_improvement_plan.md` — 計画書 v6 (1023 行、戦略全文)
- `docs/baseline.md` — 数値の公式記録 (append-only)
- `docs/deployment_guide.md` — デプロイ手順
- `tests/eval/scoring_rule.md` — 凍結された評価規則

---

## 1. エグゼクティブサマリー

| 観点 | v4 (Gemini 単独) | **v5/v6 (SAM2 + Gemini Hybrid)** | 改善 |
|------|----------------|-------------------------------|------|
| critical_recall (画像単位) | 0.400 | **0.800** [0.600, 0.950] | **+2.0倍** |
| per_gt_recall (領域単位) | 0.032 | **0.154** [0.108, 0.204] | **+4.8倍** |
| F1 @ IoU=0.3 | 0.039 | **0.116** | **+3.0倍** |
| 推論時間 (CPU) | 約 8 秒 | 約 60〜90 秒 | (許容範囲) |
| デプロイ状態 | ローカルのみ | **HF Space 公開 + ローカル並行運用** | ✅ |

**現時点の到達度**: 「画像に腐食があるかないか」の検出は実用レベル (recall=0.800)。
ただし**領域の細かさ** (per_gt_recall) は 0.154 で、「集合体をまとめて 1 つのマスクとして取る」傾向あり。
**次期目標**: release_test (n=44) で critical_recall ≥ 0.95、**CI 下限 > 0.85**、
monitoring_set 累積 100+ 枚で CI 下限 > 0.90 維持。

---

## 2. これまでの実績 (v1 → v6)

### 2.1 戦略・計画書 (6 リビジョン)

| Ver. | 日付 | 主な決定事項 |
|------|------|------------|
| v1 | 2026-04-21 | 初版。F1 主導、Self-Consistency × タイル × Few-shot 提案 |
| v2 | 2026-04-21 | 主指標を critical_recall に変更、test 汚染防止、カスケード化 |
| v3 | 2026-04-21 | GT / 運用フラグ / critical 根拠の分離、3 系統セット、CI ベース早期ゲート |
| v4 | 2026-04-21 | 単独承認モード、公開データセット移行、モック予測器導入 |
| **v5** | 2026-04-22 | **戦略転換**: Gemini を「検出」から「分類・レポート」に。検出は SAM2 へ委譲 |
| **v6** | 2026-04-22 | デプロイ即時リリース + 95% ロードマップ (Phase 5.1-5.5) |

### 2.2 データセット整備

- **Corrosion Condition State Dataset** (Bianchi & Hebdon 2021, CC0) を採用
- 440 枚の LabelMe JSON → 本プロジェクト GT スキーマに変換 (`scripts/convert_ccs_to_gt.py`)
- 4-way 分割 (train 212 / val 80 / release_test 44 / monitoring_set 104)
- `tests/eval/datasets/labels_v0.json` に 440 GT 記録、`split.json` に分割定義
- `gt_is_critical` 判定基準: `4_Severe_Steel_Corrosion` 領域 1 個以上 → critical 画像

### 2.3 評価ハーネス (再現性のある測定基盤)

- `tests/eval/run_eval.py` — CLI 評価ツール、mock predictor 対応
- `tests/eval/metrics.py` — critical_recall / per_gt_recall / precision@K + Bootstrap 95% CI
- `tests/eval/agreement.py` — Cohen's κ + ポリゴン IoU + boundary F1
- `tests/eval/mock_predictor.py` — oracle / empty / perturbed / noisy の 4 モード (理論上下限の固定)
- `scripts/visualize_predictions.py` — 予測可視化、`scripts/inspect_*.py` — キャッシュ検査

### 2.4 v5 アーキテクチャ実装 (Hybrid Pipeline)

```
[入力画像]
   ↓
[Stage 1: SAM2 Automatic Mask Generation]   ← Meta SAM2 (Apache-2.0、Small ckpt 185 MB)
   ↓
[Stage 2: 色 + 面積フィルタ]                 ← 赤茶 4 色 + BYPASS_COLOR=true で安全側
   ↓
[Stage 3: Gemini 2.5 Flash クロップ分類]     ← 構造化出力 + thinkingBudget=0
   ↓
[出力: ポリゴン + confidence + visual_label + severity]
```

**実装ファイル**:
- `api/detectors/__init__.py` — `get_detector(kind)` ファクトリ
- `api/detectors/base.py` — `BaseDetector` + `Detection` dataclass
- `api/detectors/sam2_detector.py` — SAM2 ラッパー (CPU / GPU 両対応)
- `api/detectors/hybrid_detector.py` — 3-stage パイプライン
- `api/gemini_classifier.py` — クロップ分類 (responseSchema, thinkingBudget=0, maxOutputTokens=2048)
- `api/index.py` — FastAPI、`_run_v5_detector_pipeline`、.env 自動ロード診断
- `api/preprocess.py` / `api/postprocess.py` / `api/tiling.py` / `api/schemas.py` / `api/thresholds.py`

### 2.5 デプロイ (2 系統並行運用)

| 環境 | URL / パス | 用途 | 状態 |
|------|----------|------|------|
| ローカル PC | `http://127.0.0.1:5000` | 開発・実験・実画像検証 | ✅ 稼働中 |
| Hugging Face Space | `https://musclemate-corrosion-detector.hf.space` | 公開デモ・遠隔検査支援 | ✅ 稼働中 |

**デプロイ周辺の整備**:
- `Dockerfile.hfspace` — HF Space 用 (CPU Free Tier、SAM2 ckpt 自動 DL)
- `requirements.hfspace.txt` — torch CPU 版を別レイヤーで先に install
- `scripts/deploy_to_hfspace.ps1` — PS 5.1 互換、ファイル同期 + commit + push
- `start_server.ps1` — ローカル起動ランチャー
- `README_hfspace.md` — HF Space front-matter (sdk: docker, app_port: 7860)

### 2.6 解決した重大な技術的問題

1. **Gemini が confidence=0.500 で帰ってくる** → スキーマ不整合 (`score` vs `confidence`) を修正
2. **CPU 90 秒タイムアウト** → SAM2 points_per_side 32→24、max_side 1024→768、フロントエンド timeout 240s
3. **零検出** → 4 色対応の color_score 拡張 + `HYBRID_BYPASS_COLOR=true` フォールバック
4. **HF Space で gemini_call_count=0 の沈黙失敗** → 診断 print 投入で `MAX_TOKENS` 不足を特定
5. **Gemini 2.5 Flash の thinking モードがトークン全消費** → `thinkingBudget=0` + `maxOutputTokens=2048` で解決 ⭐
6. **HF Space 404** → サブドメインは小文字配信 (`musclemate-corrosion-detector.hf.space`)
7. **PS 5.1 の here-string エスケープ不可** → 単独 .py スクリプト方式に統一
8. **メモリ OOM (440 LabelMe JSON)** → PIL 廃止、imageHeight/Width フィールド利用

### 2.7 凍結された運用ルール

- **scoring_rule.md**: 評価指標の計算規則を凍結 (今後の改善は規則を変えずに測る)
- **release_test 44 枚は本番直前まで触らない** (汚染防止)
- **改善 1 つにつき val 差分 95% CI 下限 > 0** を採用基準とする (アブレーション原理)
- 数値はすべて `docs/baseline.md` に append-only で記録

---

## 3. 現時点の限界と原因分析

### 3.1 数値で見る現状

| 指標 | 現状 | 95% 目標 | ギャップ |
|------|------|---------|---------|
| critical_recall | 0.800 | **≥ 0.95** | -0.15 |
| critical_recall CI 下限 (release_test n=44) | 0.600 | **> 0.85** | -0.25 |
| critical_recall CI 下限 (monitoring_set 100+) | (未測定) | **> 0.90** | — |
| per_gt_recall | 0.154 | ≥ 0.50 | -0.35 |
| precision@K=30 | 0.333 | ≥ 0.50 | -0.17 |

### 3.2 観察された 3 つの主要失敗パターン

**A. 集合体まとめ取り (per_gt_recall 0.154 の主因)**
SAM2 の Automatic mask generation が「点食痕の群れ」を 1 つの大マスクとして抽出。
GT は個別領域なので、N 個の GT のうち 1 つしかマッチしない (recall = 1/N)。

**B. 粒度不足**
points_per_side=24 (576 サンプリング点) は CPU 速度のために絞っているが、
細かい腐食 (< 50 px²) を取りこぼす。

**C. ドメインミスマッチ**
SAM2 は汎用セグメンタ、腐食特化ではないため「コンクリート割れ目」「影」「シール跡」を腐食候補として上げる FP が混入。
Gemini 分類で多くは弾けるが、信頼度の閾値が未校正。

### 3.3 Gemini 構造的問題 (v7 で確認、新規)

**D. Gemini JSON 出力品質問題**
Phase 5.1 評価実行中のサーバログで以下を観測:
```
"subtype": "uniform_corrosion_and_flaking_corrosion_and_pitting_corrosion_and_rust_stains_..."
"subtype": "uniform, pitting, flaking, crack, unknown, non_corrosion, pitting, ..."
```
- `responseSchema` の `subtype` を `{"type": "string"}` のみで定義していたため、
  Gemini が定義 enum を無視して候補を `_and_` 連結
- 超長文出力が `maxOutputTokens=2048` を圧迫し JSON が途中で切れる
- **約 25% のクロップ分類が `JSONDecodeError` で失われる**
- 失われた分類は対応する SAM2 候補ごと消失 → critical_recall / per_gt_recall を下振れ

**E. Gemini が SAM2 の有効候補を捨てる**
HYBRID_BYPASS_COLOR=true でも `gemini_classifier` が `is_corrosion=False` と判定したマスクは消失。
本物の腐食でも見え方によっては non_corrosion 判定され、SAM2 が正しく取った polygon が捨てられる。

### 3.4 結論 (v7 戦略の根拠)

**Gemini は現状、精度の天井を引き下げる要因**になっている。

| 比較 | 構成 | critical_recall | per_gt_recall |
|------|------|----------------|---------------|
| v5 baseline | SAM2 + Gemini ON | 0.800 [0.600, 0.950] | 0.154 |
| v5.1a (失敗実験) | SAM2 + Gemini OFF (MAX_CLASSIFY=8 のフィルタ罠) | 0.600 [0.400, 0.800] | 0.104 |
| v5.1b (期待) | SAM2 + Gemini OFF (MAX_CLASSIFY=40 で全活用) | **0.85〜0.95 (期待)** | **0.20〜0.35 (期待)** |

→ Phase 5.1/5.2 は **Gemini OFF + MAX_CLASSIFY を SAM2 出力に追従** で純粋な SAM2 性能を測る。
Gemini は Phase 5.3 で「分類器」として最小限 (MAX_CLASSIFY=3) 投入する。

---

## 4. 次期計画書 — 95% 到達ロードマップ

### 4.0 Step 0 — Phase 5 着手前の地盤固め (1〜2 日、必須)

**目的**: Phase 5.X 以降のすべての改善判定が「正しい指標」と「整合した実行環境」で行われることを保証する。

| ID | 内容 | 状態 (2026-04-25) | 影響 |
|----|------|------------|------|
| **S0-1** | `metrics.py` の TP 判定を scoring_rule §4.1 (AND 構造) 準拠に修正 (IoU 一致なし bypass を削除) | ✅ 実装済 (`tests/eval/metrics.py:106-131`) | critical_recall が下振れする可能性あり、再評価必須 |
| **S0-2** | `val` 80 枚を再評価し `docs/baseline.md` に「評価規則修正後 baseline」として追記 | ✅ **完了 (cache 再評価)** — 結果は **0.800 (変化なし)**、bypass は実は trigger していなかった (`baseline.md` 参照) | 全 Phase 5.X の比較基準点を確定 → **0.800 で確定** |
| **S0-3** | `.env.example` のプレースホルダ化 + Google AI Studio で旧キー revoke + 新規発行 | ✅ ファイル修正済 / ⏳ ローテーションはオーナー作業 | 公開リスク除去 |
| **S0-4** | `run_eval.py` の API timeout を 60s → 240s | ✅ 実装済 (`tests/eval/run_eval.py:255-257`) | Phase 5.1 の 100-120s 想定に対応 |
| **S0-5** | `run_eval.py` の出力 JSON に実効環境変数 (`SAM2_*`, `HYBRID_*`, `GEMINI_*`) を embed | ✅ 実装済 (`_capture_effective_env`) | variant 名だけでない実効パラメータの追跡可能化 |
| **S0-6** | 本ドキュメントの release_test ゲートを計画書本体 (`accuracy_improvement_plan.md` Phase 5 完了宣言節) に整合させる: **release_test CI 下限 > 0.85**、monitoring_set 100+ で CI 下限 > 0.90 | ✅ 本ドキュメント修正済 | n=44 で達成不能な閾値の設定ミスを修正 |

**Step 0 ゲート**:
- ✅ **S0-1, S0-2, S0-3 (ファイル), S0-4, S0-5, S0-6 すべて完了**
- ⏳ 残作業 (オーナー手動): GEMINI_API_KEY を Google AI Studio で revoke + 再発行、`.env` と HF Space secret を更新

**Step 0 でついでに準備したもの (前倒しで Phase 5.2 / 5.3 の地盤を作成)**:
- `api/postprocess.py` に **集合体分離 4 手法** (color_kmeans / watershed / connected_components + dispatcher) を実装、`SPLIT_METHOD` で切替、デフォルト off。Phase 5.2 のアブレーション基盤完成
- `tests/eval/calibrate_thresholds.py` を新規実装、**画像単位 PR カーブ** で target_recall 制約付き閾値選択。v5 cache スモーク済 (target_recall=0.95 で threshold=0.959, precision=0.264)。Phase 5.3 即実行可

### 4.1 戦略

> **「SAM2 が見落とすか、まとめて取るか」が現在のボトルネック。
>  これを段階的に解消し、最終的に少量データで SAM2 を腐食特化 fine-tune する。」**

4 段階で積み上げ、各段階で `val` 80 枚評価 → 95% CI 下限の前進を確認してから次へ。

### 4.2 優先順位付き実行計画

| 段階 | 工数 | コスト | 期待 critical_recall | 期待 per_gt_recall | 採否ゲート |
|------|------|--------|---------------------|-------------------|----------|
| **5.1** SAM2 高粒度化 | **1 日** | $0 | 0.85〜0.90 | 0.20〜0.30 | val Δ95% CI 下限 > 0 |
| **5.2** 集合体分離後処理 | **3〜5 日** | $0 | (-) | +0.05〜0.10 | val Δ95% CI 下限 > 0 |
| **5.3** 300 枚ラベル + 閾値校正 | **2〜3 週** | データ作業 | +0.03〜0.05 | (-) | precision@K +0.10 以上 |
| **5.4** SAM2 LoRA fine-tune | **1〜2 週** | GPU $10 程度 | **0.95〜0.98** | **0.50〜0.70** | release_test で達成宣言 |
| 5.5 マルチスケール推論 | 1 週 | $0 | +0.01〜0.03 | +0.02〜0.05 | 5.4 で 95% 未達のみ実施 |

⭐ **5.4 が 95% 到達の最有力候補**。5.1〜5.3 はその前段の地盤固め。

### 4.3 各段階の即実行内容

#### Phase 5.1b — SAM2 高粒度化 + Gemini OFF (v7 改訂版)

**v7 で変わった点**:
- Gemini OFF を追加 (コスト 0、subtype 連結問題回避)
- **`HYBRID_MAX_CLASSIFY` を `SAM2_MAX_MASKS` と同じ値に** — Phase 5.1a 失敗の教訓

**作業**: 環境変数 4 つを set
```
SAM2_POINTS_PER_SIDE=32      # 24 → 32 (576点 → 1024点)
SAM2_MAX_MASKS=40            # 25 → 40
HYBRID_MAX_CLASSIFY=40       # ⭐ default 8 → 40 (= SAM2_MAX_MASKS、フィルタ罠回避)
HYBRID_SKIP_GEMINI=true      # ⭐ Gemini 完全 OFF (コスト 0)
```

**評価コマンド**:
```powershell
python -m tests.eval.run_eval --split val --variant v5.1b_sam2_full_no_gemini `
    --api-url http://127.0.0.1:5000/api/analyze `
    --cache tests/eval/reports/cache_v5_1b_no_gemini.json
```

**判定**: `baseline.md` に追記、95% CI 下限が v5 (0.600) を上回れば採用。
**コスト: $0.00**、所要時間 40-50 分 (CPU)。

**期待値**: critical_recall **0.85〜0.95**、per_gt_recall **0.20〜0.35** (SAM2 1024 点 × 40 mask フル活用 + Gemini の偽陰性損失なし)。

**採否ゲート (v8 改訂、CODEX P1-3 反映 — recall 探索モードと運用採用モードを分離)**:

| モード | 用途 | 必須条件 |
|--------|------|---------|
| **recall 探索モード** (Phase 5.1/5.2 中の比較) | 改善方向の検証 | critical_recall 95% CI 下限が前 variant より上 |
| **運用採用モード** (本番反映の判断) | デプロイ可否 | 上記 + **`manual_review_rate ≤ 0.50`** + **`precision@K=30 ≥ 0.30`** |

manual_review_rate=1.000 は「弁を全開にした」状態 (CODEX 指摘 3) なので、本番採用には不可。
recall 探索モードでは許容するが、Phase 5.3 の閾値校正でレビュー率を抑える設計に進む。

#### Phase 5.2 — 集合体分離後処理 (来週、CODEX レビュー P2-1 反映)

**4 手法の同条件アブレーション** で per_gt_recall と precision の両面評価:

| 手法 | 実装 | 想定強み | 想定弱み |
|------|------|---------|----------|
| **A: 色 k-means** | RGB クラスタリング (k=3〜5) → 連結成分分離 | 異色腐食の分離 | 影・塗装・シール跡を分割しがち (FP 増) |
| **B: Watershed** | 距離変換 → marker → watershed (OpenCV) | 形状ベースで物体境界に強い | パラメータ感度高 |
| **C: Connected Components** | mask の thin region で自然分離 | シンプル、副作用少 | 真にくっついた集合体は分離不可 |
| **D: SAM2 内部小領域再抽出** | 大マスクの bbox 内で point prompt 再呼び | SAM2 の本来の精度を活用 | 推論時間 +30% |

**実装場所**: `api/postprocess.py` に `split_large_mask_*()` を 4 手法ぶん追加し、
HybridDetector の Stage 1 と Stage 2 の間に挟む。

**評価 (v7: Gemini OFF で実施)**: 各手法を val 80 枚で `critical_recall`, `per_gt_recall`, **`precision@K=30`**, `manual_review_rate`, `latency_p95` の 5 軸で記録。
採否は「per_gt_recall +Δ かつ precision@K の悪化が CI 上限内」を条件とする (per_gt_recall 単独最大化を避ける)。

**4 手法のアブレーションコマンド (Gemini OFF、コスト 0)**:
```powershell
$env:HYBRID_SKIP_GEMINI = "true"
$env:HYBRID_MAX_CLASSIFY = "40"   # MAX_MASKS と揃える

# 手法 A: 色 k-means
$env:SPLIT_METHOD = "color_kmeans"
python -m tests.eval.run_eval --variant v5.2a_kmeans --cache .../cache_v5_2a.json ...

# 手法 B: Watershed
$env:SPLIT_METHOD = "watershed"
python -m tests.eval.run_eval --variant v5.2b_watershed --cache .../cache_v5_2b.json ...

# 手法 C: Connected Components
$env:SPLIT_METHOD = "connected_components"
python -m tests.eval.run_eval --variant v5.2c_cc --cache .../cache_v5_2c.json ...
```

#### Phase 5.3 — 300 枚ラベル + 閾値校正 (今後 2〜3 週)

**3 ソースから 300 枚を確保**:
1. デプロイ後の運用画像から 100〜200 枚 (実フィールド)
2. Roboflow `subsea_pipeline` + `in_pipe_corrosion` (CC BY 4.0、計 846 枚から抽出)
3. CCS train 余り (212 枚) からの追加サンプリング

**Label Studio で 2 名合議 (Cohen's κ ≥ 0.7 確認)** → `tests/eval/datasets/labels_v1.json` に追記。

**閾値校正 (v8 改訂: SAM2 純粋スコアのみ、Gemini confidence は使わない)**: 
`tests/eval/calibrate_thresholds.py` を v8 で改訂予定。

scoring_rule §3 (予算単位 = 画像) に整合し、かつ LLM 由来の不安定性を排除するため、
画像スコアを **決定的特徴の合成**で作る:

```python
# 各画像 i について以下から非 LLM スコアを合成:
#   - SAM2 predicted_iou の最大値 (mask 品質の指標)
#   - mask の正規化面積の最大値 (大きい腐食の指標)
#   - color_score の最大値 (赤茶率)
#   - 上記 3 軸を [0, 1] スケールで重み付け平均
# y_true[i]  = 1 if gt_is_critical else 0
# y_score[i] = w1 * sam2_iou_max + w2 * area_norm_max + w3 * color_score_max
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
# critical_recall >= 0.95 制約下で precision 最大の閾値を採用
```

**300 枚ラベル後の発展 (v8 推奨)**:
重み付け平均の代わりに **LightGBM / Logistic Regression / 小型 CNN** を学習。
入力は同じ決定的特徴ベクトル、出力は image-is-critical 確率。
LLM 出力は学習にも校正にも一切使わない。

**前提作業**: `sam2_detector.py` の出力に `predicted_iou` と正規化面積を `Detection.meta` で保存
(現状 cache には個別保存されていないため、再キャッシュが必要)。

#### Phase 5.4 — SAM2 LoRA Fine-tune (95% 到達の本命、1〜2 週)

**前提条件 (CODEX レビュー P3 反映、Phase 5.3 と並行で揃える必須)**:

データの「数」だけでなく「内訳」を管理する:

| 指標 | 目標 | 理由 |
|------|------|------|
| 画像枚数 | 300 枚以上 | LoRA train の最低ライン |
| critical 領域数 | 600 以上 (1 画像あたり平均 2 個) | 学習信号の量 |
| Hard negative 画像 | 60 枚以上 (画像の 20%) | 影・塗装・シール跡など FP 源を含む non-critical 画像 |
| 照明スライス | normal / low_light / backlight それぞれ 30+ 枚 | 暗所 critical_recall の保証 |
| 角度スライス | frontal / oblique それぞれ 30+ 枚 | 斜め撮影への robustness |
| 材質スライス (可能なら) | sus / carbon_steel それぞれ 30+ 枚 | ドメイン拡張 |

**Fine-tune モード選定 (重要)**:
- **Automatic mode 改善 (= mask decoder の prior 強化)**: 推論時の `points_per_side` ベース呼び出しを腐食に偏らせる。本プロジェクトは automatic mode を使うのでこちら。
- **Box / Point prompt 改善**: 推論時にユーザが prompt を渡す前提。本プロジェクトでは使わない (UI に prompt UI なし)。
- 学習スクリプトは「automatic mode の prior 強化」モードで書く。検証は `SAM2_MODE=automatic` の現行パイプラインそのままで実施。

**手順**:
1. Runpod / Colab Pro で A100 1 時間借りる ($2)
2. SAM2 公式 fine-tune notebook ベースに LoRA train (300 枚、エポック 5〜10、約 2〜4 時間)
3. ckpt → `checkpoints/sam2_corrosion_lora.pt` に保存
4. `.env` 切替: `SAM2_CHECKPOINT=checkpoints/sam2_corrosion_lora.pt`
5. val 評価 → release_test 評価 → 95% 達成宣言

**95% 達成基準** (本体計画書 §Phase 5 完了宣言と整合、CODEX レビュー P1-3 反映):
- release_test (n=44) で `critical_recall ≥ 0.95`、**95% CI 下限 > 0.85**
  (n=44 で CI 下限 > 0.90 を要求すると 44/44 完全達成必須となり過剰、本体計画書の閾値を採用)
- monitoring_set 累積 100+ 枚で 3 ヶ月継続して **CI 下限 > 0.90** を維持
- `docs/baseline.md` に最終値を記録

#### Phase 5.5 — マルチスケール推論 (任意、5.4 で未達なら)

512 / 768 / 1024 の 3 スケールで推論 → 結果アンサンブル。
GPU なら並列化可能で総時間 ~1.5 倍。

### 4.4 タイムライン (4 週ロードマップ)

```
Week 1 (今週)         : Phase 5.1 (1 日) → ローカル & HF 反映
Week 2 (来週)         : Phase 5.2 集合体分離実装・評価
Week 2-3              : Phase 5.3 ラベリング作業並行 (運用画像収集 + Roboflow)
Week 3-4              : Phase 5.3 校正 → Phase 5.4 GPU fine-tune
Week 4 終了時         : release_test で 95% 評価、達成宣言 or 5.5 へ
```

### 4.5 失敗時の判断基準

- **Phase 5.4 完了後も release_test で 95% CI 下限 < 0.85 (n=44) または monitoring_set CI 下限 < 0.90** の場合:
  - 計画書 v7 で根本的設計見直し (U-Net 専用モデル、YOLOv8-seg、Mask R-CNN 検討)
  - or 「画像単位検出 + 検査員レビュー UI」のハイブリッド運用に妥協

- **Phase 5.X で前 variant 比 95% CI 下限が上がらない** → revert、次フェーズへ

---

## 4.6 v8 Gemini 役割再定義 (CODEX 第 2 次レビュー反映)

### 4.6.1 v7 → v8 の決定的な変化

| 項目 | v7 | **v8** |
|------|------|------|
| Gemini の役割 | 検出経路の分類器 (MAX_CLASSIFY=3 で投入) | **注釈器** (subtype/severity/reasoning のみ) |
| Gemini が non_corrosion 判定 | 候補削除 | **削除しない**、`gemini_is_corrosion=false` を meta に記録 |
| Gemini API 失敗時 | 候補削除 | **SAM2 候補は必ず残す**、`gemini_failed=true` を meta に記録 |
| confidence の合成 | SAM2 × Gemini の幾何平均 | **SAM2 spatial confidence のみ** |
| review_flag 決定 | Gemini が一部関与 | **Gemini 完全不関与** (SAM2 + 決定的特徴のみ) |
| 閾値校正の入力 | hybrid confidence (LLM 混入) | **SAM2 predicted_iou + mask area + color_score** |

### 4.6.2 段階別の Gemini 設定 (v8)

| Phase | 目的 | Gemini | MAX_CLASSIFY | 推定 Gemini call | 推定コスト |
|-------|------|--------|--------------|----------------|-----------|
| **5.1b** | SAM2 高粒度化の効果を測る | OFF | 40 | 0 | **$0.00** |
| **5.2** | 集合体分離 4 手法アブレーション | OFF | 40 | 0 | **$0.00** |
| **5.3** | 閾値校正 (画像単位、SAM2 純粋スコア) | OFF | 40 | 0 | **$0.00** |
| **5.3'** | (任意) subtype 補助で Gemini 注釈 | ON 注釈のみ | 40 | 80 × 40 = 3200 | ~$0.96 |
| **5.4** | LoRA fine-tune 検証 (検出のみ) | OFF | 40 | 0 | **$0.00** |
| 5.5 | マルチスケール (任意) | OFF | 40 | 0 | $0.00 |
| **検出系合計** | | | | | **$0.00** |

注: 5.3' は注釈用途で削除リスクが無いため任意。subtype が運用上不要なら全部 OFF で進行可能 (合計 $0.00)。

### 4.6.2 なぜ Gemini OFF で SAM2 効果が測れるか

`Detection` の default は `visual_label="corrosion_visible"`。
`HYBRID_SKIP_GEMINI=true` だと SAM2 + 色フィルタの polygon がそのまま返り:

| 指標 | Gemini OFF で測れるか | 理由 |
|------|---------------------|------|
| ✅ critical_recall | 完全に測れる | polygon IoU ≥ 0.3 で TP 判定、scoring_rule §4.1 条件 3 を default で満たす |
| ✅ per_gt_recall | 完全に測れる | 領域 IoU ベース、Gemini 不要 |
| ✅ F1@IoU=0.3 | 完全に測れる | 同上 |
| ✅ latency_p95 | 改善 | Gemini なしで高速化 (~30s/画像) |
| ⚠ precision@K=30 | SAM2 confidence 降順で測れる | manual_review_rate=1.000 固定だが top-K は SAM2 conf 順 |
| ❌ subtype 分布 | 測れない | Phase 5.3 以降で取得 |

### 4.6.3 環境変数の単一ソース化 (Phase 5.1a 失敗の教訓)

`HYBRID_MAX_CLASSIFY` は元々「Gemini に送る上限」用だが、`SKIP_GEMINI=true` でも同じトリミングが効く。
そのため SKIP_GEMINI モード時は **SAM2_MAX_MASKS と同じ値**にしないと SAM2 出力が頭打ちになる。

**正しい組み合わせ表**:

| モード | SAM2_MAX_MASKS | HYBRID_MAX_CLASSIFY | 備考 |
|--------|----------------|---------------------|------|
| Gemini OFF (Phase 5.1b/5.2) | 40 | **40 (= SAM2)** | SAM2 出力を全評価 |
| Gemini 最小投入 (Phase 5.3) | 40 | **3** | SAM2 上位 3 のみ Gemini 分類 |

`run_eval.py` のレポート JSON には `effective_env` セクション (S0-5 で実装済) があるので、
組み合わせミスは事後検証可能。

### 4.6.4 Gemini を完全廃止しない理由

- subtype 判定 (pitting / uniform / flaking / crack) は分類ラベルの提供元として有用
- Phase 5.3 の閾値校正には confidence 出力が必要
- 監視・運用文書のための日本語レポート生成に使う (将来)

ただし **検出精度の主戦場ではない** ことを v7 で確定。

---

## 5. 並行運用計画 (Phase 5 実装中も継続)

### 5.1 週次サイクル

- **月**: 運用画像 5〜10 枚の目視スポットチェック + 失敗事例アーカイブ
- **火-木**: Phase 5.X 実装
- **金**: val 評価 → `tests/eval/ablation_log.md` 追記
- **土**: 採否判定、採用なら `.env` を本番反映 + HF Space に再デプロイ

### 5.2 監視指標

- `gemini_call_count` (HF Space ログ): 0 が連続 → API key 失効
- `latency_ms`: 90 秒超は調整を検討
- `manual_review_rate`: 0.95 超は閾値校正タイミング

### 5.3 Cloud Run / GPU 移行検討タイミング

- HF Space CPU 80 秒/画像 → 業務利用で限界感
- Phase 5.1 採用で 100〜120 秒に伸びる → このタイミングで Cloud Run T4 ($0.40/h) 移行を検討
- GPU なら 5〜10 秒/画像、Phase 5.4 fine-tune 後の運用も同環境で実施

---

## 6. 判断記録 (なぜこの計画か)

1. **「Gemini をもっと頑張らせる」を捨てた**: per_gt_recall=0.032 が天井。Vision LLM は領域認識は得意でもピクセル精度は構造的限界。
2. **「SAM2 fine-tune を最後の砦に」**: 汎用セグメンタ + 少量教師で大幅向上は実証済 (Medium / GitHub 多数)。300 枚で十分。
3. **「ラベリングが律速」**: 5.4 を本命にしても 5.3 のデータが無いと進まない → ラベリングを 5.2 と並行スタートさせるのが要点。
4. **「画像単位の recall を先に守る」**: 業務インパクトは "重大腐食を見逃さない" が最優先。per_gt_recall は二次目標。
5. **「単独承認モード」維持**: 1 人運用なので軽量サイクル (週次レビュー、CI 下限ベースの採否)。

---

## 7. 完了の定義

- [x] v6 デプロイ完了 (HF Space + ローカル並行)
- [ ] Phase 5.1 採用判定
- [ ] Phase 5.2 集合体分離実装 + 採用判定
- [ ] Phase 5.3 ラベル 300 枚到達 + 閾値校正完了
- [ ] Phase 5.4 SAM2 LoRA fine-tune 完了
- [ ] **release_test (44 枚) で critical_recall ≥ 0.95、95% CI 下限 > 0.85** ← ゴール (n=44 制約)
- [ ] **monitoring_set 累積 100+ 枚で CI 下限 > 0.90** を 3 ヶ月継続維持
- [ ] `docs/baseline.md` に「v6 Phase 5 完了、95% 到達」追記

---

**次のアクション (今すぐ、v7 改訂版)**:

サーバウィンドウで:
```powershell
$env:SAM2_POINTS_PER_SIDE = "32"
$env:SAM2_MAX_MASKS       = "40"
$env:HYBRID_MAX_CLASSIFY  = "40"   # ⭐ SAM2_MAX_MASKS と揃える (5.1a 失敗の教訓)
$env:HYBRID_SKIP_GEMINI   = "true" # ⭐ Gemini OFF (コスト 0、subtype 連結問題回避)
.\start_server.ps1
```

評価ウィンドウで:
```powershell
Remove-Item tests/eval/reports/cache_v5_1*.json -ErrorAction SilentlyContinue
python -m tests.eval.run_eval --split val --variant v5.1b_sam2_full_no_gemini `
    --api-url http://127.0.0.1:5000/api/analyze `
    --cache tests/eval/reports/cache_v5_1b_no_gemini.json
```

完走 (~40-50 分、コスト 0) → `docs/baseline.md` に Phase 5.1b 結果追記 → Phase 5.2 (集合体分離 4 手法) へ。

**1 行で言うと**: **「Gemini OFF で SAM2 を磨く (Phase 5.1〜5.2 = $0) → 300 枚ラベル → SAM2 LoRA fine-tune (= $10) → Gemini を分類器として最小限投入で校正 (= $0.14)」が v7 の核心。Gemini を抜くことが、皮肉にも 95% 到達への最短ルート。**
