# Odds Pipeline Audit — 2026-04-18

> **本書の位置付け**
> この文書は `docs/trading_constitution.md`（運用憲法）の §7.1「変更できるもの
> → バグ修正」枠で行うオッズ反映の修正についての整理である。
> LOOSE ルールの 4 つの数値条件は一切変更しない。
> 賭け額・成功指標・撤退条件にも触れない。
> 触るのはオッズの **取得／保存／整合性** のみ。

## 0. TL;DR

現在、システムには **3 つのオッズ源** が並存しており、それぞれが独立に
`entries[*].odds` を書き換えうる構造になっている。結果として
`prediction_log` に記録されるオッズは、レースごとに別々の取得経路・時刻・粒度
の混合物になっており、事後に「この数字はいつ・どこから来たのか」が確定しない。

さらに致命的なのは、`data/snapshot/*.json` を介した **バックテスト側では
221 スナップショット中 152 でオッズが全馬 0** になっており、
`trigger_loose_capped` の第一条件 (`odds > 0`) が毎回落ちる結果、
**221 レースで LOOSE トリガーが 1 回も発火していない**。
憲法 §5.1 が求める「100 ベット以上」はバックテストからは永久に到達できない。

本書はこの構造を 6 つの root cause に分解し、最小限の修正案を提示する。

---

## 1. オッズ源の 3 本立て

| # | 源 | 取得関数 | 型 | 粒度 | 取得タイミング |
|---|----|---------|----|------|----------------|
| A | shutuba HTML (`td[9]`) | `scraper.fetch_entries_netkeiba` | str `"4.5"` / `"---.-"` | HTMLスクレイプ時 | 出馬表スクレイプ時（数日前〜直前）|
| B | netkeiba JSON API (`/api/api_get_jra_odds.html`) | `scraper.fetch_odds_netkeiba` | float (by 馬番) | ほぼリアルタイム | 毎 `predict_live` 呼び出し時 |
| C | 結果ページ (`/race/result.html`) | `scraper.fetch_result_netkeiba` | str | レース後確定 | 結果取得時（レース後） |

**A と B の不一致**：HTML は pari-mutuel 市場の直近のキャッシュを表示するが、
SPA のため初期ロードは `---.-` プレースホルダ。一方で B は常時更新の
JSON API。A の値は数分〜数十分遅延しうる。

**C は単勝 payout の発散原点**。JRA は pari-mutuel なので **確定オッズ = 払戻
オッズ** であり、事後の ROI 計算の唯一の正しい基準は C。

---

## 2. 6 つの構造的問題（root causes）

### RC-1: `_inject_odds_if_missing` の 50% 閾値

`live_pipeline._inject_odds_if_missing` は「shutuba で欠損している馬が
半数未満なら injection しない（partial-kept）」という設計。

```python
# live_pipeline.py:111-118
missing = [e for e in entries if _parse_odds_safe(e.get("odds", 0)) <= 0]
if not missing:
    meta["odds_source"] = "shutuba"
    return "ok", meta
if len(missing) < len(entries) / 2:
    meta["odds_source"] = "shutuba-partial"
    return f"partial-kept ({len(missing)})", meta
```

これは「出走取消で `---` になった 1〜2 頭は無視する」意図で入っているが、
**shutuba HTML が古く全馬に古い数字が載っている場合、欠損ゼロ = `ok` と判定されて
B (live API) は一度も呼ばれない**。結果、ログに載るのは「いつ取られたか不明
な shutuba のキャッシュオッズ」。

### RC-2: 確定オッズが snapshot に入っていない

`data/snapshot/*.json`（221 本、`tools/build_snapshot.py` で生成）は
`entries[*].odds = "0"` のまま保存されている。

- **原因**: snapshot の元データ `enrich_race` キャッシュは出馬表スクレイプ時点
  の HTML をそのまま保存しており、`---.-` → `"0"` 置換のルートを
  通って `"0"` になる（`scraper.py:1286`）。
- **バックテスト実行時**: `tools/snapshot_reader.py:163-171` の
  `_fetch_odds_netkeiba` stub は常に `status: not-published` を返し、
  `_fetch_result_netkeiba` は leak 防止のためハードブロック。
- **結果**: 152/221 スナップショットで `ranked[*].odds == 0` となり、
  `trigger_loose_capped` は
  `(False, "non-positive-odds (0)")` を返し LOOSE ベットが発火しない。

#### 検証結果

```
zero-odds files:    152/221
nonzero-odds files:  69/221
loose_bets fired:     0/221    ← これが致命傷
```

### RC-3: オッズ provenance が per-race 粒度

`odds_api_meta` は **レース単位** でしか保存されない。しかし実際は
partial-kept のとき、「この馬は shutuba、あの馬は API」のような
**馬ごとに源が違う**状況が起きうる（RC-1 の設計上）。
`loose_bets[i].odds` を監査するときに、「この 1 頭のこの数字」の出典が
辿れない。

`prediction_log.py:436` は `"odds": lb.get("odds", 0)` としか書いていない。

### RC-4: bet-time odds と settled odds の二重性が明示されていない

- `loose_bets[i].odds` = `predict_live` 実行時に `sf_horses[name].odds`
  から取った値（bet-time odds）
- `result.finishing_order[i].odds` = 結果ページの確定オッズ (settled odds)

この 2 つは普通 5〜10% ズレる。KPI は `result.payouts.単勝` を使うので
ROI 計算は正しいが、**`recent_loose_bets_table` の表示オッズ**は bet-time
のままで、ユーザが「あれ？ 実際のオッズと違う」と感じる箇所。

### RC-5: `fetch_odds_netkeiba` の下限フィルタの off-by-one

```python
# scraper.py:1189
if um > 0 and 1.0 < v < 10000.0:
    parsed[um] = v
```

`v == 1.0` は JRA の単勝オッズ最小値として合法（超人気馬）。現状では
**オッズ 1.0 ちょうどの馬は silently 捨てられる**。頻度は低いが無視の理由はない。

### RC-6: `---` 系のセンチネル値が `0` に潰されている

```python
# scraper.py:1286
odds = odds.replace("---.-", "0").replace("---", "0")
```

`"0"` は後段で `_parse_odds` → `0.0` に変換され、`odds <= 0` 判定で
「欠損」として扱われる。つまり「オッズ未公開」と「取得失敗」と「オッズ 0」が
**同じセンチネル**で表現される。憲法条文は `odds is None` を欠損として扱って
いるので、言語的にも実装的にも **`None` を使うべき**（現状は `0` で代用）。

---

## 3. 対処方針（最小限のバグ修正）

憲法 §7.1（変更できる＝バグ修正）の範囲内で、次の 4 箇所だけ触る。

### FIX-1: snapshot のオッズを結果ページの確定オッズでバックフィル

- `data/results.json`（316 race、`bt_` プレフィクス）は既にディスク上にあり、
  全 221 snapshot と 1:1 で交差する（`horse_id`・`name` ともに 100% マッチ確認済み）。
- `tools/backfill_snapshot_odds.py` を新設し、snapshot 内 `entries[*].odds`
  が `0` の場合に限って確定オッズを流し込む。
- provenance は `leak_audit.odds_source = "final-odds-from-result"`
  として明示記録。
- **注**: 確定オッズは厳密には post-race 情報だが、
  (i) pari-mutuel なので「直前市場の最終状態」に極めて近く、
  (ii) 単勝 payout の divisor そのものであり、
  (iii) これを使わないと LOOSE トリガーの **オフライン評価自体が成立しない**（全 0 発火）。
  使わない害 >> 使う害なので、**明示ラベル付きで採用**する。

### FIX-2: `live_pipeline._inject_odds_if_missing` の優先順位を修正

- live odds API (`status=result`) が生きているときは、shutuba の欠損率に
  関係なく **API 値で overlay**（上書き）する。
- shutuba は fallback として残す。
- 新ステータス `overlaid-from-live-odds (n)` を追加。

これで「shutuba が古く全部 `5.0` みたいな残骸を吐き続けている」状態でも
直前の真オッズが入る。

### FIX-3: 馬ごとの odds provenance を loose_bets / triggers に添付

- `per_horse_trigger_info[i]` に `odds_source`, `odds_fetched_at` を追加。
- それが `loose_bets[i]` に射影されて `prediction_log` に保存される。
- `recent_loose_bets_table` は `odds_source` を出力カラムに含める。

### FIX-4: `fetch_odds_netkeiba` の off-by-one と comment

- `1.0 < v` → `1.0 <= v` に修正（RC-5）。
- docstring に「v == 1.0 は JRA 単勝最小値として合法」と明記。

---

## 4. やらないこと（scope out）

- LOOSE の 4 条件の **数値**は変更しない（憲法 §7.2）。
- `fact_weighted_score` や probability engine は触らない。
- bet 額（100 yen flat）は変更しない。
- 終了条件・撤退条件は変更しない。
- 「bet-time odds と settled odds を揃える」ような意味論的変更はしない
  （表示カラムの追加のみ）。

---

## 5. 期待される副次効果

| 現状 | FIX 後 |
|------|-------|
| 221 バックテスト中 0 件 loose 発火 | 数十〜100 件オーダーで発火（要測定） |
| prediction_log のオッズの出典不明 | `odds_source` で追跡可能 |
| 単勝 1.0 ちょうどが欠損扱い | 正しく取れる |
| "shutuba 古いが欠損ゼロ" で API 無視 | API 値で overlay |

**憲法 §5.1 が求める 100 ベット統計検証の経路がようやく通る。**

---

## 6. 検証プラン

1. `tools/backfill_snapshot_odds.py --dry-run` で影響範囲を確認
   （何本の snapshot でどれだけ odds が入るか）。
2. 実行後、バックテスト再実行 → LOOSE 発火率を確認。
3. `prediction_log.compute_kpis()` の `loose_bet_roi` が有意味な値で
   返ることを確認（= 母集団 >= 100 に到達できる経路が復活したか）。
4. 実ライブ運用は週末の少なくとも 1 開催で「`odds_source = live-odds-api`」
   がログに載ることを目視確認。

---

## 7. 関連ファイル

| ファイル | 役割 | 本修正での扱い |
|---|---|---|
| `scraper.py` | 3 ソースの原始取得 | FIX-4（1 行修正）|
| `live_pipeline.py` | オッズ注入ロジック | FIX-2, FIX-3 |
| `prediction_log.py` | ログ＋KPI | FIX-3（列追加のみ）|
| `tools/build_snapshot.py` | snapshot 生成 | 未変更（新規スクリプトで対応）|
| `tools/backfill_snapshot_odds.py` | **新設**、snapshot 補正 | FIX-1 |
| `dual_mode_scoring.py` | LOOSE ルール | **変更しない** |
| `docs/trading_constitution.md` | 運用憲法 | **変更しない** |

---

**ルールを守れるかどうかが、勝てるかどうか。**
オッズ反映は「守れているかどうか以前に、**見えているかどうか**」の問題である。
まず見えるようにしてから、守る。

---

## 8. Addendum (2026-04-18 第2波): Multi-source consensus layer

### 問題の続報

第1波 (上記 §1〜§7) 適用後もユーザから
「単勝オッズが明らかにおかしい」との指摘が継続。
単一ソース (netkeiba JSON API) だけでは、
ソース側のキャッシュ・SPA 古値・構造変更いずれでも
検知できないまま stale odds が流れる。

### 設計決定 (ユーザ合意)

- **主権 (primary)**: JRA 公式 (www.jra.go.jp)
- **並列**: netkeiba JSON API, Yahoo 競馬
- **衝突時**: 主権 odds を採用、かつ他ソースと 20% 以上ずれたら
  **その馬の LOOSE 発火を保留** (held)。
  憲法 §7.2 の数値条件は不変更（入力健全性の安全弁のみ追加）。

### 実装

新モジュール `odds_sources.py`：

| 関数 | 役割 |
|---|---|
| `fetch_odds_netkeiba_api(race_id)` | 既存 scraper.fetch_odds_netkeiba のラッパ |
| `fetch_odds_jra_official(...)` | JRA 公式。現状 POST session 制約で "not-implemented" プレースホルダ。構造だけ用意。 |
| `fetch_odds_yahoo_keiba(...)` | Yahoo 競馬 HTML scraper (best-effort) |
| `fetch_odds_consensus(...)` | 3 ソースを並列取得し primary 選択 + disagreement 計算 |

`live_pipeline._inject_odds_if_missing` は consensus を最優先で叩き、
馬ごとに `odds_source`, `odds_by_source`, `odds_disagreement_flag`,
`odds_disagreement_pct` を entry に書く。

LOOSE トリガー評価直後に **disagreement veto** が入り、
`odds_disagreement_flag == True` の馬は
`held: odds source disagreement 50% (jra-official=5.0, netkeiba-api=8.0)`
という reason でドロップされる。発火条件の 4 数値は不変更。

### 永続化

`prediction_log` に以下の per-race フィールドが増える:
- `odds_api_meta.consensus_primary_source`
- `odds_api_meta.consensus_per_source` (各ソースの status/schema/fetched_at)
- `odds_api_meta.consensus_disagreements` (馬番→ {max_pct, values, flag})
- `odds_api_meta.consensus_summary` (human-readable)

`loose_bets` / `triggers` の各行に:
- `odds_by_source` (bet 時の各ソース値)
- `odds_disagreement_flag`
- `odds_disagreement_pct`

`recent_loose_bets_table` はこれら + `odds_settled` を併記するので、
事後に「bet-time の各ソース値 → 確定オッズ」が一目でわかる。

### JRA 公式の現状

POST form + session cookie + hidden token が必要で、
単純 GET では安定取得できない。現在の実装はプレースホルダで
`status="not-implemented"` を返す。

運用での改善候補 (優先度順):
1. **JRA-VAN Data Lab** (有償) への切替。API 形式で安定取得可。
2. POST session 対応。`www.jra.go.jp/JRADB/accessO.html` に
   accessN ページ由来の session cookie + CNAME token を渡す実装。
3. **ラジオNIKKEI 競馬** や **keiba.com** など、JRA データを
   反映する別 HTML サイト経由。

当面は netkeiba JSON API を primary、Yahoo 競馬を secondary と
して運用し、disagreement が出たときは人手で JRA 公式を目視確認する
ことで済ませられる。disagreement veto が入っているので
「気付かず stale odds でベットする」事故は発生しない。

### 成功指標

- 週末 1 開催で `consensus_summary` の分布を確認
- `loose_trigger_reason` に `"held: odds source disagreement"` が
  出る頻度を記録 (過剰だとソースの品質問題、ゼロだと cross-check が
  機能していない可能性)
- 全ソース同一のケースで primary がちゃんと jra-official になっている
  (JRA 実装完了後)

---

## 9. Addendum (2026-04-19 第3波): フォルテアンジェロ事件

### 現象

2026 皐月賞 (race_id `202606030811`) で app_live.py が次のような
「単勝オッズ」を表示：

| 馬名 | 表示された「オッズ」 |
|---|---|
| フォルテアンジェロ | 168.8 |
| アスクエジンバラ | 681.5 |
| アドマイヤクワッズ | 782.7 |

3 桁オッズは単勝では現実味がなく、**数値パターンが 3連単払戻金 ÷ 100**
(16,880 / 68,150 / 78,270 円) あるいは **収得賞金 (100万円単位)** と
一致する。明らかに別カラムを読んでいる。

### 根本原因

`scraper.py:fetch_entries_netkeiba` がオッズ列を **positional index
`cells[9]`** で取得していた。netkeiba の shutuba HTML は改訂のたびに
列が追加・移動しており、2026 時点では `cells[9]` が「収得賞金」
(1.688 億円 → 表示 "168.8") を指している。

加えて、第1波で入れた `_inject_odds_if_missing` は「`odds > 0`
だったらそのまま使う」設計なので、168.8 という「値がある」数字は
"正常" と判断されて consensus overlay 前に確定され、そのまま UI に
伝わる。サニティ境界 `[1.0, 500.0]` も 168.8 は通ってしまう。

### 3 層で同時に防御する修正

**Layer 1 — scraper: class-based selector (`scraper._parse_shutuba_odds`)**

```python
# 旧: odds_td = cells[9]                  ← 列追加で即死
# 新: td = row.select_one("td.Odds")      ← netkeiba が常に付与するクラス
```

`td.Odds` が無い古い/別レイアウトには `cells[9]` にフォールバックするが、
必ず `_SHUTUBA_ODDS_MIN/MAX` でサニティを掛けてから返す。

**Layer 2 — scraper: 範囲外を "0" に畳む**

範囲外の値 (e.g. 168.8 が 収得賞金として入ってくる場合も含め 500 超は
問答無用で) を `"0"` に畳む → 下流で「欠損」扱いとなり overlay 対象になる。
500 以下の偽値 (168.8 等) は Layer 1 の selector で先に弾く。

**Layer 3 — live_pipeline: consensus primary overlay が最終権威**

`_inject_odds_if_missing` に `Step 0: entries sanity clean` を追加し、
500 超の値は即座に "0" に落とす。そして consensus の primary
(通常 netkeiba-api) が存在するときは **shutuba の値に関係なく全頭を
上書き**する (RC-1 の修正済み挙動)。これにより、万が一 Layer 1/2 を
すり抜けた値があっても、最終的な表示は常に consensus 主権ソース由来に
揃う。

### 診断ツール

`tools/diagnose_odds.py` で当該レースを即時検証可能：

```bash
python tools/diagnose_odds.py --race-id 202606030811 --enable-yahoo
```

`Per-source status` セクションに各ソースの `rejected` 数が出るので、
「どのソースがサニティに引っかかる値を返しているか」一目で分かる。

### 回帰テスト (pass 済)

1. `td.Odds` class selector で正しい列を拾う（168.8 ではなく 4.5）
2. `td.Odds` が欠けていても cells[9] フォールバック + サニティで
   500 超は "0" に畳む
3. consensus primary overlay が entries をどんな状態でも上書きする

### 状態

- shutuba-path の構造的バグは解消 (class selector + 3 層サニティ)
- ユーザ側でまだ同じ値が見えるなら、**ブラウザキャッシュ or 古い
  `prediction_log.json` の再描画**の可能性が高い。
  再実行して `consensus_primary_source=netkeiba-api` が UI に
  出ていれば直っている。
