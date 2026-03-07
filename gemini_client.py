"""Gemini AI client for race analysis and PDCA reflection.

Anti-odds-bias design: scoring is driven by environmental, biological,
and strategic signals. Odds are used ONLY to compute expected-value gap,
never as a direct quality proxy.
"""

import json
import re

_gemini_model = None
_current_api_key = None


def _get_model(api_key: str):
    global _gemini_model, _current_api_key
    if _gemini_model is None or api_key != _current_api_key:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            "gemini-2.0-flash-001",
            generation_config={"max_output_tokens": 600, "temperature": 0.3},
        )
        _current_api_key = api_key
    return _gemini_model


def _call_gemini(api_key: str, prompt: str) -> str:
    try:
        model = _get_model(api_key)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[Gemini APIエラー] {e}"


def score_paddock_text(api_key: str, horse_name: str, paddock_text: str) -> dict:
    """
    パドック短評テキストをGeminiで5段階スコアに変換。

    Returns:
      hindquarter_tension: int  1-5  筋肉の張り（トモ）
      coat_gloss:          int  1-5  毛艶（内臓の状態）
      mental_energy:       int  1-5  気合（メンタル）
      summary:             str       1行評価
    """
    if not paddock_text:
        return {"hindquarter_tension": 3, "coat_gloss": 3, "mental_energy": 3, "summary": "情報なし"}

    prompt = f"""あなたは競馬の馬体評価専門家です。「{horse_name}」のパドック短評を読み、以下の3項目を1〜5の整数で採点してください。

パドック短評:「{paddock_text}」

採点基準:
1=最低/要注意, 2=やや悪い, 3=普通/標準, 4=良好, 5=最高/絶好調

JSONのみ返答:
{{"hindquarter_tension":3,"coat_gloss":3,"mental_energy":3,"summary":"1行評価"}}

hindquarter_tension: トモ(後肢・臀部)の筋肉の張り・パンプアップ度
coat_gloss: 毛艶・皮膚の光沢（内臓コンディションの代理指標）
mental_energy: 気合・活気・前向きさ（レース適性メンタル）"""

    raw = _call_gemini(api_key, prompt)
    try:
        data = json.loads(_extract_json(raw))
        return {
            "hindquarter_tension": max(1, min(5, int(data.get("hindquarter_tension", 3)))),
            "coat_gloss": max(1, min(5, int(data.get("coat_gloss", 3)))),
            "mental_energy": max(1, min(5, int(data.get("mental_energy", 3)))),
            "summary": data.get("summary", ""),
        }
    except Exception:
        return {"hindquarter_tension": 3, "coat_gloss": 3, "mental_energy": 3, "summary": raw[:100]}


def analyze_bio_mechanics(
    api_key: str,
    horse_name: str,
    training_physics: dict,
    paddock_scores: dict,
    training_nlp: dict,
    best_weight_analysis: dict,
    transport_profile: dict,
) -> dict:
    """
    Dedicated exercise-physiology analysis for a single horse.

    Returns:
      bio_score:        int      (0–100)
      cardio_rating:    str      (A/B/C/D)
      condition_summary: str
      key_bio_risk:     str
      recommendation:   str
    """
    acc = training_physics.get("acceleration_rate", 0)
    cardio = training_physics.get("cardio_index", 0)
    final_f = training_physics.get("final_split", 0)

    vasc = paddock_scores.get("vascularity_index", 0)
    hindq = paddock_scores.get("hindquarter_power", 0)
    gait = paddock_scores.get("gait_fluidity", 0)

    coat = training_nlp.get("coat_gloss", 0)
    stride = training_nlp.get("stride_quality", 0)
    wt_status = training_nlp.get("weight_status", 0)

    bw = best_weight_analysis
    tp = transport_profile

    prompt = f"""あなたは馬の運動生理学専門家です。「{horse_name}」のバイオメカニカルデータを基に科学的評価を行ってください。

## 調教物理解析 (Training Physics)
- ラスト1F: {final_f}秒
- 加速率(ラスト2F→1F): {acc:+.4f} （正値=加速中、心肺機能が最終局面で稼働）
- 心肺機能指標(Cardio Index): {cardio:+.4f} （3F→2F→1Fの継続加速 = 有酸素能力の高さ）

## 生体コンディション指標 (Visual Condition Proxy) [-1〜1スケール]
- 血管露出指標(Vascularity Index): {vasc:+.3f}  （高=皮膚薄く仕上がり完成）
- 後肢パワー指標(Hindquarter Power): {hindq:+.3f}  （高=トモがパンプアップ、推進力◎）
- 歩様流動性(Gait Fluidity): {gait:+.3f}  （高=重心移動効率高い）

## 調教NLPスコア [-1〜1スケール]
- 毛艶スコア: {coat:+.3f}
- ストライド品質: {stride:+.3f}
- 体重仕上がり: {wt_status:+.3f}  （負値=太め残り）

## ベスト体重分析
- ベスト体重平均: {bw.get('best_weight') or '不明'}kg
- 現在との乖離: {bw.get('deviation', 0):+d}kg
- 判定: {bw.get('classification', '不明')} （信頼度:{bw.get('confidence', 0):.0%}、{bw.get('sample_size', 0)}戦）

## 個体別弱点プロファイル
{chr(10).join('- ' + p for p in tp.get('patterns', ['データなし']))}

JSONのみ返答:
{{"bio_score":75,"cardio_rating":"B","condition_summary":"コンディション要約(1文)","key_bio_risk":"最大リスク要因(1文)","recommendation":"推奨アクション(1文)"}}"""

    raw = _call_gemini(api_key, prompt)
    try:
        data = json.loads(_extract_json(raw))
        return data
    except Exception:
        return {
            "bio_score": 0, "cardio_rating": "?",
            "condition_summary": raw, "key_bio_risk": "", "recommendation": "",
        }


def analyze_race(
    api_key: str,
    race_name: str,
    horses: list[dict],
    track_condition: str = "良",
    weather: str = "",
    temperature: str = "",
    cushion_value: str = "",
    paddock_notes: str = "",
    coat_gloss: str = "",
    hindquarter_pump: str = "",
    weights: dict = None,
    jra_track_data: dict = None,
) -> dict:
    """
    Multi-dimensional race analysis with anti-odds-bias logic.
    Scoring priority: environment + biology + strategy > odds.
    """
    if weights is None:
        weights = {"bio_condition": 0.40, "environment": 0.30, "human_skill": 0.20, "background": 0.10}

    horses_text = "\n".join(
        _format_horse_bio(h) for h in horses
    )

    # JRA公式データをGround Truthとして環境ブロックに反映（優先）
    jra = jra_track_data or {}
    eff_going = jra.get("going") or track_condition
    eff_cushion = jra.get("cushion_value") or cushion_value
    cushion_source = "【JRA公式】" if jra.get("cushion_value") else ""
    going_source = "【JRA公式】" if jra.get("going") else ""

    env_block = (
        f"天気:{weather or '不明'} 気温:{temperature or '不明'} "
        f"馬場状態:{eff_going}{going_source} "
        f"クッション値:{eff_cushion or '不明'}{cushion_source}"
    )

    # JRA公式馬場詳細ブロック
    jra_block = ""
    if jra:
        lines = ["## JRA公式馬場情報 【最優先データ・Ground Truth】"]
        if jra.get("cushion_value"):
            cv = float(jra["cushion_value"])
            hardness = "硬め(8以下=高速馬場→先行有利)" if cv <= 8.0 else \
                       "標準(8〜10=バランス型)" if cv <= 10.0 else "軟め(10超=パワー型有利)"
            lines.append(
                f"クッション値: {cv} → {hardness}"
                f"\n  ※ 硬いほどスピード型・軽い馬体が有利。軟いほどパワー・重心低い馬が有利。"
            )
        if jra.get("water_content_goal"):
            lines.append(f"含水率(ゴール前): {jra['water_content_goal']}")
        if jra.get("water_content_4c"):
            lines.append(f"含水率(4コーナー): {jra['water_content_4c']}")
        if jra.get("track_bias_text"):
            lines.append(f"馬場傾向: {jra['track_bias_text']}")
        if jra.get("inner_rail_moved"):
            lines.append("⚠ 内柵移動あり → 内ラチ沿いの距離ロス・有利不利が変化")
        if jra.get("turf_replaced"):
            lines.append("⚠ 芝張り替えあり → 新芝は時計が出やすく前有利傾向")
        if jra.get("scratched"):
            lines.append(f"出走取消: {', '.join(jra['scratched'])}")
        jra_block = "\n".join(lines)

    paddock_block = ""
    if any([paddock_notes, coat_gloss, hindquarter_pump]):
        paddock_block = (
            f"パドック所見:{paddock_notes or 'なし'} "
            f"毛並み光沢(内臓コンディション):{coat_gloss or '未記入'} "
            f"トモのパンプアップ:{hindquarter_pump or '未記入'}"
        )

    prompt = f"""あなたは「反オッズ依存型×運動生理学」の競馬AI専門家です。
【絶対禁止】オッズが低い（人気がある）という理由だけで馬を評価すること。
【黄金比】 生体(40%) > 環境(30%) > 人間(20%) > 背景(10%) で採点し、オッズは期待値ギャップ判定にのみ使用せよ。

## 科学的黄金比フレームワーク
1. 生体・コンディション [{weights.get('bio_condition',0.40):.0%} = 40点]:
   - 調教加速率(Cardio Index): ラスト区間のピッチ向上 = 心肺エンジン稼働証拠
   - Vascularity Index: 皮膚の薄さ / 仕上げ完成度
   - Hindquarter Power: トモのパンプアップ / 推進力ポテンシャル
   - Gait Fluidity: 踏み込みの深さ / 重心移動効率
   - ベスト体重乖離: 過去1・2着時との体重差 + 成長/消耗/仕上がり判定
   - 個体弱点: 高温・長距離輸送での過去パフォーマンス低下
2. 環境・適性 [{weights.get('environment',0.30):.0%} = 30点]:
   - 馬場状態・クッション値への血統&近走適性
   - 天気・気温・降水量の影響
   - 輸送ストレス(km距離)と個体の輸送耐性
3. 人間・相性 [{weights.get('human_skill',0.20):.0%} = 20点]:
   - 騎手の重賞・G1での大舞台勝負強さ
   - 騎手×馬のコンビ成績・呼吸の一致
   - 斤量が馬体に与える負荷
4. 背景・資本 [{weights.get('background',0.10):.0%} = 10点]:
   - 外厩（NFしがらき/NF天栄等）の調整密度
   - 馬主の資金力・使い分け戦略
   - 血統の適性（父系・母系の距離/馬場傾向）
期待値ギャップ(加点): (実力推定スコア - オッズ逆算期待値) > 0 なら最大+10点

## レース: {race_name}
環境: {env_block}
{jra_block}
{paddock_block}

## 出走馬（バイオメカニカルデータ統合）
{horses_text}

JSONのみ返答:
{{"predictions":[{{"rank":1,"name":"馬名","score":85,"ev_gap":"+12","reason":"理由(運動生理学的根拠を含む1文)","bet":"推奨"}},{{"rank":2,"name":"馬名","score":72,"ev_gap":"+5","reason":"理由","bet":"推奨"}},{{"rank":3,"name":"馬名","score":61,"ev_gap":"-2","reason":"理由","bet":"推奨"}}],"overall_comment":"総評(運動生理学視点含む1文)"}}"""

    raw = _call_gemini(api_key, prompt)

    try:
        data = json.loads(_extract_json(raw))
        # Normalize field names: score -> confidence for backward compat
        for h in data.get("predictions", []):
            if "score" in h and "confidence" not in h:
                h["confidence"] = h.pop("score")
        return {
            "horses": data.get("predictions", []),
            "comment": data.get("overall_comment", ""),
            "raw_response": raw,
        }
    except Exception:
        return {"horses": [], "comment": raw, "raw_response": raw}


def generate_reflection(
    api_key: str,
    race_name: str,
    prediction: dict,
    result: dict,
    current_weights: dict,
) -> dict:
    """
    PDCA reflection with odds-bias self-audit:
    did we pick favourites when true value was elsewhere?
    """
    pred_horses = prediction.get("horses", [])
    finishing = result.get("finishing_order", [])
    payouts = result.get("payouts", {})

    pred_text = " / ".join(
        f"予想{h.get('rank')}位:{h['name']}(EV:{h.get('ev_gap','?')})"
        for h in pred_horses
    )
    result_text = " / ".join(
        f"{h['rank']}着:{h['name']}" for h in finishing[:5]
    )
    payout_text = " ".join(f"{k}:{v}円" for k, v in payouts.items())

    prompt = f"""競馬PDCA分析AI。レース「{race_name}」の予想vs結果を黄金比フレームワークで自己評価せよ。

予想: {pred_text or 'なし'}
結果: {result_text or 'なし'} {payout_text}
現重み(黄金比): 生体{current_weights.get('bio_condition',0.40):.0%} 環境{current_weights.get('environment',0.30):.0%} 人間{current_weights.get('human_skill',0.20):.0%} 背景{current_weights.get('background',0.10):.0%}

必ず以下を分析:
A) オッズバイアス監査: 人気馬を過大評価・穴馬の期待値を見逃していなかったか
B) 生体シグナルの精度: 調教加速率・パドックNLPスコア・ベスト体重乖離が的中に寄与したか
C) 環境要因の見落とし: 馬場・天気・輸送ストレスの影響を正しく評価できたか
D) 人間・背景シグナル: 騎手相性・外厩調整の評価が正しかったか

JSONのみ返答(合計1.0):
{{"reflection":"分析(2文)","odds_bias_audit":"バイアス自己評価(1文)","key_lessons":["教訓1","教訓2"],"suggested_weights":{{"bio_condition":0.40,"environment":0.30,"human_skill":0.20,"background":0.10}},"weight_reasoning":"変更理由(1文)"}}"""

    raw = _call_gemini(api_key, prompt)

    try:
        data = json.loads(_extract_json(raw))
        return {
            "reflection": data.get("reflection", raw),
            "odds_bias_audit": data.get("odds_bias_audit", ""),
            "key_lessons": data.get("key_lessons", []),
            "suggested_weights": data.get("suggested_weights", current_weights),
            "weight_reasoning": data.get("weight_reasoning", ""),
        }
    except Exception:
        return {
            "reflection": raw,
            "odds_bias_audit": "",
            "key_lessons": [],
            "suggested_weights": current_weights,
            "weight_reasoning": "",
        }


def _format_horse_bio(h: dict) -> str:
    """Format a single horse dict into a rich bio-mechanical summary line for the prompt."""
    physics = h.get("training_physics") or {}
    paddock = h.get("paddock_scores") or {}
    nlp = h.get("training_nlp") or {}
    bw = h.get("best_weight_analysis") or {}
    tp = h.get("transport_profile") or {}

    # Core identity
    base = (
        f"  {h.get('number','?')}番 {h['name']} "
        f"(騎手:{h.get('jockey','?')}, 斤量:{h.get('weight','?')}kg, "
        f"オッズ:{h.get('odds','?')}倍, 近走:{h.get('recent_form','?')}, "
        f"厩舎:{h.get('stable','?')}, 外厩:{h.get('ritto','?')}, 輸送:{h.get('transport_stress','?')})"
    )

    # Physics line (only if data present)
    phys_parts = []
    if physics.get("final_split"):
        phys_parts.append(f"ラスト1F:{physics['final_split']}秒")
    if physics.get("acceleration_rate") != 0:
        phys_parts.append(f"加速率:{physics['acceleration_rate']:+.3f}")
    if physics.get("cardio_index") != 0:
        phys_parts.append(f"心肺指標:{physics['cardio_index']:+.3f}")
    if nlp.get("weight_status") is not None and nlp["weight_status"] < -0.3:
        phys_parts.append("★太め残り")
    if nlp.get("stride_quality") is not None and nlp["stride_quality"] > 0.3:
        phys_parts.append("踏み込み◎")

    # Bio scores line
    bio_parts = []
    for key, label in [("vascularity_index", "血管露出"), ("hindquarter_power", "トモ力"),
                        ("gait_fluidity", "歩様流動")]:
        v = paddock.get(key)
        if v is not None and v != 0:
            bio_parts.append(f"{label}:{v:+.2f}")

    # Best weight
    bw_parts = []
    if bw.get("best_weight"):
        bw_parts.append(
            f"ベスト体重:{bw['best_weight']}kg({bw.get('deviation', 0):+d}kg) "
            f"→{bw.get('classification','?')}"
        )

    # Weakness
    wp = tp.get("patterns", [])
    weakness = f"個体弱点: {wp[0]}" if wp and wp[0] != "顕著な個体弱点パターン未検出" else ""

    extra = " | ".join(filter(None, [
        " ".join(phys_parts),
        " ".join(bio_parts),
        " ".join(bw_parts),
        weakness,
    ]))
    return f"{base}\n    [{extra}]" if extra.strip() else base


def _extract_json(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        return match.group(1)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return match.group(0)
    return text
