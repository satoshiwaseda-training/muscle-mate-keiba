"""Streamlit UI for 競馬G1/G2/G3 自律進化型予想アプリ."""

import os
import re as _re
from datetime import date

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import scraper
import gemini_client
import pdca_engine
from data_store import (
    save_prediction,
    save_result,
    load_weights,
    get_prediction,
    get_result,
    compute_stats,
    upsert_best_weight_record,
    save_jra_ground_truth,
    get_jra_ground_truth,
    merge_track_data,
    get_weight_history,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="競馬AI予想 - 自律進化型",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# API key: st.secrets → .env → empty (mock mode)
# ─────────────────────────────────────────────

def _resolve_api_key() -> str:
    # 1. Streamlit Cloud Secrets (最優先)
    for secret_key in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        try:
            key = st.secrets[secret_key]
            if key and key not in ("your_gemini_api_key_here", "your_google_api_key_here"):
                return key
        except (KeyError, Exception):
            pass
    # 2. .env / 環境変数
    return os.getenv("GOOGLE_API_KEY", "") or os.getenv("GEMINI_API_KEY", "")


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────

# 毎回Secretsから最新のAPIキーを取得（Secrets追加後も即反映）
_resolved = _resolve_api_key()
if _resolved:
    st.session_state.api_key = _resolved
elif "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "races" not in st.session_state:
    st.session_state.races = []
if "selected_race" not in st.session_state:
    st.session_state.selected_race = None
if "entries" not in st.session_state:
    st.session_state.entries = []
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "weather_data" not in st.session_state:
    st.session_state.weather_data = {}
if "paddock_reports" not in st.session_state:
    st.session_state.paddock_reports = {}
if "jra_track_data" not in st.session_state:
    st.session_state.jra_track_data = {}
# スライダー状態（明示的にkey管理して自動更新を可能にする）
if "sl_coat" not in st.session_state:
    st.session_state.sl_coat = 3
if "sl_hindq" not in st.session_state:
    st.session_state.sl_hindq = 3
if "sl_gait" not in st.session_state:
    st.session_state.sl_gait = 3
# 結果保存後に自動PDCA実行するためのフラグ
if "auto_pdca_race" not in st.session_state:
    st.session_state.auto_pdca_race = None


# ─────────────────────────────────────────────
# Helper functions (must be defined before use)
# ─────────────────────────────────────────────

def _mock_analysis(horses: list, race_name: str) -> dict:
    import random
    shuffled = horses[:]
    random.shuffle(shuffled)
    reasons = [
        "調教加速率が高く心肺機能の仕上がりが確認できる。トモのパンプアップも◎でベスト体重帯。",
        "血統的に馬場適性が高く、前走から斤量減で有利。外厩調整密度も高い。",
        "オッズに対して実力が過小評価されており期待値ギャップ+。輸送ストレス低く侮れない。",
    ]
    bets = ["単勝 + 複勝", "複勝 + ワイド", "連複BOX"]
    predictions = [
        {
            "rank": i + 1,
            "name": shuffled[i]["name"],
            "confidence": [72, 58, 45][i],
            "ev_gap": ["+8", "+3", "-2"][i],
            "reason": reasons[i],
            "bet": bets[i],
        }
        for i in range(min(3, len(shuffled)))
    ]
    return {
        "horses": predictions,
        "comment": f"[デモ分析] {race_name} の予想です（黄金比フレームワーク適用・ランダム）。",
        "raw_response": "",
    }

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

WEIGHT_LABELS = {
    "bio_condition": "生体・コンディション",
    "environment": "環境・適性",
    "human_skill": "人間・相性",
    "background": "背景・資本",
}
WEIGHT_COLORS = {
    "bio_condition": "#00ff88",
    "environment": "#7fdbff",
    "human_skill": "#FFD700",
    "background": "#ff6b6b",
}

with st.sidebar:
    st.title("🏇 競馬AI予想")
    st.caption("自律進化型 G1/G2/G3 予想システム")

    st.divider()

    # ── Gemini API 状態表示 ─────────────────────────────
    if st.session_state.api_key:
        st.success("Gemini API: 接続済み")
        if st.button("接続テスト", use_container_width=True):
            with st.spinner("Gemini疎通確認中..."):
                try:
                    from google import genai as _genai
                    from google.genai import types as _gtypes
                    _client = _genai.Client(api_key=st.session_state.api_key)
                    _r = _client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents="日本語で「接続OK」とだけ返答してください",
                        config=_gtypes.GenerateContentConfig(max_output_tokens=20),
                    )
                    st.success(f"Gemini応答: {_r.text.strip()[:50]}")
                except Exception as _e:
                    st.error(f"Gemini接続エラー: {_e}")
    else:
        st.error("Gemini API: 未接続（Secrets未設定）")
        st.caption("Streamlit Cloud → Settings → Secrets に\nGOOGLE_API_KEY を設定してください")

    st.divider()

    # ── 今週末クイック取得 ────────────────────────────────
    saturday, sunday = scraper.get_this_week_race_dates()
    if st.button(
        f"今週末のレースを取得\n({saturday.strftime('%m/%d土')} & {sunday.strftime('%m/%d日')})",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("今週の土日レースを取得中..."):
            races = scraper.fetch_this_week_races()
            st.session_state.races = races
            st.session_state.selected_race = None
            st.session_state.entries = []
            st.session_state.prediction_result = None
        if races:
            st.success(f"{len(races)}件のG1/G2/G3レースを取得（土日合計）")
        else:
            st.warning("今週末のG1/G2/G3レースが見つかりませんでした")

    st.caption("または特定日を指定して取得")

    # ── 個別日付指定 ─────────────────────────────────────
    # デフォルトを直近土曜に設定
    selected_date = st.date_input(
        "開催日付（個別指定）",
        value=saturday,
        min_value=date(2020, 1, 1),
        max_value=date(2030, 12, 31),
    )

    if st.button("指定日のレースを取得", use_container_width=True):
        with st.spinner("レース情報を取得中..."):
            races = scraper.fetch_race_list(selected_date)
            st.session_state.races = races
            st.session_state.selected_race = None
            st.session_state.entries = []
            st.session_state.prediction_result = None
        if races:
            st.success(f"{len(races)}件のG1/G2/G3レースを取得")
        else:
            st.warning(f"{selected_date.strftime('%m/%d')}のG1/G2/G3レースが見つかりませんでした")

    st.divider()

    # Golden ratio weight display
    weights = load_weights()
    st.subheader("科学的黄金比 (PDCA学習済)")
    for k, v in weights.items():
        label = WEIGHT_LABELS.get(k, k)
        color = WEIGHT_COLORS.get(k, "#aaa")
        st.markdown(
            f'<div style="margin-bottom:6px;">'
            f'<span style="color:{color};font-weight:bold;">{label}</span>'
            f'<span style="color:#aaa;font-size:0.85em;"> {v:.0%}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.progress(min(v, 1.0))


# ─────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────

st.title("🏇 競馬AI予想システム")

tab1, tab2, tab3, tab4 = st.tabs(["予想", "結果入力 & PDCA", "ダッシュボード", "過去Gレース学習"])

# ═════════════════════════════════════════════
# Tab 1: Prediction
# ═════════════════════════════════════════════

with tab1:
    st.header("レース予想")

    if not st.session_state.races:
        st.info("サイドバーから日付を選択し「レース情報取得」をクリックしてください。")
    else:
        race_options = {
            f"[{r['grade']}] {r['race_name']} {r['venue']} {r['time']}"
            + (f" 【{r['race_date']}】" if r.get("race_date") else ""): r
            for r in st.session_state.races
        }
        selected_label = st.selectbox("レース選択", options=list(race_options.keys()))
        selected_race = race_options[selected_label]

        # ── 天気自動取得 ──────────────────────────────────
        col_weather_btn, col_weather_info = st.columns([1, 3])
        with col_weather_btn:
            if st.button("天気を自動取得 (OpenMeteo)", use_container_width=True):
                venue = selected_race.get("venue", "")
                try:
                    race_hour = int(selected_race.get("time", "15:00").split(":")[0])
                except Exception:
                    race_hour = 15
                # race_date フィールドがあればそれを優先、なければ selected_date
                _rdate = selected_race.get("race_date")
                if _rdate:
                    from datetime import date as _date
                    _fetch_date = _date.fromisoformat(_rdate)
                else:
                    _fetch_date = selected_date
                with st.spinner("天気データ取得中..."):
                    st.session_state.weather_data = scraper.fetch_weather(
                        venue, _fetch_date, race_hour
                    )

        weather_data = st.session_state.weather_data
        with col_weather_info:
            if weather_data:
                st.info(
                    f"**{weather_data.get('description','')}**  "
                    f"気温:{weather_data.get('temperature','')}  "
                    f"降水:{weather_data.get('precipitation','')}  "
                    f"風速:{weather_data.get('windspeed','')}  "
                    f"({weather_data.get('source','')})"
                )

        # ── JRA公式馬場情報 (Ground Truth) ───────────────
        st.markdown("---")
        jra_col1, jra_col2 = st.columns([1, 3])
        with jra_col1:
            if st.button("JRA公式馬場情報を取得\n(Ground Truth)", use_container_width=True, type="primary"):
                _rdate = selected_race.get("race_date")
                from datetime import date as _date
                _fetch_date = _date.fromisoformat(_rdate) if _rdate else selected_date
                with st.spinner("JRA公式サイトから馬場情報・取消情報を取得中..."):
                    jra_track = scraper.fetch_jra_track_conditions(
                        selected_race.get("venue", ""), _fetch_date
                    )
                    jra_changes = scraper.fetch_jra_race_changes(selected_race["race_id"])
                    # マージ
                    jra_merged = {**jra_track, **{k: v for k, v in jra_changes.items() if v}}
                    st.session_state.jra_track_data = jra_merged
                    save_jra_ground_truth(selected_race["race_id"], jra_merged)
                st.rerun()
        with jra_col2:
            jra_data = st.session_state.jra_track_data
            if jra_data:
                items = []
                if jra_data.get("cushion_value"):
                    cv = jra_data["cushion_value"]
                    try:
                        cv_f = float(cv)
                        hardness = "硬め" if cv_f <= 8.0 else "標準" if cv_f <= 10.0 else "軟め"
                        items.append(f"**クッション値: {cv}** ({hardness})")
                    except Exception:
                        items.append(f"**クッション値: {cv}**")
                if jra_data.get("water_content_goal"):
                    items.append(f"含水率(ゴール前): {jra_data['water_content_goal']}")
                if jra_data.get("water_content_4c"):
                    items.append(f"含水率(4C): {jra_data['water_content_4c']}")
                if jra_data.get("going"):
                    items.append(f"馬場状態: **{jra_data['going']}**")
                if jra_data.get("inner_rail_moved"):
                    items.append("⚠ 内柵移動あり")
                if jra_data.get("turf_replaced"):
                    items.append("⚠ 芝張り替えあり")
                if jra_data.get("scratched"):
                    items.append(f"🚫 取消: {', '.join(jra_data['scratched'])}")
                if items:
                    st.info("【JRA公式 Ground Truth】 " + "　|　".join(items))
                    if jra_data.get("track_bias_text"):
                        st.caption(f"馬場傾向: {jra_data['track_bias_text']}")
            else:
                st.caption("JRA公式馬場情報未取得（レース当日に取得するとAI精度が向上します）")

        # ── 環境情報 ──────────────────────────────────────
        st.caption("環境・馬場情報（JRA公式取得後は自動反映・手動調整可）")
        jra_data = st.session_state.jra_track_data
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            going_opts = ["良", "稍重", "重", "不良"]
            jra_going = jra_data.get("going", "")
            going_default = going_opts.index(jra_going) if jra_going in going_opts else 0
            track_condition = st.selectbox(
                "馬場状態" + (" 【JRA公式】" if jra_going else ""),
                going_opts, index=going_default,
            )
        with col2:
            weather_opts = ["晴", "曇", "小雨", "雨", "雪", "不明"]
            auto_weather = jra_data.get("weather") or weather_data.get("description", "")
            w_default = next((i for i, w in enumerate(weather_opts) if w in auto_weather), 5)
            weather = st.selectbox("天気", weather_opts, index=w_default)
        with col3:
            auto_temp = weather_data.get("temperature", "").replace("℃", "")
            temperature = st.text_input("気温 (℃)", value=auto_temp, placeholder="例: 18.5")
        with col4:
            jra_cv = jra_data.get("cushion_value", "")
            cushion_value = st.text_input(
                "クッション値" + (" 【JRA公式】" if jra_cv else ""),
                value=jra_cv, placeholder="例: 9.2",
            )

        # ── 詳細データ取得 ────────────────────────────────
        col_fetch, col_fetch_status = st.columns([1, 3])
        with col_fetch:
            fetch_detail = st.button(
                "詳細データ取得\n(騎手・調教・血統)",
                help="netkeiba DBから各馬の詳細情報を取得します。約1分かかります。",
                use_container_width=True,
            )

        if fetch_detail:
            with st.spinner("出走馬ベースデータ取得中..."):
                base_entries = scraper.fetch_entries(
                    selected_race["race_id"], venue=selected_race.get("venue", "")
                )

            progress_bar = st.progress(0)
            status_text = st.empty()

            def _progress(i, total, name):
                progress_bar.progress((i + 1) / total)
                status_text.text(f"詳細取得中 ({i+1}/{total}): {name}")

            with st.spinner("騎手・血統・調教データ取得中..."):
                enriched = scraper.enrich_entries(
                    base_entries, selected_race["race_id"], _progress
                )
            progress_bar.empty()
            status_text.empty()

            for h in enriched:
                hid = h.get("horse_id", "")
                current_w = 0
                m = _re.search(r"(\d{3,4})kg", h.get("weight_trend", ""))
                if m:
                    current_w = int(m.group(1))
                h["best_weight_analysis"] = (
                    scraper.compute_best_weight_analysis(hid, current_w)
                    if hid and current_w else {}
                )
                h["transport_profile"] = (
                    scraper.build_transport_weight_profile(hid) if hid else {}
                )

            st.session_state.entries = enriched
            st.success(f"{len(enriched)}頭の詳細データを取得しました")

        with col_fetch_status:
            if st.session_state.entries and st.session_state.entries[0].get("bloodline"):
                st.caption(f"詳細データ取得済: {len(st.session_state.entries)}頭")

        # ── パドック所見 ──────────────────────────────────
        st.markdown("---")
        st.subheader("パドック所見（生体コンディション評価）")

        # 自動取得ボタン
        col_pad_btn, col_pad_status = st.columns([1, 3])
        with col_pad_btn:
            if st.button("パドック情報を自動取得\n(調教/SNS/ニュース)", use_container_width=True):
                entries_now = st.session_state.entries or []
                horse_names = [h["name"] for h in entries_now]
                with st.spinner("パドック情報を複数ソースから取得中..."):
                    reports = {}

                    # ① すでに取得済みの調教評価テキストをパドック代理として使用（最速・確実）
                    for h in entries_now:
                        name = h["name"]
                        sources = []
                        # 調教評価コメント
                        if h.get("training_eval"):
                            sources.append(scraper._clean_text(h["training_eval"]))
                        # 近走成績テキスト
                        if h.get("recent_form"):
                            sources.append(scraper._clean_text(h["recent_form"]))
                        combined = " ".join(sources)
                        if combined.strip():
                            reports[name] = {
                                "text": combined,
                                "source": "調教データ(取得済)",
                                "scores": scraper.parse_paddock_comment(combined),
                                "weight_kg": None,
                                "weight_change": None,
                            }

                    # ② netkeiba/SNS/ニュース系（追加テキスト取得）
                    web_reports = scraper.fetch_paddock_reports(
                        race_id=selected_race["race_id"],
                        horse_names=horse_names,
                        race_name=selected_race["race_name"],
                    )
                    for name, rep in web_reports.items():
                        if rep.get("text"):
                            if name in reports:
                                # 既存テキストに追記してスコア再計算
                                merged_text = reports[name]["text"] + " " + rep["text"]
                                reports[name]["text"] = merged_text
                                reports[name]["source"] += f"+{rep['source']}"
                                reports[name]["scores"] = scraper.parse_paddock_comment(merged_text)
                            else:
                                reports[name] = rep

                    # ③ KeibaLab 馬体重
                    kl_weights = scraper.fetch_keibalab_horse_weights(
                        selected_race["race_id"], horse_names
                    )
                    for name, kl in kl_weights.items():
                        if name not in reports:
                            reports[name] = {"text": "", "source": "", "scores": {}}
                        reports[name]["weight_kg"] = kl.get("weight")
                        reports[name]["weight_change"] = kl.get("change")
                        if kl.get("comment"):
                            reports[name]["text"] += " " + kl["comment"]

                    # ④ Gemini NLPスコアリング（テキストがある馬のみ）
                    if st.session_state.api_key:
                        for name, rep in reports.items():
                            if rep.get("text") and len(rep["text"]) > 5:
                                nlp = gemini_client.score_paddock_text(
                                    st.session_state.api_key, name, rep["text"]
                                )
                                rep["gemini_scores"] = nlp

                    st.session_state.paddock_reports = reports

                    # ⑤ スライダーを全馬平均スコアで自動更新
                    def _avg_score(key: str) -> int:
                        vals = [v["scores"].get(key, 0) for v in reports.values()
                                if v.get("scores")]
                        if not vals:
                            return 3
                        avg = sum(vals) / len(vals)
                        return max(1, min(5, round(avg * 2 + 3)))

                    st.session_state.sl_coat  = _avg_score("vascularity_index")
                    st.session_state.sl_hindq = _avg_score("hindquarter_power")
                    st.session_state.sl_gait  = _avg_score("gait_fluidity")

                found = sum(1 for v in reports.values() if v.get("text"))
                if found:
                    st.success(f"{found}頭分のパドック情報を取得・スライダーを自動更新しました")
                else:
                    st.warning("パドック情報が見つかりませんでした（詳細データを先に取得してください）")
                st.rerun()

        # 取得済みレポートを表示
        paddock_reports = st.session_state.paddock_reports
        if paddock_reports:
            with col_pad_status:
                sources = list({v["source"] for v in paddock_reports.values() if v.get("source")})
                st.caption(f"取得元: {', '.join(sources)}")

            with st.expander("取得したパドック情報（馬別・Gemini採点）", expanded=False):
                _score_colors = {1: "#ff4444", 2: "#ff9900", 3: "#aaaaaa", 4: "#88ee44", 5: "#00ff88"}
                _score_icons  = {1: "▼▼", 2: "▼", 3: "－", 4: "▲", 5: "▲▲"}
                for hname, rep in paddock_reports.items():
                    if not rep.get("text") and not rep.get("weight_kg"):
                        continue
                    src = rep.get("source", "")
                    src_badge = f'<span style="color:#FFD700;font-size:0.8em;">[{src}]</span>' if src else ""

                    # 馬体重表示
                    wt_str = ""
                    if rep.get("weight_kg"):
                        chg = rep.get("weight_change", 0)
                        chg_color = "#00ff88" if chg > 0 else "#ff4444" if chg < 0 else "#aaa"
                        wt_str = (
                            f'<span style="color:#7fdbff;font-weight:bold;">'
                            f'{rep["weight_kg"]}kg'
                            f'<span style="color:{chg_color}">({chg:+d})</span></span>　'
                        )

                    # Geminiスコア表示
                    gs = rep.get("gemini_scores", {})
                    gemini_str = ""
                    if gs:
                        def _badge(label, val):
                            c = _score_colors.get(val, "#aaa")
                            ic = _score_icons.get(val, "－")
                            return f'<span style="color:{c};font-size:0.85em;">{label}:{ic}{val}</span>'
                        gemini_str = (
                            "　" +
                            _badge("トモ", gs.get("hindquarter_tension", 3)) + "　" +
                            _badge("毛艶", gs.get("coat_gloss", 3)) + "　" +
                            _badge("気合", gs.get("mental_energy", 3))
                        )
                        if gs.get("summary"):
                            gemini_str += f'　<span style="color:#ccc;font-size:0.8em;">→{gs["summary"]}</span>'

                    st.markdown(
                        f'**{hname}** {src_badge}　{wt_str}{gemini_str}<br>'
                        f'<span style="color:#ddd;font-size:0.9em;">{rep.get("text","")}</span>',
                        unsafe_allow_html=True,
                    )
                    st.divider()

        st.caption("各指標を 1（最低）〜 5（最高）で評価（パドック自動取得後は自動更新）")

        paddock_notes = st.text_input(
            "総合所見メモ（補足）",
            placeholder="例: 3番の血管が浮き出ている。5番は踏み込みが深く滑らか。",
        )

        _label_colors = {1: "#ff4444", 2: "#ff9900", 3: "#aaaaaa", 4: "#88ee44", 5: "#00ff88"}
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            coat_score = st.slider(
                "毛並み・光沢（内臓コンディション）",
                min_value=1, max_value=5, key="sl_coat",
                help="1=くすみ/疲労感あり　5=ピカピカ/仕上げ完成",
            )
            coat_labels = {1: "くすんでいる(要注意)", 2: "やや悪い", 3: "普通",
                           4: "光沢あり", 5: "光沢あり(優秀)"}
            coat_gloss = coat_labels[coat_score]
            st.markdown(
                f'<div style="color:{_label_colors[coat_score]};font-weight:bold;">'
                f'→ {coat_gloss}</div>', unsafe_allow_html=True)

        with col_p2:
            hindq_score = st.slider(
                "トモのパンプアップ（推進力ポテンシャル）",
                min_value=1, max_value=5, key="sl_hindq",
                help="1=張りなし/細い　5=パンプアップ最高/力強い",
            )
            hindq_labels = {1: "張りなし", 2: "やや甘い", 3: "普通",
                            4: "パンプアップ良好", 5: "パンプアップ最高"}
            hindquarter_pump = hindq_labels[hindq_score]
            st.markdown(
                f'<div style="color:{_label_colors[hindq_score]};font-weight:bold;">'
                f'→ {hindquarter_pump}</div>', unsafe_allow_html=True)

        with col_p3:
            gait_score = st.slider(
                "歩様・踏み込みの流動性（重心移動効率）",
                min_value=1, max_value=5, key="sl_gait",
                help="1=硬い/ぎこちない　5=踏み込み深く滑らか",
            )
            gait_labels = {1: "硬い(ぎこちない)", 2: "やや硬い", 3: "普通",
                           4: "滑らか", 5: "踏み込み最高"}
            gait_fluidity = gait_labels[gait_score]
            st.markdown(
                f'<div style="color:{_label_colors[gait_score]};font-weight:bold;">'
                f'→ {gait_fluidity}</div>', unsafe_allow_html=True)

        # ── AI分析実行 ────────────────────────────────────
        st.markdown("---")
        if st.button("AI分析実行", type="primary", use_container_width=False):
            if not st.session_state.entries:
                with st.spinner("出走馬情報を取得中..."):
                    st.session_state.entries = scraper.fetch_entries(
                        selected_race["race_id"], venue=selected_race.get("venue", "")
                    )
            entries = st.session_state.entries

            if entries:
                with st.spinner("Gemini AIが科学的黄金比で分析中..."):
                    # JRA Ground Truthを取得（DB保存済みがあれば復元）
                    jra_gt = st.session_state.jra_track_data or \
                             get_jra_ground_truth(selected_race["race_id"])
                    # 出走取消馬を除外
                    scratched = jra_gt.get("scratched", [])
                    active_entries = [h for h in entries if h["name"] not in scratched]

                    if st.session_state.api_key:
                        result = gemini_client.analyze_race(
                            api_key=st.session_state.api_key,
                            race_name=selected_race["race_name"],
                            horses=active_entries,
                            track_condition=track_condition,
                            weather=weather,
                            temperature=temperature,
                            cushion_value=cushion_value,
                            paddock_notes=paddock_notes,
                            coat_gloss=coat_gloss,
                            hindquarter_pump=hindquarter_pump,
                            weights=load_weights(),
                            jra_track_data=jra_gt,
                        )
                    else:
                        result = _mock_analysis(entries, selected_race["race_name"])

                    save_prediction(
                        race_id=selected_race["race_id"],
                        prediction={
                            "race_name": selected_race["race_name"],
                            "grade": selected_race["grade"],
                            "horses": result["horses"],
                            "gemini_comment": result["comment"],
                        },
                    )
                    st.session_state.prediction_result = result
                    st.session_state.selected_race = selected_race
                    st.success("予想が完了しました！")

        # ── 予想結果カード ────────────────────────────────
        if st.session_state.prediction_result:
            result = st.session_state.prediction_result
            horses = result.get("horses", [])[:3]

            st.subheader("予想結果")

            CARD_CONF = {
                1: {"border": "#FFD700", "bg": "#1f1a00", "icon": "1st", "badge_bg": "#FFD700", "badge_fg": "#000"},
                2: {"border": "#C0C0C0", "bg": "#1a1a1a", "icon": "2nd", "badge_bg": "#C0C0C0", "badge_fg": "#000"},
                3: {"border": "#CD7F32", "bg": "#1a0f00", "icon": "3rd", "badge_bg": "#CD7F32", "badge_fg": "#000"},
            }

            # 馬名→馬番マップ（entriesから引く）
            _num_map = {h["name"]: h.get("number", "") for h in st.session_state.entries}

            if not horses:
                st.warning("予想馬のデータを取得できませんでした。")
            else:
                cols = st.columns(len(horses))
                for i, horse in enumerate(horses):
                    rank = horse.get("rank", i + 1)
                    conf = horse.get("confidence", 0)
                    ev_gap = str(horse.get("ev_gap", ""))
                    ev_positive = ev_gap.startswith("+")
                    ev_color = "#00ff88" if ev_positive else ("#ff6b6b" if ev_gap.startswith("-") else "#aaa")
                    ev_label = f"EV乖離: {ev_gap}" if ev_gap else ""
                    cc = CARD_CONF.get(rank, CARD_CONF[3])
                    conf_pct = min(conf, 100)
                    horse_name = horse.get("name", "?")
                    horse_num = _num_map.get(horse_name, "")
                    num_display = f"<span style='font-size:1.1em;font-weight:900;color:{cc['border']};margin-right:4px;'>#{horse_num}</span>" if horse_num else ""

                    with cols[i]:
                        st.markdown(
                            f"""
<div style="
    border: 2px solid {cc['border']};
    border-radius: 14px;
    padding: 20px 18px 16px;
    background: {cc['bg']};
    margin-bottom: 12px;
    box-shadow: 0 0 12px {cc['border']}44;
">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <span style="
      background:{cc['badge_bg']};color:{cc['badge_fg']};
      font-weight:900;font-size:1.0em;padding:3px 10px;
      border-radius:20px;letter-spacing:1px;
    ">{cc['icon']}</span>
    <span style="color:{cc['border']};font-size:0.8em;font-weight:600;">スコア {conf}</span>
  </div>
  <div style="font-size:1.5em;font-weight:900;margin-bottom:6px;">
    {num_display}{horse_name}
  </div>
  <div style="margin-bottom:10px;">
    <div style="background:#333;border-radius:4px;height:6px;width:100%;">
      <div style="background:{cc['border']};border-radius:4px;height:6px;width:{conf_pct}%;"></div>
    </div>
  </div>
  <div style="color:{ev_color};font-size:1.0em;font-weight:700;margin-bottom:8px;">
    {ev_label if ev_label else '&nbsp;'}
  </div>
  <div style="color:#ccc;font-size:0.88em;line-height:1.5;margin-bottom:10px;">
    {horse.get('reason', '')}
  </div>
  <div style="
    background:#0a2a1a;border:1px solid #00ff8844;
    border-radius:6px;padding:6px 10px;
    color:#00ff88;font-weight:700;font-size:0.9em;
  ">推奨: {horse.get('bet', '')}</div>
</div>
""",
                            unsafe_allow_html=True,
                        )

            if result.get("comment"):
                with st.expander("AI総合コメント（運動生理学視点）"):
                    st.write(result["comment"])

            # Entry list with bio columns
            if st.session_state.entries:
                st.subheader("出走馬一覧（バイオメカニカルデータ統合）")
                pad_reps = st.session_state.paddock_reports

                for h in st.session_state.entries:
                    name = h.get("name", "")

                    # 調教物理解析
                    tp = h.get("training_physics") or {}
                    h["_acc_rate"] = f"{tp['acceleration_rate']:+.3f}" if tp.get("acceleration_rate") else "－"
                    h["_cardio"]   = f"{tp['cardio_index']:+.3f}"       if tp.get("cardio_index")    else "－"

                    # パドックNLPスコア（自動取得分 or 手入力分）
                    rep = pad_reps.get(name, {})
                    gs  = rep.get("gemini_scores") or {}
                    ps  = rep.get("scores") or h.get("paddock_scores") or {}

                    def _fmt_score(v, digits=2):
                        return f"{v:+.{digits}f}" if v else "－"

                    h["_tomu"]  = str(gs.get("hindquarter_tension", "")) or _fmt_score(ps.get("hindquarter_power"))
                    h["_coat"]  = str(gs.get("coat_gloss", ""))          or _fmt_score(ps.get("vascularity_index"))
                    h["_kiiai"] = str(gs.get("mental_energy", ""))        or "－"
                    h["_gait"]  = _fmt_score(ps.get("gait_fluidity"))

                    # KeibaLab 当日馬体重
                    kl_wt = rep.get("weight_kg")
                    kl_ch = rep.get("weight_change")
                    if kl_wt:
                        sign = "+" if kl_ch and kl_ch > 0 else ""
                        h["_today_weight"] = f"{kl_wt}kg({sign}{kl_ch})" if kl_ch is not None else f"{kl_wt}kg"
                    else:
                        h["_today_weight"] = ""

                    # パドックコメント要約
                    h["_pad_comment"] = rep.get("text", "")[:40] if rep.get("text") else ""

                    # ベスト体重
                    bwa = h.get("best_weight_analysis") or {}
                    h["_best_wt"] = (
                        f"{bwa['best_weight']}kg({bwa.get('deviation',0):+d}) {bwa.get('classification','')}"
                        if bwa.get("best_weight") else ""
                    )

                    # 個体弱点
                    tpro = h.get("transport_profile") or {}
                    h["_weakness"] = (tpro.get("patterns") or [""])[0]

                priority_cols = [
                    ("number",         "馬番"),
                    ("name",           "馬名"),
                    ("jockey",         "騎手"),
                    ("weight",         "斤量"),
                    ("odds",           "オッズ"),
                    ("_today_weight",  "当日体重(KeibaLab)"),
                    ("stable",         "所属"),
                    ("ritto",          "外厩"),
                    ("transport_stress","輸送"),
                    ("bloodline",      "血統"),
                    ("recent_form",    "近走"),
                    ("weight_trend",   "馬体重推移"),
                    ("jockey_win_rate","騎手勝率"),
                    ("jockey_g1_wins", "G1勝数"),
                    ("trainer_win_rate","調教師勝率"),
                    ("training_eval",  "調教評価"),
                    ("_acc_rate",      "加速率"),
                    ("_cardio",        "心肺指標"),
                    ("_tomu",          "トモ(1-5)"),
                    ("_coat",          "毛艶(1-5)"),
                    ("_kiiai",         "気合(1-5)"),
                    ("_gait",          "歩様流動"),
                    ("_pad_comment",   "パドック短評"),
                    ("_best_wt",       "ベスト体重"),
                    ("_weakness",      "個体弱点"),
                ]
                df = pd.DataFrame(st.session_state.entries)
                show = [(c, l) for c, l in priority_cols if c in df.columns]
                df_show = df[[c for c, _ in show]].rename(columns={c: l for c, l in show})
                st.dataframe(df_show, use_container_width=True, hide_index=True)
                st.caption("※ トモ/毛艶/気合はGemini採点(1-5)。パドック情報自動取得後に更新されます。")


# ═════════════════════════════════════════════
# Tab 2: Results & PDCA
# ═════════════════════════════════════════════

with tab2:
    st.header("結果入力 & PDCA自己進化")

    # ── レース一覧未取得の場合はTab2内でも取得できる ──
    if not st.session_state.races:
        st.info("レース一覧が未取得です。以下から取得するか、「予想」タブでも取得できます。")
        _t2c1, _t2c2, _t2c3 = st.columns([2, 1, 1])
        with _t2c1:
            _t2_date = st.date_input("取得日付", value=date.today(), key="tab2_race_date")
        with _t2c2:
            if st.button("今日のGレース取得", key="tab2_fetch_today", use_container_width=True):
                with st.spinner("取得中..."):
                    _fetched = scraper.fetch_race_list(_t2_date)
                st.session_state.races = _fetched
                st.rerun()
        with _t2c3:
            if st.button("直近4週の過去レース取得", key="tab2_fetch_past", use_container_width=True):
                with st.spinner("過去Gレース取得中..."):
                    _fetched = scraper.fetch_past_g_races(4)
                st.session_state.races = _fetched
                st.rerun()

    if st.session_state.races:
        race_options2 = {
            f"[{r['grade']}] {r['race_name']}": r for r in st.session_state.races
        }
        sel2 = st.selectbox("対象レース", options=list(race_options2.keys()), key="result_race_sel")
        race2 = race_options2[sel2]
        race_id2 = race2["race_id"]

        col_fetch, _ = st.columns([1, 2])
        with col_fetch:
            if st.button("スクレイピングで結果取得"):
                with st.spinner("結果を取得中..."):
                    fetched = scraper.fetch_result(race_id2)
                if fetched:
                    save_result(race_id2, {"race_name": race2["race_name"], **fetched})
                    st.success("結果を自動取得しました")
                else:
                    st.warning("結果を取得できませんでした。手動入力してください。")

        st.subheader("手動結果入力")
        with st.form("result_form"):
            st.caption("着順を入力（最低1着〜3着）")
            cols = st.columns(3)
            with cols[0]:
                w1 = st.text_input("1着 馬名")
                t1 = st.text_input("1着 タイム", placeholder="1:33.5")
            with cols[1]:
                w2 = st.text_input("2着 馬名")
                t2 = st.text_input("2着 タイム", placeholder="1:33.8")
            with cols[2]:
                w3 = st.text_input("3着 馬名")
                t3 = st.text_input("3着 タイム", placeholder="1:34.0")

            st.caption("配当")
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                payout_win = st.number_input("単勝 (円)", min_value=0, value=0, step=10)
            with col_p2:
                payout_place = st.number_input("複勝1着 (円)", min_value=0, value=0, step=10)
            with col_p3:
                payout_exacta = st.number_input("馬連 (円)", min_value=0, value=0, step=10)

            submitted = st.form_submit_button("結果を保存", type="primary")
            if submitted and w1:
                finishing_order = [
                    {"rank": r, "name": n, "time": t}
                    for r, n, t in [(1, w1, t1), (2, w2, t2), (3, w3, t3)]
                    if n
                ]
                payouts = {}
                if payout_win:
                    payouts["単勝"] = payout_win
                if payout_place:
                    payouts["複勝"] = payout_place
                if payout_exacta:
                    payouts["馬連"] = payout_exacta

                save_result(race_id2, {
                    "race_name": race2["race_name"],
                    "finishing_order": finishing_order,
                    "payouts": payouts,
                })

                # Update horse profile DB for best-weight / transport tracking
                venue2 = race2.get("venue", "")
                weather_temp2 = st.session_state.weather_data.get("temperature", "")
                stable_map = {h["name"]: h for h in st.session_state.get("entries", [])}
                for fo in finishing_order:
                    hdata = stable_map.get(fo["name"], {})
                    hid = hdata.get("horse_id", "")
                    if not hid:
                        continue
                    wm = _re.search(r"(\d{3,4})kg", hdata.get("weight_trend", ""))
                    w_kg = int(wm.group(1)) if wm else 0
                    km_match = _re.search(r"(\d+)km", hdata.get("transport_stress", ""))
                    km = int(km_match.group(1)) if km_match else 0
                    if w_kg:
                        _rec_date = race2.get("race_date", selected_date.isoformat())
                        upsert_best_weight_record(
                            horse_id=hid,
                            race_date=_rec_date,
                            rank=fo["rank"],
                            weight_kg=w_kg,
                            venue=venue2,
                            weather_temp=weather_temp2,
                            transport_km=km,
                        )
                st.success("結果を保存しました")
                # 予想データがあれば自動的にPDCA分析を実行するフラグをセット
                if get_prediction(race_id2):
                    st.session_state.auto_pdca_race = race_id2
                st.rerun()

        # ── PDCA ────────────────────────────────────────
        st.divider()
        st.subheader("PDCA自己進化実行")

        pred_exists = get_prediction(race_id2)
        result_exists = get_result(race_id2)

        if not pred_exists:
            st.warning("このレースの予想データがありません。")
            if st.button("AIで遡及予想を生成してPDCAを実行", key="gen_retro_pred", type="primary"):
                with st.spinner("出走馬データ取得中..."):
                    _entries = scraper.fetch_entries(race_id2, venue=race2.get("venue", ""))
                if not _entries:
                    st.error("出走馬データを取得できませんでした。")
                else:
                    with st.spinner("Geminiで遡及予想を生成中..."):
                        _retro = gemini_client.analyze_race(
                            api_key=st.session_state.api_key,
                            race_name=race2["race_name"],
                            horses=_entries,
                            track_condition=race2.get("track_condition", "良"),
                            weather=race2.get("weather", ""),
                            weights=load_weights(),
                        )
                    save_prediction(race_id2, {
                        "race_name": race2["race_name"],
                        "grade": race2.get("grade", ""),
                        "horses": _retro.get("horses", []),
                        "gemini_comment": _retro.get("comment", ""),
                        "retroactive": True,
                    })
                    st.success("遡及予想を生成しました。PDCA分析を続けます...")
                    st.session_state.auto_pdca_race = race_id2
                    st.rerun()
        elif not result_exists:
            st.warning("結果データがありません。上記フォームで入力してください。")
        else:
            # 結果保存後の自動実行 or 手動ボタン
            auto_run = st.session_state.auto_pdca_race == race_id2
            if auto_run:
                st.session_state.auto_pdca_race = None  # フラグをクリア

            if auto_run or st.button("PDCA分析 & 黄金比重み自動更新", type="primary"):
                with st.spinner("Geminiが外れ要因を分析・重みを自動調整中..."):
                    pdca_result = pdca_engine.compare_and_evolve(
                        race_id=race_id2,
                        api_key=st.session_state.api_key,
                    )

                if "error" in pdca_result:
                    st.error(pdca_result["error"])
                else:
                    col_hit1, col_hit3, col_bias = st.columns(3)
                    with col_hit1:
                        if pdca_result["hit_1st"]:
                            st.success(f"1着的中！ ({pdca_result['top_pick']})")
                        else:
                            st.error(f"1着外れ (予想: {pdca_result['top_pick']})")
                    with col_hit3:
                        if pdca_result["hit_top3"]:
                            st.success("3着内的中！")
                        else:
                            st.warning("3着内も外れ")
                    with col_bias:
                        bias = pdca_result.get("odds_bias_flag", {})
                        if bias.get("missed_ev"):
                            st.error(f"EV機会損失あり (実際の1着オッズ: {bias.get('winner_odds')}倍)")
                        elif bias.get("biased"):
                            st.warning("人気馬優先バイアスを検出")
                        else:
                            st.success("オッズバイアスなし")

                    if pdca_result.get("odds_bias_audit"):
                        st.info(f"オッズバイアス監査: {pdca_result['odds_bias_audit']}")

                    # ── ミスカテゴリ分析 ────────────────────
                    miss_cats = pdca_result.get("miss_categories", {})
                    if miss_cats:
                        st.subheader("要因別ミスカテゴリ分析")
                        _cat_color = {
                            "過大評価": "#ff6b6b",
                            "過小評価": "#ffd700",
                            "適切":     "#00ff88",
                        }
                        mc_cols = st.columns(4)
                        for i, (k, label) in enumerate(WEIGHT_LABELS.items()):
                            verdict = miss_cats.get(k, "－")
                            color = _cat_color.get(verdict, "#aaa")
                            old_w = pdca_result["old_weights"].get(k, 0)
                            new_w = pdca_result["new_weights"].get(k, 0)
                            delta = new_w - old_w
                            delta_str = f"{delta:+.1%}" if delta != 0 else "変更なし"
                            delta_color = "#ff6b6b" if delta < 0 else "#00ff88" if delta > 0 else "#aaa"
                            with mc_cols[i]:
                                st.markdown(
                                    f'<div style="border:1px solid {color};border-radius:8px;padding:10px;text-align:center;">'
                                    f'<div style="font-size:0.85em;color:#ccc;">{label}</div>'
                                    f'<div style="font-size:1.1em;font-weight:bold;color:{color};">{verdict}</div>'
                                    f'<div style="font-size:0.8em;color:{delta_color};margin-top:4px;">'
                                    f'重み: {old_w:.0%} → {new_w:.0%} ({delta_str})</div>'
                                    f'</div>',
                                    unsafe_allow_html=True,
                                )

                    if pdca_result.get("reflection"):
                        st.subheader("AI反省レポート（外れ要因の詳細分析）")
                        st.write(pdca_result["reflection"])

                    if pdca_result.get("key_lessons"):
                        st.subheader("今回の教訓")
                        for lesson in pdca_result["key_lessons"]:
                            st.markdown(f"- {lesson}")

                    if pdca_result.get("weight_reasoning"):
                        st.info(f"重み変更の根拠: {pdca_result['weight_reasoning']}")
                    st.caption("※ 重みの変化は最大±5%/レースに制限されています（急激な過学習防止）")


# ═════════════════════════════════════════════
# Tab 3: Dashboard
# ═════════════════════════════════════════════

with tab3:
    st.header("ダッシュボード")

    stats = compute_stats()

    # ── KPI ──────────────────────────────────────────────
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("総分析レース数", stats["total_races"])
    with kpi2:
        st.metric("1着的中率", f"{stats['rate_1st']:.1%}", f"{stats['hit_1st']}回")
    with kpi3:
        st.metric("3着内的中率", f"{stats['rate_top3']:.1%}", f"{stats['hit_top3']}回")
    with kpi4:
        weights_now = load_weights()
        bio_w = weights_now.get("bio_condition", 0.40)
        st.metric("生体ウェイト", f"{bio_w:.0%}", help="科学的黄金比の現在値 (目標40%)")
    with kpi5:
        env_w = weights_now.get("environment", 0.30)
        st.metric("環境ウェイト", f"{env_w:.0%}", help="科学的黄金比の現在値 (目標30%)")

    st.divider()

    history = pdca_engine.get_hit_rate_history()
    if history:
        df_hist = pd.DataFrame(history)
        df_hist["累計1着的中率"] = df_hist["hit_1st"].expanding().mean()
        df_hist["累計3着内的中率"] = df_hist["hit_top3"].expanding().mean()

        # ── 2カラム: 勝率推移 | 要因別寄与度レーダー ──────
        chart_col, radar_col = st.columns([3, 2])

        with chart_col:
            st.subheader("予想精度推移")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_hist) + 1)),
                y=df_hist["累計1着的中率"],
                mode="lines+markers",
                name="1着的中率",
                line=dict(color="#FFD700", width=2),
                marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, len(df_hist) + 1)),
                y=df_hist["累計3着内的中率"],
                mode="lines+markers",
                name="3着内的中率",
                line=dict(color="#7fdbff", width=2),
                marker=dict(size=6),
            ))
            fig.add_hrect(y0=0.25, y1=0.35, fillcolor="#00ff8822", line_width=0,
                          annotation_text="1着25-35%ゾーン", annotation_position="top left")
            fig.update_layout(
                xaxis_title="レース数",
                yaxis_title="的中率",
                yaxis=dict(tickformat=".0%"),
                template="plotly_dark",
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)

        with radar_col:
            st.subheader("要因別寄与度")
            w = load_weights()
            radar_labels = [WEIGHT_LABELS.get(k, k) for k in w]
            radar_values = list(w.values())
            # Golden ratio targets
            golden = [0.40, 0.30, 0.20, 0.10]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=golden + [golden[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(255,215,0,0.08)",
                line=dict(color="#FFD70066", dash="dot"),
                name="黄金比目標",
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=radar_values + [radar_values[0]],
                theta=radar_labels + [radar_labels[0]],
                fill="toself",
                fillcolor="rgba(127,219,255,0.18)",
                line=dict(color="#7fdbff", width=2),
                name="現在値",
            ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 0.5],
                                           tickformat=".0%")),
                template="plotly_dark",
                height=320,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15),
                margin=dict(t=20, b=60),
            )
            st.plotly_chart(fig_r, use_container_width=True)

        # ── バイアス指標 ──────────────────────────────────
        if "odds_biased" in df_hist.columns:
            bias_rate = df_hist["odds_biased"].mean()
            ev_miss = df_hist["missed_ev"].mean() if "missed_ev" in df_hist.columns else 0
            bc1, bc2 = st.columns(2)
            with bc1:
                color = "normal" if bias_rate < 0.4 else "inverse"
                st.metric("人気偏重率", f"{bias_rate:.0%}",
                          help="予想1位が最低オッズ馬だった割合。40%未満が目標。",
                          delta=f"{'良好' if bias_rate < 0.4 else '要改善'}",
                          delta_color=color)
            with bc2:
                color2 = "normal" if ev_miss < 0.3 else "inverse"
                st.metric("EV機会損失率", f"{ev_miss:.0%}",
                          help="実際の勝馬が予想馬より1.5倍以上高オッズだった割合。30%未満が目標。",
                          delta=f"{'良好' if ev_miss < 0.3 else '要改善'}",
                          delta_color=color2)

        st.divider()

        # ── 直近履歴テーブル ─────────────────────────────
        show_cols = ["race_name", "hit_1st", "hit_top3", "confidence",
                     "ev_gap", "odds_biased", "missed_ev", "timestamp"]
        show_cols = [c for c in show_cols if c in df_hist.columns]
        rename_map = {
            "race_name": "レース名", "hit_1st": "1着的中", "hit_top3": "3着内的中",
            "confidence": "スコア", "ev_gap": "EV乖離", "odds_biased": "人気偏重",
            "missed_ev": "EV機会損失", "timestamp": "分析日時",
        }
        st.subheader("分析履歴")
        st.dataframe(
            df_hist[show_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )

        # ── 重み推移グラフ（PDCAログ） ──────────────────
        weight_hist = get_weight_history()
        if len(weight_hist) >= 2:
            st.subheader("重み自動調整の推移（PDCA履歴）")
            wh_labels = [h["race_name"] for h in weight_hist]
            wh_colors = {
                "bio_condition": "#ff7f7f",
                "environment":   "#7fdbff",
                "human_skill":   "#00ff88",
                "background":    "#ffd700",
            }
            fig_w = go.Figure()
            for key, label in WEIGHT_LABELS.items():
                vals = [h["new_weights"].get(key, 0) for h in weight_hist]
                fig_w.add_trace(go.Scatter(
                    x=wh_labels,
                    y=vals,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=wh_colors.get(key, "#aaa"), width=2),
                    marker=dict(size=7),
                ))
            # 黄金比の目標ライン
            golden_vals = {"bio_condition": 0.40, "environment": 0.30,
                           "human_skill": 0.20, "background": 0.10}
            for key, label in WEIGHT_LABELS.items():
                fig_w.add_hline(
                    y=golden_vals[key],
                    line=dict(color=wh_colors.get(key, "#aaa"), dash="dot", width=1),
                    opacity=0.4,
                )
            fig_w.update_layout(
                xaxis_title="レース",
                yaxis_title="重み",
                yaxis=dict(tickformat=".0%", range=[0, 0.6]),
                template="plotly_dark",
                height=300,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig_w, use_container_width=True)
            st.caption("点線 = 科学的黄金比目標。実線 = PDCAによる自動調整後の値。")

        # ── 外れパターン ─────────────────────────────────
        trend = pdca_engine.get_trend_analysis()
        misses = trend.get("recent_misses", [])
        if misses:
            st.subheader("最近の外れパターン")
            df_miss = pd.DataFrame(misses)
            df_miss.columns = ["レース名", "予想本命", "実際の1着"]
            st.dataframe(df_miss, use_container_width=True, hide_index=True)

    else:
        st.info("予想データがありません。「予想」タブでレース分析を実行してください。")

        # Show the golden ratio radar even with no data
        st.subheader("科学的黄金比フレームワーク")
        w = load_weights()
        radar_labels = [WEIGHT_LABELS.get(k, k) for k in w]
        radar_values = list(w.values())
        fig_init = go.Figure(go.Scatterpolar(
            r=radar_values + [radar_values[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            fillcolor="rgba(127,219,255,0.2)",
            line=dict(color="#7fdbff", width=2),
            name="初期黄金比",
        ))
        fig_init.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 0.5], tickformat=".0%")),
            template="plotly_dark",
            height=350,
        )
        st.plotly_chart(fig_init, use_container_width=True)


# ═════════════════════════════════════════════
# Tab 4: 過去Gレース学習
# ═════════════════════════════════════════════

with tab4:
    st.header("過去Gレース一括学習")
    st.caption(
        "直近の終了済みG1/G2/G3レースを自動取得し、AIが「当時の予想」を再構築して "
        "実際の結果と照合します。PDCA自己進化ループを複数レース分まとめて実行し、"
        "黄金比パラメーターを実績ベースで調整します。"
    )

    if not st.session_state.api_key:
        st.warning("Gemini API キーが設定されていません。サイドバーで設定してください。")
    else:
        # ── 過去レース取得 ───────────────────────────────────
        col_fetch_past, col_n_weeks = st.columns([2, 1])
        with col_n_weeks:
            n_weeks = st.number_input("取得週数", min_value=1, max_value=8, value=4, step=1,
                                      help="最大8週前までのGレースを対象にします")
        with col_fetch_past:
            if st.button("直近の終了済みGレースを取得", type="primary", use_container_width=True):
                with st.spinner(f"直近{n_weeks}週のGレース一覧を取得中..."):
                    past_races = scraper.fetch_past_g_races(int(n_weeks))
                st.session_state["past_g_races"] = past_races
                if past_races:
                    st.success(f"{len(past_races)}レースを取得しました")
                else:
                    st.warning("Gレースが見つかりませんでした（netkeiba接続を確認してください）")

        past_races = st.session_state.get("past_g_races", [])

        if past_races:
            st.divider()

            # ── レース選択 ──────────────────────────────────
            st.subheader("学習対象レースを選択")
            race_options = {
                f"[{r['grade']}] {r['race_name']} ({r.get('race_date','?')} {r.get('venue','')})": r
                for r in past_races
            }
            selected_labels = st.multiselect(
                "学習するレースを選択（複数可）",
                options=list(race_options.keys()),
                default=list(race_options.keys()),
                help="デフォルトで全件選択。不要なレースはチェックを外してください。"
            )
            selected_past = [race_options[lbl] for lbl in selected_labels]

            if not selected_past:
                st.info("学習対象レースを1つ以上選択してください。")
            else:
                st.info(
                    f"**{len(selected_past)}レース**を対象に学習します。\n\n"
                    "処理内容: ① 出走馬データ取得 → ② 結果取得 → "
                    "③ AI遡及予想生成 → ④ PDCA重み自動調整"
                )

                if st.button(
                    f"一括学習を実行（{len(selected_past)}レース）",
                    type="primary",
                    use_container_width=True,
                ):
                    progress = st.progress(0)
                    status = st.empty()
                    log_area = st.empty()
                    log_lines: list[str] = []

                    learned = 0
                    skipped = 0
                    errors = 0

                    for idx, race in enumerate(selected_past):
                        rid = race["race_id"]
                        rname = race["race_name"]
                        progress.progress((idx) / len(selected_past))
                        status.markdown(f"**[{idx+1}/{len(selected_past)}]** {rname} を処理中...")

                        try:
                            # ① 結果取得
                            result_data = get_result(rid)
                            if not result_data:
                                fetched = scraper.fetch_result(rid)
                                if not fetched:
                                    log_lines.append(f"⚠ {rname}: 結果を取得できませんでした（スキップ）")
                                    skipped += 1
                                    log_area.markdown("\n".join(log_lines[-10:]))
                                    continue
                                result_data = {"race_name": rname, **fetched}
                                save_result(rid, result_data)
                                log_lines.append(f"✓ {rname}: 結果を取得・保存")
                            else:
                                log_lines.append(f"○ {rname}: 結果は保存済み")

                            # ② 予想データが未存在なら遡及予想を生成
                            pred_data = get_prediction(rid)
                            if not pred_data:
                                # 出走馬データを取得（過去レースでも利用可能）
                                entries = scraper.fetch_entries(
                                    rid, venue=race.get("venue", "")
                                )
                                if not entries:
                                    log_lines.append(f"⚠ {rname}: 出走馬データを取得できませんでした（スキップ）")
                                    skipped += 1
                                    log_area.markdown("\n".join(log_lines[-10:]))
                                    continue

                                # 遡及AI予想（当時のデータで再構築）
                                retro_result = gemini_client.analyze_race(
                                    api_key=st.session_state.api_key,
                                    race_name=rname,
                                    horses=entries,
                                    track_condition=race.get("track_condition", "良"),
                                    weather=race.get("weather", ""),
                                    weights=load_weights(),
                                )
                                save_prediction(rid, {
                                    "race_name": rname,
                                    "grade": race.get("grade", ""),
                                    "horses": retro_result.get("horses", []),
                                    "gemini_comment": retro_result.get("comment", ""),
                                    "retroactive": True,  # 遡及予想フラグ
                                })
                                log_lines.append(f"✓ {rname}: 遡及予想を生成・保存")
                            else:
                                log_lines.append(f"○ {rname}: 予想は保存済み")

                            # ③ PDCA実行（重み更新）
                            pdca_res = pdca_engine.compare_and_evolve(
                                race_id=rid,
                                api_key=st.session_state.api_key,
                            )
                            if "error" in pdca_res:
                                log_lines.append(f"✗ {rname}: PDCA エラー - {pdca_res['error']}")
                                errors += 1
                            else:
                                hit = "1着的中" if pdca_res["hit_1st"] else ("3着内" if pdca_res["hit_top3"] else "外れ")
                                cats = pdca_res.get("miss_categories", {})
                                cat_str = " / ".join(
                                    f"{WEIGHT_LABELS.get(k,k)}:{v}"
                                    for k, v in cats.items() if v and v != "適切"
                                )
                                log_lines.append(
                                    f"✓ {rname}: PDCA完了 [{hit}]"
                                    + (f" 要調整: {cat_str}" if cat_str else " 全要因適切")
                                )
                                learned += 1

                        except Exception as e:
                            log_lines.append(f"✗ {rname}: 予期しないエラー - {e}")
                            errors += 1

                        log_area.markdown("\n".join(log_lines[-10:]))

                    progress.progress(1.0)
                    status.markdown("**処理完了！**")

                    # ── 学習結果サマリー ─────────────────────
                    st.divider()
                    st.subheader("学習結果サマリー")
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.metric("PDCA完了", f"{learned}レース")
                    with s2:
                        st.metric("スキップ", f"{skipped}レース",
                                  help="データ取得不可のレース")
                    with s3:
                        st.metric("エラー", f"{errors}レース")

                    # 更新後の重みを表示
                    new_w = load_weights()
                    st.subheader("学習後の最新重み（黄金比）")
                    w_cols = st.columns(4)
                    for i, (k, label) in enumerate(WEIGHT_LABELS.items()):
                        with w_cols[i]:
                            default = {"bio_condition": 0.40, "environment": 0.30,
                                       "human_skill": 0.20, "background": 0.10}
                            delta = new_w.get(k, 0) - default[k]
                            st.metric(label, f"{new_w.get(k, 0):.1%}",
                                      delta=f"{delta:+.1%}",
                                      delta_color="normal" if abs(delta) < 0.05 else "inverse")
                    st.caption("※ 黄金比初期値（生体40% / 環境30% / 人間20% / 背景10%）との差分を表示")

                    # 全ログ表示
                    with st.expander("詳細ログを確認"):
                        st.text("\n".join(log_lines))


