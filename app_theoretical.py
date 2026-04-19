"""One-click theoretical prediction Streamlit app.

Run:
    streamlit run app_theoretical.py

Provides:
  - race date / race selection dropdowns
  - 最新情報を収集 button (refreshes race list, bypasses cache)
  - 理論予想を実行 button (runs the full prediction pipeline)
  - ranked horse table with per-horse win probability
  - selected top-3 set with P1 / P2
  - per-horse feature reasons
  - feature coverage + scratch status

This is intentionally a separate entry point — the legacy app.py stays
untouched so production dashboards keep working.
"""

from __future__ import annotations

from datetime import date

import streamlit as st
import pandas as pd

import scraper
import probability_engine as pe
import prediction_pipeline as pp


st.set_page_config(page_title="理論予想 (One-click)", page_icon="🎯", layout="wide")

st.title("🎯 理論予想 — ワンクリック")
st.caption("勝率 / Top-3 セット理論 / 特徴量カバレッジを一体評価")


# ── Sidebar: configuration ───────────────────────────────

with st.sidebar:
    st.header("設定")
    cfg = pe.load_config()

    enrich = st.toggle("詳細情報(調教/騎手/パドック)", value=True,
                       help="オフにすると高速だがカバレッジが低下します")

    st.subheader("選択目的関数")
    alpha = st.slider("α  (P1: 勝ち馬を含む確率)", 0.0, 1.0,
                      float(cfg.get("alpha", pe.DEFAULT_ALPHA)), 0.05)
    beta = st.slider("β  (P2: 1-2着両取り確率)", 0.0, 1.0,
                     float(cfg.get("beta", pe.DEFAULT_BETA)), 0.05)
    if abs((alpha + beta) - 1.0) > 1e-6:
        st.caption(f"※ α+β = {alpha+beta:.2f} (正規化は内部で不要)")

    st.subheader("キャリブレーション")
    T = st.slider("softmax 温度 T", 2.0, 30.0,
                  float(cfg.get("temperature", pe.DEFAULT_TEMPERATURE)), 0.5,
                  help="小さいほど最上位に確率が集中")

    if st.button("💾 設定を保存"):
        pe.save_config({"temperature": T, "alpha": alpha, "beta": beta})
        st.success("保存しました")

    st.divider()
    st.caption("Powered by score_runner + Plackett–Luce")


# ── Top: race selection ──────────────────────────────────

col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    race_date = st.date_input("レース日", value=date.today())
with col2:
    if "race_list" not in st.session_state:
        st.session_state.race_list = []
    race_options = [f"{r.get('race_name','?')} ({r.get('race_id','')})"
                    for r in st.session_state.race_list]
    choice = st.selectbox("レース", race_options or ["(最新情報を収集してください)"],
                          index=0)
with col3:
    st.write("")
    st.write("")
    refresh = st.button("🔄 最新情報を収集", use_container_width=True)

if refresh:
    with st.spinner(f"{race_date} のレース一覧を取得中..."):
        try:
            races = scraper.fetch_race_list(race_date) or []
            st.session_state.race_list = races
            if races:
                st.success(f"{len(races)} レースを取得しました")
            else:
                st.warning("この日付のレースは見つかりませんでした")
        except Exception as e:
            st.error(f"取得に失敗しました: {e}")
    st.rerun()


# ── Main: run prediction ─────────────────────────────────

run = st.button("🎯 理論予想を実行", type="primary", use_container_width=True,
                disabled=not st.session_state.race_list)

if run and st.session_state.race_list:
    idx = race_options.index(choice) if choice in race_options else 0
    race = st.session_state.race_list[idx]
    race_id = race.get("race_id", "")
    venue = race.get("venue", "")

    log_box = st.empty()

    def _log(msg):
        log_box.info(f"⏳ {msg}")

    with st.spinner(f"{race.get('race_name','?')} を予想中..."):
        result = pp.predict_live(
            race_id=race_id, venue=venue, enrich=enrich,
            alpha=alpha, beta=beta, temperature=T,
            progress_cb=_log,
        )
    log_box.empty()

    st.session_state.last_result = result

if "last_result" in st.session_state:
    r = st.session_state.last_result
    st.divider()

    # ── Header ─────────────────────────────
    head = st.columns([3, 1, 1, 1, 1])
    head[0].subheader(f"{r.race_name} ({r.grade})")
    head[1].metric("出走予定", r.n_entries_raw)
    head[2].metric("取消・除外", r.n_scratched)
    head[3].metric("スコア対象", r.n_scored)
    head[4].metric("特徴量カバレッジ", f"{r.feature_coverage*100:.0f}%")

    if r.scratched:
        st.caption(f"取消・除外馬: {', '.join(r.scratched)}")
    else:
        st.caption("取消・除外馬なし")

    # ── Selected top-3 ────────────────────
    st.subheader("🥇 選抜 Top-3 セット")
    a, b, c = st.columns(3)
    a.metric("P1: 1着が含まれる確率", f"{r.p1*100:.1f}%")
    b.metric("P2: 1-2着が両方含まれる確率", f"{r.p2*100:.1f}%")
    c.metric("目的関数 α·P1+β·P2", f"{r.objective:.3f}")

    if r.selected_top3:
        sel_df = pd.DataFrame([
            {
                "馬名": h["name"],
                "オッズ": f'{h["odds"]:.1f}' if h["odds"] else "-",
                "勝率": f'{h["win_prob"]*100:.1f}%',
                "スコア": round(h["score"], 2),
                "主な理由": " / ".join(h.get("reasons", [])) or "—",
            }
            for h in r.selected_top3
        ])
        st.dataframe(sel_df, use_container_width=True, hide_index=True)

    # ── Override gate ────────────────────
    od = r.override_decision or {}
    if od:
        col = st.columns([2, 3])
        with col[0]:
            if od.get("allow"):
                st.success(f"✅ 市場オーバーライド許可\n\n{od.get('reason','')}")
            else:
                st.warning(f"⏸ 市場オーバーライド非許可\n\n{od.get('reason','')}")
        with col[1]:
            st.caption(f"モデルTop: **{od.get('model_top','?')}** "
                       f"/ オッズTop: **{od.get('odds_fav','?')}**")

    # ── Full ranking ─────────────────────
    st.subheader("📊 全頭ランキング")
    full_df = pd.DataFrame([
        {
            "順位": i + 1,
            "馬名": h["name"],
            "オッズ": f'{h["odds"]:.1f}' if h["odds"] else "-",
            "勝率": h["win_prob"],
            "勝率(%)": f'{h["win_prob"]*100:.1f}%',
            "スコア": round(h["score"], 2),
        }
        for i, h in enumerate(r.horses)
    ])
    st.dataframe(
        full_df.drop(columns=["勝率"]),
        use_container_width=True, hide_index=True,
    )
    st.bar_chart(full_df.set_index("馬名")["勝率"])

    if r.feature_coverage < 0.25:
        st.info("⚠️ 特徴量カバレッジが低いため、モデルは実質オッズを模倣しています。"
                "`詳細情報` をオンにするか、scraper キャッシュを補充してください。")
