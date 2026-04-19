"""Fact-driven theoretical prediction Streamlit app.

Run:
    streamlit run app_facts.py

Flow:
    1. Inputs  — race date, venue, race selector, source toggles
    2. Button  — 最新情報を収集 (runs fact_pipeline.run)
    3. Progress cards — one per collector (JRA, Yahoo, KeibaLab, news, merge, scoring)
    4. Quality display — coverage, consensus count, trust grade A/B/C/D
    5. Prediction output — top-3 with P1/P2, per-horse facts panel
"""

from __future__ import annotations

from datetime import date

import streamlit as st
import pandas as pd

import scraper
import fact_pipeline as fp


st.set_page_config(page_title="理論予想 — Facts-First", page_icon="📑", layout="wide")

st.title("📑 理論予想 — Facts-First Collection")
st.caption("JRA 公式ファクトを最優先、観察情報を補助に、予想・印・買い目は排除")


# ── Sidebar: source controls ─────────────────────────

with st.sidebar:
    st.header("ソース設定")
    use_detail = st.toggle(
        "詳細ソースを使用 (Yahoo + KeibaLab)",
        value=True,
        help="オフにすると JRA 相当情報のみで予想します。",
    )
    use_news = st.toggle(
        "ニュース/コラムを補助情報に使用",
        value=True,
        help="Yahoo で発見したリンク先の記事本文から観察ファクトを抽出します。",
    )
    st.divider()
    st.caption("""
**Source Trust Policy**

| Tier | Source | Base conf. |
|---|---|---|
| 1 | JRA official | 1.00 |
| 1 | netkeiba shutuba | 0.90 |
| 2 | Yahoo!競馬 (hub only) | — |
| 3 | KeibaLab | 0.70 |
| 3 | Sports news | 0.50 |

**Consensus bonus**
- 2 sources: +0.20
- 3+ sources: +0.40

**Excluded forever**: 印, 本命, 買い目, 推奨, 人気予想, popularity-derived opinions
""")


# ── Top: race selection ──────────────────────────────

col1, col2, col3, col4 = st.columns([1, 1, 2, 1])

with col1:
    race_date = st.date_input("レース日", value=date.today())

if "race_list" not in st.session_state:
    st.session_state.race_list = []

with col2:
    if st.button("🔄 レース一覧を取得", use_container_width=True):
        with st.spinner(f"{race_date} のレースを取得中..."):
            try:
                races = scraper.fetch_race_list(race_date) or []
                st.session_state.race_list = races
                if races:
                    st.success(f"{len(races)} レースを取得")
                else:
                    st.warning("レースが見つかりませんでした")
            except Exception as e:
                st.error(f"取得失敗: {e}")

venues = sorted({r.get("venue", "") for r in st.session_state.race_list if r.get("venue")})
with col3:
    chosen_venue = st.selectbox("開催場", ["(すべて)"] + venues, index=0)

filtered_races = [
    r for r in st.session_state.race_list
    if chosen_venue == "(すべて)" or r.get("venue") == chosen_venue
]
race_labels = [
    f"{r.get('race_name', '?')} ({r.get('venue', '?')}/{r.get('time', '?')})"
    for r in filtered_races
]

with col4:
    st.write("")
    st.write("")
    race_idx = st.selectbox(
        "レース選択",
        options=list(range(len(race_labels))),
        format_func=lambda i: race_labels[i] if race_labels and i < len(race_labels) else "—",
        disabled=not race_labels,
    ) if race_labels else None


# ── Collect + Predict ────────────────────────────────

run_btn = st.button(
    "🎯 最新情報を収集 / 理論予想を実行",
    type="primary",
    use_container_width=True,
    disabled=not filtered_races,
)

if run_btn and filtered_races:
    race = filtered_races[race_idx or 0]
    race_id = race.get("race_id", "")
    venue = race.get("venue", "")
    race_name = race.get("race_name", "")

    # Progress cards live here
    prog_container = st.container()
    prog_msgs: list[str] = []

    def _prog(msg: str):
        prog_msgs.append(msg)
        with prog_container:
            st.info(f"⏳ {msg}")

    with st.spinner(f"{race_name} を処理中..."):
        result = fp.run(
            race_id=race_id, venue=venue, race_name=race_name,
            use_detail_sources=use_detail,
            use_news_supplements=use_news,
            progress_cb=_prog,
        )
    st.session_state.last_result = result


# ── Render last result ───────────────────────────────

if "last_result" in st.session_state:
    r = st.session_state.last_result
    st.divider()
    st.subheader(f"{r['race_name']} — 結果")

    # ── Quality / trust block ─────────────
    q1, q2, q3, q4 = st.columns(4)
    cov = r.get("coverage", {})
    q1.metric("JRA カバレッジ", f"{cov.get('jra', 0)*100:.0f}%")
    q2.metric("補助カバレッジ", f"{cov.get('supplemental', 0)*100:.0f}%")
    q3.metric("コンセンサス事実", cov.get("consensus_count", 0))
    grade = r.get("trust_grade", "?")
    grade_color = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}.get(grade, "⚪")
    q4.metric("信頼度グレード", f"{grade_color} {grade}")

    # ── Collection log (progress cards) ────
    st.markdown("### 📥 収集プログレス")
    log_cols = st.columns(min(len(r["collection_log"]), 5) or 1)
    for i, rec in enumerate(r["collection_log"]):
        col = log_cols[i % len(log_cols)]
        status = rec.get("status", "?")
        icon = {"ok": "✅", "partial": "🟡", "failed": "❌",
                "skipped": "⏭"}.get(status, "❓")
        n_facts = len(rec.get("facts", []))
        items = rec.get("items_seen", 0)
        elapsed = rec.get("elapsed_s", 0)
        with col:
            st.markdown(
                f"**{icon} {rec.get('source', '?')}**\n\n"
                f"facts: **{n_facts}**\n\n"
                f"items: {items}\n\n"
                f"elapsed: {elapsed}s"
            )
            if rec.get("error"):
                st.caption(f"⚠ {rec['error'][:90]}")

    # ── Override decision ─────────────────
    od = r.get("override_decision", {}) or {}
    if od.get("model_top") or od.get("odds_fav"):
        c_ov = st.columns([2, 3])
        with c_ov[0]:
            if od.get("allow"):
                st.success(f"✅ マーケットを上書き  \n{od.get('reason', '')}")
            else:
                st.warning(f"⏸ マーケット追従  \n{od.get('reason', '')}")
        with c_ov[1]:
            st.caption(
                f"Model Top: **{od.get('model_top', '?')}**  /  "
                f"Odds Favorite: **{od.get('odds_fav', '?')}**"
            )

    # ── Top-3 block ───────────────────────
    st.markdown("### 🥇 選抜 Top-3 セット")
    a, b, c = st.columns(3)
    a.metric("P1 (1着が含まれる確率)", f"{r['p1']*100:.1f}%")
    b.metric("P2 (1-2着が両方含まれる確率)", f"{r['p2']*100:.1f}%")
    c.metric("目的関数", f"{r.get('p1', 0)*0.5 + r.get('p2', 0)*0.5:.3f}")

    sel_rows = []
    for i, h in enumerate(r["selected_top3"], start=1):
        per_facts = r["per_horse_facts"].get(h["name"], [])
        per_score = r["per_horse_score"].get(h["name"], {})
        pos_fact_names = [f["type"] for f in per_facts if f["polarity"] > 0][:3]
        neg_fact_names = [f["type"] for f in per_facts if f["polarity"] < 0][:2]
        sel_rows.append({
            "順位": i, "馬名": h["name"],
            "オッズ": f'{h["odds"]:.1f}' if h["odds"] else "-",
            "勝率": f'{h["win_prob"]*100:.1f}%',
            "composite": f'{per_score.get("composite_condition", 0.5):.2f}',
            "fact #": per_score.get("n_facts", 0),
            "正": " ".join(pos_fact_names) or "—",
            "負": " ".join(neg_fact_names) or "—",
        })
    if sel_rows:
        st.dataframe(pd.DataFrame(sel_rows), use_container_width=True, hide_index=True)

    # ── Per-horse fact viewer ─────────────
    st.markdown("### 🔍 馬別ファクト詳細")
    horse_names = [h["name"] for h in r["ranked"]]
    picked = st.selectbox("馬を選択", horse_names)
    if picked:
        pfacts = r["per_horse_facts"].get(picked, [])
        per_score = r["per_horse_score"].get(picked, {})
        hdr_a, hdr_b, hdr_c = st.columns(3)
        hdr_a.metric("Composite", f"{per_score.get('composite_condition', 0.5):.3f}")
        hdr_b.metric("Fact count", per_score.get("n_facts", 0))
        hdr_c.metric("Positive - Negative", f"{per_score.get('n_positive', 0)} / {per_score.get('n_negative', 0)}")
        if pfacts:
            df = pd.DataFrame([
                {
                    "type": f["type"], "polarity": "+" if f["polarity"] > 0 else "-",
                    "conf": round(f["confidence"], 2),
                    "source": f["source"],
                    "raw": f["raw_text"],
                }
                for f in pfacts
            ])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.caption("この馬の観察ファクトはまだ取得できていません。")

    # ── Full ranking ──────────────────────
    st.markdown("### 📊 全頭ランキング")
    rows = [
        {"順位": i + 1, "馬名": h["name"],
         "オッズ": f'{h["odds"]:.1f}' if h["odds"] else "-",
         "勝率": f'{h["win_prob"]*100:.1f}%',
         "score": round(h["score"], 2)}
        for i, h in enumerate(r["ranked"])
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Scratched
    if r.get("scratched"):
        st.caption(f"取消・除外: {', '.join(r['scratched'])}")
