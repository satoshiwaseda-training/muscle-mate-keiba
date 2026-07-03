"""単勝+ワイド 買い目レコメンドの Streamlit 描画 (v3.0, 2026-07-04).

app_live.py から `render(rec)` を 1 行呼ぶだけ。
bet_recommender.recommend_bets() の戻り値をそのまま受け取る。
streamlit は関数内 import (app_live 以外でも import できるよう)。
"""
from __future__ import annotations


_VERDICT_STYLE = {
    "投資推奨": ("✅", "#1b5e20", "#e8f5e9"),
    "見送り推奨": ("⚠️", "#b71c1c", "#fff3e0"),
}
_TAG_EMOJI = {"投資級": "✅", "準投資級": "🟢", "見送り": "⚠️", "参考": "➖"}


def _pct(v) -> str:
    try:
        return f"{float(v) * 100:.0f}%"
    except (TypeError, ValueError):
        return "-"


def _sig(v) -> str:
    try:
        return f"{float(v) * 100:+.0f}%"
    except (TypeError, ValueError):
        return "-"


def _legs_block(legs, indent="　") -> str:
    """改行区切りの買い目テキスト (markdown用)。"""
    lines = []
    for leg in legs or []:
        for ln in leg.get("lines", []):
            lines.append(f"{indent}**{leg.get('ticket','')}**: `{ln}`")
    return "  \n".join(lines)


def render(rec: dict) -> None:
    import streamlit as st
    import pandas as pd

    if not rec or not rec.get("applicable"):
        if rec and rec.get("note"):
            st.caption(rec["note"])
        return

    st.markdown("#### 🛒 単勝 + ワイド 買い目レコメンド")
    st.caption(
        f"G1/G2 バックテスト実値（finishing_order重複除去・ワイド着順ペア順実証済）"
        f" / G1 n=46（〜2026年6月） / G2 n=71　{rec.get('version','')}"
    )

    r = rec.get("recommendation") or {}
    if r:
        emoji, fg, bg = _VERDICT_STYLE.get(r.get("verdict", ""), ("➖", "#555", "#f5f5f5"))

        # ── ヘッドライン カード ──
        st.markdown(
            f"<div style='border-left:7px solid {fg};background:{bg};"
            f"padding:12px 16px;border-radius:8px;margin-bottom:8px'>"
            f"<div style='font-size:1.1em'><b>{emoji} {r.get('verdict','')}</b>"
            f"　{r.get('label','')}</div>"
            f"<div style='margin:6px 0'>"
            f"的中率 <b style='font-size:1.2em'>{_pct(r.get('hit_rate'))}</b>　"
            f"回収率 <b style='font-size:1.2em;color:{fg}'>{_sig(r.get('roi'))}</b>　"
            f"一発除外 {_sig(r.get('roi_ex2'))}　最長連敗 {r.get('max_streak','-')}"
            f"　投資 <b>{r.get('total_stake','-')}円</b>"
            f"　[{r.get('tag','')}]"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ── 具体的な買い目 ──
        for leg in r.get("legs", []):
            for ln in leg.get("lines", []):
                st.markdown(f"　**{leg.get('ticket','')}**：`{ln}`")

        if r.get("why"):
            st.caption("💡 " + r["why"])

    # ── 代替案テーブル ──
    alts = rec.get("alternatives") or []
    if alts:
        st.markdown("**目的別の代替案**")
        rows = []
        for a in alts:
            emoji = _TAG_EMOJI.get(a.get("tag", "参考"), "➖")
            buy = "  /  ".join(
                ln for leg in a.get("legs", []) for ln in leg.get("lines", [])
            )
            rows.append({
                "目的": a.get("purpose", ""),
                "判定": f"{emoji} {a.get('tag','')}",
                "構成": a.get("label", ""),
                "的中率": _pct(a.get("hit_rate")),
                "回収率": _sig(a.get("roi")),
                "一発除外": _sig(a.get("roi_ex2")),
                "投資額": f"{a.get('total_stake','-')}円",
                "買い目": buy,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── 注釈 ──
    note = rec.get("note", "")
    if note:
        if r.get("verdict") == "見送り推奨":
            st.warning(note)
        else:
            st.info(note)
