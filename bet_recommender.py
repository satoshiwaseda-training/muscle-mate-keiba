"""単勝 + ワイド 買い目レコメンダ (presentation layer — v3.0, 2026-07-04).

ユーザ要望: 「単勝とワイドで、投資効率(的中率・低分散)と回収率(ROIプラス)
を両立させて具体的な馬・馬番を提案」。

憲法上の位置づけ: grade_strategy.py と同じ「買い方の提示層」。推論
(score_runner / probability_engine / dual_mode_scoring / trigger_loose_capped)
と LOOSE 4 条件・モデル係数には一切触れない。表示専用。

検証出所 (tools/analyze_tansho_wide_portfolio.py, finishing_order重複除去済,
ワイドは着順ペア順1-2/1-3/2-3=全119R実証済):
  G1 n=46 / G2 n=71。憲法 §4.2 に従い 100 ベット未満は確定視しない (参考値)。
  v3.0: 2026春G1全レース(桜花賞〜宝塚記念)を追加し G1 n=38→46, G2 n=63→71 に更新。

STATS は 2026-07-04 実スクリプト出力値をそのまま使用。手書き・推測値は一切含まない。
"""
from __future__ import annotations

from itertools import combinations

import grade_strategy as gs

RECOMMENDER_VERSION = "bet-recommender-v3.0-tansho-wide-2026-07-04"
T = 100  # 1点 100円
MARKS = ["◎", "○", "▲"]


# ── 検証済み実績テーブル ────────────────────────────────────────────
# 出所: tools/analyze_tansho_wide_portfolio.py 実行結果 (2026-07-04)
# finishing_order 重複除去済。ワイドは着順ペア順(1-2/1-3/2-3)で名前突合。
# v3.0: 2026春G1(桜花賞〜宝塚記念)+G2 8R 追加 → G1 n=46, G2 n=71。
# tag 判定: ROI≥0 かつ roi_ex2≥-0.20 → "投資級"
#           ROI≥-0.10                → "準投資級"
#           その他                   → "見送り"
STATS: dict = {
    "G1": {
        "単勝本命のみ":            {"hit": 0.457, "roi": +0.328, "roi_ex2": +0.170, "streak": 3, "n": 46, "tag": "投資級"},
        "単勝本命+ワイド本命対抗":  {"hit": 0.565, "roi": +0.124, "roi_ex2": -0.020, "streak": 3, "n": 46, "tag": "投資級"},
        "単勝本命+ワイド軸流し":    {"hit": 0.609, "roi": +0.059, "roi_ex2": -0.061, "streak": 3, "n": 46, "tag": "投資級"},
        "単勝本命×2+ワイドBOX3頭": {"hit": 0.652, "roi": +0.055, "roi_ex2": -0.076, "streak": 3, "n": 46, "tag": "投資級"},
        "単勝本命+ワイドBOX3頭":   {"hit": 0.652, "roi": -0.014, "roi_ex2": -0.181, "streak": 3, "n": 46, "tag": "準投資級"},
        "ワイドBOX上位3頭":        {"hit": 0.478, "roi": -0.128, "roi_ex2": -0.357, "streak": 4, "n": 46, "tag": "見送り"},
        "ワイドBOX人気1-4":        {"hit": 0.717, "roi": -0.134, "roi_ex2": -0.233, "streak": 3, "n": 46, "tag": "見送り"},
    },
    "G2": {
        "単勝本命のみ":            {"hit": 0.225, "roi": -0.372, "roi_ex2": -0.467, "streak": 12, "n": 71, "tag": "見送り"},
        "単勝本命+ワイド本命対抗":  {"hit": 0.394, "roi": -0.385, "roi_ex2": -0.457, "streak": 7,  "n": 71, "tag": "見送り"},
        "単勝本命+ワイド軸流し":    {"hit": 0.535, "roi": -0.277, "roi_ex2": -0.372, "streak": 7,  "n": 71, "tag": "見送り"},
        "単勝本命×2+ワイドBOX3頭": {"hit": 0.606, "roi": -0.252, "roi_ex2": -0.347, "streak": 6,  "n": 71, "tag": "見送り"},
        "単勝本命+ワイドBOX3頭":   {"hit": 0.606, "roi": -0.222, "roi_ex2": -0.322, "streak": 6,  "n": 71, "tag": "見送り"},
        "ワイドBOX上位3頭":        {"hit": 0.465, "roi": -0.172, "roi_ex2": -0.282, "streak": 6,  "n": 71, "tag": "見送り"},
        "ワイドBOX人気1-4":        {"hit": 0.592, "roi": -0.226, "roi_ex2": -0.329, "streak": 3,  "n": 71, "tag": "見送り"},
    },
}

# 各構成の一言メモ (display用)
RATIONALE: dict = {
    "単勝本命のみ":            "G1 ROI最強 (+33%)・ex2+17%。1点100円。外れても被害最小、当たれば最大回収。",
    "単勝本命+ワイド本命対抗":  "G1 バランスベスト。ROI+12%かつ的中57%。200円少点数で投資効率と回収を両立。",
    "単勝本命+ワイド軸流し":    "G1 投資級。単勝本命+ワイド本命→2番手・3番手の2点。的中61%・ROI+6%。",
    "単勝本命×2+ワイドBOX3頭": "G1 投資級。的中率最大(65%)でROI+6%。的中を最優先しつつプラス収支を狙う。",
    "単勝本命+ワイドBOX3頭":   "G1 準投資級。的中65%・ROI-1%。ほぼトントンだが回収率は保証できない。",
    "ワイドBOX上位3頭":        "的中48%・安価(300円)。G2で最も傷が浅い(ROI-17%)。G1では見送り推奨。",
    "ワイドBOX人気1-4":        "的中率最大(G1=72%/G2=59%)。ただし投資に対しROIが-13〜-23%。娯楽枠。",
}


def _f(v) -> float:
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def _num(h: dict):
    for k in ("number", "umaban", "horse_number"):
        v = h.get(k)
        try:
            n = int(str(v).strip())
            if n > 0:
                return n
        except (TypeError, ValueError):
            continue
    return None


def _disp(n) -> str:
    if n is None:
        return "?"
    if 1 <= n <= 20:
        return "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"[n - 1]
    return f"[{n}]"


def _grade_key(grade: str):
    g = (grade or "").upper()
    if "G1" in g or g == "GI":
        return "G1"
    if "G2" in g or g == "GII":
        return "G2"
    return None


def _card(h: dict, mark: str) -> dict:
    n = _num(h)
    return {
        "mark": mark,
        "name": (h.get("name") or "").strip(),
        "number": n,
        "number_disp": _disp(n),
        "odds": _f(h.get("odds")),
        "win_prob": _f(h.get("win_prob")),
    }


def _leg(ticket, lines, points):
    return {"ticket": ticket, "lines": lines, "points": points, "stake_yen": points * T}


def recommend_bets(ranked: list[dict], grade: str) -> dict:
    """単勝+ワイドの買い目レコメンドを返す (馬名+馬番付き)。

    Returns dict with:
      recommendation: 両立ベスト (G1) or 傷最小 (G2)
      alternatives:   目的別の代替案リスト
      note:           一言補足
    """
    gk = _grade_key(grade)
    ranked = [h for h in (ranked or []) if (h.get("name") or "").strip()]
    out = {
        "version": RECOMMENDER_VERSION, "grade": gk or (grade or ""),
        "applicable": gk is not None, "recommendation": None,
        "alternatives": [], "note": "",
    }
    if gk is None:
        out["note"] = "本レコメンダは G1/G2 のみ対応 (検証範囲)。"
        return out
    if len(ranked) < 3:
        out["note"] = "出走頭数/予想データ不足のため買い目を生成できません。"
        return out

    s = STATS[gk]
    c1 = _card(ranked[0], MARKS[0])
    c2 = _card(ranked[1], MARKS[1])
    c3 = _card(ranked[2], MARKS[2])
    top3 = [c1, c2, c3]

    # 市場人気順上位4頭 (ワイドBOX人気1-4用)
    mrank = gs.build_market_rank_map(ranked)
    fav4 = [_card(h, "・") for h in sorted(ranked, key=lambda h: mrank.get((h.get("name") or "").strip(), 99))[:4]]

    def wide_box_lines(cards):
        return [f"ワイド {a['number_disp']}-{b['number_disp']} {T}円"
                for a, b in combinations(cards, 2)]

    def build(method_key, label, legs, purpose=""):
        st = s.get(method_key, {})
        total = sum(lg["stake_yen"] for lg in legs)
        return {
            "method": method_key, "label": label, "legs": legs,
            "purpose": purpose,
            "total_stake": total,
            "hit_rate": st.get("hit"), "roi": st.get("roi"),
            "roi_ex2": st.get("roi_ex2"), "max_streak": st.get("streak"),
            "sample_n": st.get("n"), "tag": st.get("tag", "参考"),
            "rationale": RATIONALE.get(method_key, ""),
        }

    combos = {
        "単勝本命のみ": build(
            "単勝本命のみ", "単勝 本命1点 — 最高回収率",
            [_leg("単勝", [f"単勝 {c1['number_disp']}{c1['name']} {T}円"], 1)],
            purpose="回収率最大"),

        "単勝本命+ワイド本命対抗": build(
            "単勝本命+ワイド本命対抗", "単勝(本命) + ワイド 本命-対抗 2点",
            [_leg("単勝", [f"単勝 {c1['number_disp']}{c1['name']} {T}円"], 1),
             _leg("ワイド", [f"ワイド {c1['number_disp']}-{c2['number_disp']} {T}円"], 1)],
            purpose="投資効率×回収率の両立"),

        "単勝本命+ワイド軸流し": build(
            "単勝本命+ワイド軸流し", "単勝(本命) + 本命軸ワイド流し 3点",
            [_leg("単勝", [f"単勝 {c1['number_disp']}{c1['name']} {T}円"], 1),
             _leg("ワイド", [f"ワイド {c1['number_disp']}-{c2['number_disp']} {T}円",
                              f"ワイド {c1['number_disp']}-{c3['number_disp']} {T}円"], 2)],
            purpose="的中率を上げつつほぼトントン"),

        "単勝本命×2+ワイドBOX3頭": build(
            "単勝本命×2+ワイドBOX3頭", "単勝(本命×2倍) + ワイドBOX 上位3頭",
            [_leg("単勝", [f"単勝 {c1['number_disp']}{c1['name']} 200円 (2倍)"], 2),
             _leg("ワイド", wide_box_lines(top3), 3)],
            purpose="的中率60%・準投資級"),

        "ワイドBOX上位3頭": build(
            "ワイドBOX上位3頭", "ワイドBOX 上位3頭 3点",
            [_leg("ワイド", wide_box_lines(top3), 3)],
            purpose="G2で最も傷が浅い"),

        "ワイドBOX人気1-4": build(
            "ワイドBOX人気1-4", "ワイドBOX 人気上位4頭 6点",
            [_leg("ワイド", wide_box_lines(fav4) if len(fav4) >= 2 else [], len(list(combinations(fav4, 2))))],
            purpose="的中率最大"),
    }

    if gk == "G1":
        # バランスベスト = ROI≥0 かつ 少点数で回収率・的中率を両立
        # 実測 n=46: 単勝本命+ワイド本命対抗 → 的中56.5% / ROI+12.4% / ex2-2.0% / 連敗3
        rec = combos["単勝本命+ワイド本命対抗"]
        rec["verdict"] = "投資推奨"
        rec["why"] = (
            "G1 バックテスト(n=46, 〜2026年6月)で「ROI・的中率・少点数」すべての条件を満たす組み合わせ。"
            "単勝本命(ROI+33%)にワイド本命-対抗1点を加えることで的中率を46%→57%に引き上げ、"
            "ROI+12%・ex2-2%をキープ。1レース200円の少点数で最も投資効率が高い。"
        )
        out["recommendation"] = rec
        out["alternatives"] = [
            combos["単勝本命のみ"],           # ROI最大 +32.8% ex2+17% — 純粋回収重視
            combos["単勝本命+ワイド軸流し"],   # ROI+5.9% 投資級 — 的中61%
            combos["単勝本命×2+ワイドBOX3頭"],# ROI+5.5% 投資級 — 的中65%(最大)
        ]
        out["note"] = (
            "G1は5構成が投資級(ROI≥0)。的中最優先なら「単勝×2+ワイドBOX3頭」(65%/ROI+6%)、"
            "回収最優先なら「単勝本命のみ」(46%/ROI+33%)。"
            "※バックテスト n=46。憲法§4.2 により100ベット未満は確定視せず参考値扱い。"
        )

    else:  # G2
        # G2 は全構成マイナス。最も傷が浅い: ワイドBOX上位3頭 (ROI-17.2%)
        rec = combos["ワイドBOX上位3頭"]
        rec["verdict"] = "見送り推奨"
        rec["why"] = (
            "G2は単勝・ワイド全構成でROIがマイナス(本命単勝的中率23%・ROI-37%)。"
            "n=71レースで一度も正期待値は確認されていない。投資対象外として見送りが正解。"
            "どうしても参加するなら、最も傷が浅いワイドBOX上位3頭(ROI-17%、300円)に限定。"
        )
        out["recommendation"] = rec
        out["alternatives"] = [
            combos["ワイドBOX人気1-4"],  # 的中率最大 59.2% (ただしROI-23%)
            combos["単勝本命のみ"],      # 最低コスト100円 (ROI-37%)
        ]
        out["note"] = (
            "G2は n=71 で投資水準に一度も達していません。"
            "見送り、または予算ゼロで観戦が最適解です。"
        )
    return out


def format_text(rec: dict) -> str:
    """CLI/ログ用テキスト整形。"""
    if not rec.get("applicable"):
        return f"[{rec.get('grade')}] {rec.get('note')}"
    L = [f"=== 単勝+ワイド レコメンド [{rec['grade']}] {rec['version']} ==="]
    r = rec.get("recommendation") or {}
    if r:
        L.append(
            f"【{r.get('verdict','')}】{r['label']}"
            f"  的中{(r.get('hit_rate') or 0)*100:.0f}% / 回収{(r.get('roi') or 0)*100:+.0f}%"
            f" / 一発除外{(r.get('roi_ex2') or 0)*100:+.0f}% / 連敗{r.get('max_streak')}"
            f" / {r.get('total_stake')}円 [{r.get('tag')}]"
        )
        for leg in r.get("legs", []):
            for ln in leg["lines"]:
                L.append(f"    {leg['ticket']}: {ln}")
        if r.get("why"):
            L.append(f"    » {r['why']}")
    alts = rec.get("alternatives") or []
    if alts:
        L.append("--- 代替案 ---")
        for a in alts:
            L.append(
                f"[{a.get('tag')}] ({a.get('purpose','')}) {a['label']}"
                f"  的中{(a.get('hit_rate') or 0)*100:.0f}%  回収{(a.get('roi') or 0)*100:+.0f}%"
                f"  {a.get('total_stake')}円"
            )
            for leg in a.get("legs", []):
                for ln in leg["lines"]:
                    L.append(f"    {leg['ticket']}: {ln}")
    if rec.get("note"):
        L.append(rec["note"])
    return "\n".join(L)
