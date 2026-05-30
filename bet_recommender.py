"""単勝 + ワイド 買い目レコメンダ (presentation layer — v2.1, 2026-05-30).

ユーザ要望: 「単勝とワイドで、投資効率(的中率・低分散)と回収率(ROIプラス)
を両立させて具体的な馬・馬番を提案」。

憲法上の位置づけ: grade_strategy.py と同じ「買い方の提示層」。推論
(score_runner / probability_engine / dual_mode_scoring / trigger_loose_capped)
と LOOSE 4 条件・モデル係数には一切触れない。表示専用。

検証出所 (tools/analyze_tansho_wide_portfolio.py, finishing_order重複除去済,
ワイドは着順ペア順1-2/1-3/2-3=全102R実証済):
  G1 n=38 / G2 n=63。憲法 §4.2 に従い 100 ベット未満は確定視しない (参考値)。

STATS は 2026-05-30 実スクリプト出力値をそのまま使用。手書き・推測値は一切含まない。
"""
from __future__ import annotations

from itertools import combinations

import grade_strategy as gs

RECOMMENDER_VERSION = "bet-recommender-v2.1-tansho-wide-2026-05-30"
T = 100  # 1点 100円
MARKS = ["◎", "○", "▲"]


# ── 検証済み実績テーブル ────────────────────────────────────────────
# 出所: tools/analyze_tansho_wide_portfolio.py 実行結果 (2026-05-30)
# finishing_order 重複除去済。ワイドは着順ペア順(1-2/1-3/2-3)で名前突合。
# tag 判定: ROI≥0 かつ roi_ex2≥-0.20 → "投資級"
#           ROI≥-0.10                → "準投資級"
#           その他                   → "見送り"
STATS: dict = {
    "G1": {
        "単勝本命のみ":            {"hit": 0.421, "roi": +0.242, "roi_ex2": +0.053, "streak": 3, "n": 38, "tag": "投資級"},
        "単勝本命+ワイド本命対抗":  {"hit": 0.526, "roi": +0.084, "roi_ex2": -0.094, "streak": 3, "n": 38, "tag": "投資級"},
        "単勝本命+ワイド軸流し":    {"hit": 0.553, "roi": -0.034, "roi_ex2": -0.157, "streak": 3, "n": 38, "tag": "準投資級"},
        "単勝本命×2+ワイドBOX3頭": {"hit": 0.605, "roi": -0.032, "roi_ex2": -0.175, "streak": 3, "n": 38, "tag": "準投資級"},
        "単勝本命+ワイドBOX3頭":   {"hit": 0.605, "roi": -0.101, "roi_ex2": -0.262, "streak": 3, "n": 38, "tag": "見送り"},
        "ワイドBOX上位3頭":        {"hit": 0.447, "roi": -0.215, "roi_ex2": -0.451, "streak": 4, "n": 38, "tag": "見送り"},
        "ワイドBOX人気1-4":        {"hit": 0.684, "roi": -0.171, "roi_ex2": -0.294, "streak": 3, "n": 38, "tag": "見送り"},
    },
    "G2": {
        "単勝本命のみ":            {"hit": 0.206, "roi": -0.437, "roi_ex2": -0.538, "streak": 11, "n": 63, "tag": "見送り"},
        "単勝本命+ワイド本命対抗":  {"hit": 0.381, "roi": -0.398, "roi_ex2": -0.480, "streak": 7,  "n": 63, "tag": "見送り"},
        "単勝本命+ワイド軸流し":    {"hit": 0.508, "roi": -0.344, "roi_ex2": -0.454, "streak": 7,  "n": 63, "tag": "見送り"},
        "単勝本命×2+ワイドBOX3頭": {"hit": 0.587, "roi": -0.292, "roi_ex2": -0.401, "streak": 6,  "n": 63, "tag": "見送り"},
        "単勝本命+ワイドBOX3頭":   {"hit": 0.587, "roi": -0.256, "roi_ex2": -0.370, "streak": 6,  "n": 63, "tag": "見送り"},
        "ワイドBOX上位3頭":        {"hit": 0.460, "roi": -0.196, "roi_ex2": -0.321, "streak": 6,  "n": 63, "tag": "見送り"},
        "ワイドBOX人気1-4":        {"hit": 0.587, "roi": -0.253, "roi_ex2": -0.371, "streak": 3,  "n": 63, "tag": "見送り"},
    },
}

# 各構成の一言メモ (display用)
RATIONALE: dict = {
    "単勝本命のみ":            "G1 ROI最大 (+24%)。1点100円で最も効率的。単勝が当たれば十分な回収。",
    "単勝本命+ワイド本命対抗":  "G1 両立ベスト。ROI+8%かつ的中53%。少点数(200円)で当たりやすさと回収を両立。",
    "単勝本命+ワイド軸流し":    "G1 準投資級。単勝本命+ワイド本命→2・3番手。的中55%・ROI-3%でほぼトントン。",
    "単勝本命×2+ワイドBOX3頭": "G1 準投資級。本命を単勝2倍買い+ワイドBOX。的中61%だが ROI-3%。娯楽枠。",
    "単勝本命+ワイドBOX3頭":   "的中61%だが G1 ROI-10%。点数(400円)に対して回収が追いつかない。",
    "ワイドBOX上位3頭":        "的中45%・安価(300円)。G2で最も傷が浅い(ROI-20%)。G1では見送り推奨。",
    "ワイドBOX人気1-4":        "的中率最大(G1=68%/G2=59%)。ただし投資に対しROIが-17〜-25%。娯楽枠。",
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
        # 両立ベスト = ROI≥0 かつ 的中率が最大の組み合わせ
        # 実測: 単勝本命+ワイド本命対抗 → 的中52.6% / ROI+8.4% / ex2-9.4% / 連敗3
        rec = combos["単勝本命+ワイド本命対抗"]
        rec["verdict"] = "投資推奨"
        rec["why"] = (
            "G1 バックテスト(n=38)で「ROI≥0 かつ的中率最大」の組み合わせ。"
            "単勝本命の期待値プラス(+24%)にワイド本命-対抗1点を加えることで、"
            "的中率を42%→53%まで高めながらROI+8%をキープ。"
            "1レース200円の少点数で投資効率と回収率を両立できる唯一の構成。"
        )
        out["recommendation"] = rec
        out["alternatives"] = [
            combos["単勝本命のみ"],           # ROI最大 +24.2% — 回収重視
            combos["単勝本命+ワイド軸流し"],   # ROI-3.4% 準投資級 — 的中55%
            combos["単勝本命×2+ワイドBOX3頭"],# ROI-3.2% 準投資級 — 的中61%
        ]
        out["note"] = (
            "G1は単勝本命と「+ワイド本命-対抗」の2点構成が両立ベスト。"
            "資金を増やしたい場合は単勝本命のみ(ROI+24%)が最も効率的。"
            "※データは〜2026年4月・38Rのサンプル。参考値として使用してください。"
        )

    else:  # G2
        # G2 は全構成マイナス。最も傷が浅い: ワイドBOX上位3頭 (ROI-19.6%)
        rec = combos["ワイドBOX上位3頭"]
        rec["verdict"] = "見送り推奨"
        rec["why"] = (
            "G2は単勝・ワイド全構成でROIがマイナス(本命単勝的中率21%・ROI-44%)。"
            "投資対象外として見送りが正解。"
            "どうしても参加するなら、最も傷が浅いワイドBOX上位3頭(ROI-20%、300円)に限定。"
        )
        out["recommendation"] = rec
        out["alternatives"] = [
            combos["ワイドBOX人気1-4"],  # 的中率最大 58.7% (ただしROI-25%)
            combos["単勝本命のみ"],      # 最低コスト100円 (ROI-44%)
        ]
        out["note"] = (
            "G2は現データで投資水準に達していません。"
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
