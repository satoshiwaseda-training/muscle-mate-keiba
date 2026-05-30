"""単勝 + ワイド ポートフォリオの投資グレード検証 (G1/G2).

ユーザ要望: 単勝とワイドで「投資効率(的中率・低分散) と 回収率(ROIプラス)
を両立」させる。本ツールは finishing_order を dedup し、ワイドは着順ペア順
(検証: payouts_detail のワイド3組は全102Rで 1-2着/1-3着/2-3着 順=ALL_MATCH)
で名前突合して、単勝・ワイド単体と複数の複合ポートフォリオを実測する。

READ-ONLY。推論/LOOSE/モデルには触れない。
出所: data/results.json (payouts_detail), backtest_predictions/*_on.json。

指標: 的中率(回収のあったレース割合), ROI, ROI_ex2(高配当上位2R除外),
      最長連敗, profit factor。
Run: python3 tools/analyze_tansho_wide_portfolio.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import grade_strategy as gs  # noqa: E402

PRED_DIR = ROOT / "data" / "backtest_predictions"
RES = json.loads((ROOT / "data" / "results.json").read_text(encoding="utf-8"))
T = 100  # 1点 100円


def _norm(n):
    return (n or "").strip()


def _f(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return 0.0


def bucket(g):
    g = (g or "").upper()
    return "G1" if "G1" in g else ("G2" if "G2" in g else "OTHER")


def dedup_top3(res):
    """Return [1着名,2着名,3着名] with sticky-clone rows removed, else None."""
    fo = res.get("finishing_order") or []
    seen = set()
    r2 = {}
    for h in fo:
        rk = h.get("rank")
        if not str(rk).isdigit():
            continue
        rk = int(rk)
        key = h.get("horse_id") or _norm(h.get("name"))
        if key in seen:
            continue
        seen.add(key)
        if rk in (1, 2, 3) and rk not in r2:
            r2[rk] = _norm(h.get("name"))
    if all(r in r2 for r in (1, 2, 3)):
        return [r2[1], r2[2], r2[3]]
    return None


def load_races():
    out = []
    for f in sorted(PRED_DIR.glob("*_on.json")):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        g = bucket(p.get("grade"))
        if g not in ("G1", "G2"):
            continue
        res = RES.get(f"bt_{p.get('race_id')}")
        if not res:
            continue
        d = res.get("payouts_detail") or {}
        wide = d.get("ワイド") or []
        if len(wide) != 3:
            continue
        place = dedup_top3(res)
        if not place:
            continue
        ranked = [h for h in (p.get("ranked") or []) if _norm(h.get("name"))]
        if len(ranked) < 3:
            continue
        # 単勝(1着)配当: flat payouts["単勝"] = winner payout
        tansho = _f((res.get("payouts") or {}).get("単勝"))
        # ワイド: 着順ペア順 [1-2, 1-3, 2-3]
        wy = [int(w.get("yen") or 0) for w in wide]
        pair_pay = {
            frozenset((place[0], place[1])): wy[0],
            frozenset((place[0], place[2])): wy[1],
            frozenset((place[1], place[2])): wy[2],
        }
        out.append({
            "grade": g, "name": p.get("race_name", ""), "date": p.get("race_date", ""),
            "place": set(place), "winner": place[0], "tansho_pay": tansho,
            "pair_pay": pair_pay, "ranked": ranked,
            "market": gs.build_market_rank_map(ranked),
        })
    out.sort(key=lambda r: (r["date"], r["name"]))
    return out


def topk(r, k):
    return [_norm(h.get("name")) for h in r["ranked"][:k]]


# ── bet primitives: each returns (cost, payout) for one race ──
def tansho_honmei(r):
    return T, (r["tansho_pay"] if topk(r, 1)[0] == r["winner"] else 0)


def wide_box(r, picks):
    pairs = list(combinations(picks, 2))
    cost = len(pairs) * T
    pay = 0
    for a, b in pairs:
        if a in r["place"] and b in r["place"]:
            pay += r["pair_pay"].get(frozenset((a, b)), 0)
    return cost, pay


def wide_nagashi(r, axis, others):
    """本命軸ワイド流し: axis-others の (len others) 点."""
    cost = len(others) * T
    pay = 0
    for o in others:
        if axis in r["place"] and o in r["place"]:
            pay += r["pair_pay"].get(frozenset((axis, o)), 0)
    return cost, pay


# ── portfolio definitions: name -> fn(race)->(cost,payout) ──
def make_portfolios():
    def P_tansho_only(r):
        return tansho_honmei(r)

    def P_wide_box3(r):
        return wide_box(r, topk(r, 3))

    def P_wide_box_fav4(r):
        favs = [nm for nm, _ in sorted(r["market"].items(), key=lambda kv: kv[1])[:4]]
        return wide_box(r, favs)

    def P_t1_widebox3(r):
        c1, p1 = tansho_honmei(r)
        c2, p2 = wide_box(r, topk(r, 3))
        return c1 + c2, p1 + p2

    def P_t1_wide_nagashi(r):
        # 単勝本命 + 本命軸ワイド流し(相手=2,3番手) → 単勝1 + ワイド2 = 3点
        c1, p1 = tansho_honmei(r)
        picks = topk(r, 3)
        c2, p2 = wide_nagashi(r, picks[0], picks[1:3])
        return c1 + c2, p1 + p2

    def P_t2_widebox3(r):
        # 単勝本命を2倍(200円) + ワイドBOX3頭(300円) → 単勝厚め
        c1, p1 = tansho_honmei(r)
        c2, p2 = wide_box(r, topk(r, 3))
        return c1 * 2 + c2, p1 * 2 + p2

    def P_t1_wide_honmei1(r):
        # 単勝本命 + ワイド本命-対抗 1点 = 2点 (最小)
        c1, p1 = tansho_honmei(r)
        picks = topk(r, 3)
        c2, p2 = wide_nagashi(r, picks[0], picks[1:2])
        return c1 + c2, p1 + p2

    return [
        ("単勝本命のみ(1点)",                 P_tansho_only),
        ("ワイドBOX 上位3頭(3点)",            P_wide_box3),
        ("ワイドBOX 人気1-4(6点)",            P_wide_box_fav4),
        ("単勝本命 + ワイド本命-対抗(2点)",    P_t1_wide_honmei1),
        ("単勝本命 + 本命軸ワイド流し(3点)",   P_t1_wide_nagashi),
        ("単勝本命 + ワイドBOX3頭(4点)",       P_t1_widebox3),
        ("単勝本命×2 + ワイドBOX3頭(5点)",     P_t2_widebox3),
    ]


def evaluate(races, fn):
    seq = []
    for r in races:
        c, p = fn(r)
        if c <= 0:
            continue
        seq.append((c, p))
    n = len(seq)
    if not n:
        return None
    hits = sum(1 for _, p in seq if p > 0)
    cost = sum(c for c, _ in seq)
    pay = sum(p for _, p in seq)
    ss = sorted(seq, key=lambda x: x[1], reverse=True)
    pe = sum(p for _, p in ss[2:])
    ce = sum(c for c, _ in ss[2:])
    cum = peak = mdd = 0.0
    st = mx = 0
    for c, p in seq:
        cum += p - c
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
        st = 0 if p > 0 else st + 1
        mx = max(mx, st)
    return {
        "n": n, "hit": hits / n, "roi": (pay - cost) / cost,
        "roi_ex2": ((pe - ce) / ce if ce else 0), "streak": mx,
        "pf": pay / cost, "avg_cost": cost / n,
    }


def tag(m):
    if m["roi"] >= 0 and m["roi_ex2"] >= -0.20:
        return "投資級"
    if m["roi"] >= -0.10:
        return "準投資級"
    return "見送り"


def row(label, m):
    if not m:
        return f"  {label:30s} (n/a)"
    return (f"  {label:30s} n={m['n']:>3d} 的中{m['hit']*100:5.1f}% "
            f"ROI{m['roi']*100:+6.1f}% ex2{m['roi_ex2']*100:+6.1f}% "
            f"連敗{m['streak']:>2d} PF{m['pf']:.2f} 平均{m['avg_cost']:.0f}円 [{tag(m)}]")


def main():
    races = load_races()
    g1 = [r for r in races if r["grade"] == "G1"]
    g2 = [r for r in races if r["grade"] == "G2"]
    print("=" * 104)
    print("単勝 + ワイド ポートフォリオ検証 — G1/G2 (finishing_order重複除去, ワイド着順ペア順=実証済)")
    print("  的中率=単勝orワイドどちらか回収があったレース割合 / ex2=高配当上位2R除外ROI")
    print("=" * 104)
    ports = make_portfolios()
    for gname, pool in [("G1", g1), ("G2", g2), ("G1+G2", races)]:
        print(f"\n[{gname}] n={len(pool)}")
        print("-" * 104)
        scored = [(lab, evaluate(pool, fn)) for lab, fn in ports]
        for lab, m in scored:
            print(row(lab, m))
        valid = [(l, m) for l, m in scored if m]
        if valid:
            # 「両立」= ROI>=0 を満たす中で的中率最大
            inv = [(l, m) for l, m in valid if m["roi"] >= 0]
            if inv:
                best = max(inv, key=lambda x: x[1]["hit"])
                print(f"  ★両立(ROI≥0で的中最大): {best[0]} → 的中{best[1]['hit']*100:.0f}% "
                      f"ROI{best[1]['roi']*100:+.0f}% 連敗{best[1]['streak']}")
            else:
                best = max(valid, key=lambda x: x[1]["roi"])
                print(f"  ※ROI≥0は無し。ROI最大: {best[0]} → ROI{best[1]['roi']*100:+.0f}%")


if __name__ == "__main__":
    main()
