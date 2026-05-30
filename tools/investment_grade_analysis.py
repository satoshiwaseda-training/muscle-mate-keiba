"""Investment-grade analysis for G1/G2: hit rate + ROI + ROBUSTNESS.

READ-ONLY. Does not touch frozen rules/model/live pipeline.

The user's goal is "investment, not gambling": high hit rate AND positive
recovery AND low dependence on rare jackpots. So beyond hit/ROI we compute:
  - ROI_ex_big2 : ROI after removing the 2 biggest single-race payouts
                  (if ROI collapses -> it was a lottery, not an edge)
  - max_drawdown: deepest cumulative trough in stakes (chronological)
  - max_streak  : longest run of consecutive losing races
  - profit_factor: payout / cost

Also tests RACE SELECTION: betting 単勝 only when the model's top-1
win_prob is high (confidence-gated), to see whether selectivity creates
an edge (the constitution's "bet only when conditions are met").

Run: python3 tools/investment_grade_analysis.py
"""
from __future__ import annotations

import json
import sys
from itertools import combinations, permutations
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
TICKET = 100


def _norm(n):
    return (n or "").strip()


def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def bucket(g):
    g = (g or "").upper()
    if "G1" in g:
        return "G1"
    if "G2" in g:
        return "G2"
    return "OTHER"


def load_races():
    races = []
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
        fo = res.get("finishing_order") or []
        w = s2 = None
        for h in fo:
            try:
                r = int(h.get("rank", 0) or 0)
            except (ValueError, TypeError):
                continue
            if r == 1:
                w = _norm(h.get("name"))
            elif r == 2:
                s2 = _norm(h.get("name"))
        if not w or not s2:
            continue
        po = res.get("payouts") or {}
        pu, pe, pt = _po(po.get("馬連")), _po(po.get("馬単")), _po(po.get("単勝"))
        if pu <= 0 or pe <= 0:
            continue
        ranked = p.get("ranked") or []
        if len(ranked) < 3:
            continue
        races.append({
            "race_id": p.get("race_id"), "race_name": p.get("race_name", ""),
            "race_date": p.get("race_date", ""), "grade": g, "ranked": ranked,
            "winner": w, "second": s2, "p_umaren": pu, "p_umatan": pe,
            "p_tansho": pt, "market": gs.build_market_rank_map(ranked),
            "top1_winprob": float(ranked[0].get("win_prob", 0) or 0),
        })
    races.sort(key=lambda r: (r["race_date"], r["race_id"]))
    return races


# selections
def sel_model_topk(r, k):
    return [_norm(h.get("name")) for h in r["ranked"][:k]]


def sel_diversified(r):
    v = gs.build_prediction_variants(r["ranked"], r["grade"])
    return [_norm(h.get("name")) for h in v["primary"]["top3"]]


def sel_favk(r, k):
    return [nm for nm, _ in sorted(r["market"].items(), key=lambda kv: kv[1])[:k]]


# bets -> (cost, payout)
def umaren_box(S, r):
    cost = len(list(combinations(S, 2))) * TICKET
    hit = any(set(c) == {r["winner"], r["second"]} for c in combinations(S, 2))
    return cost, (r["p_umaren"] if hit else 0.0)


def umatan_box(S, r):
    cost = len(list(permutations(S, 2))) * TICKET
    hit = (r["winner"], r["second"]) in set(permutations(S, 2))
    return cost, (r["p_umatan"] if hit else 0.0)


def umatan_form1(S, r):
    cost = (len(S) - 1) * TICKET
    hit = (r["winner"] == S[0]) and (r["second"] in S[1:])
    return cost, (r["p_umatan"] if hit else 0.0)


def tansho_top1(S, r):
    return TICKET, (r["p_tansho"] if r["winner"] == S[0] else 0.0)


def metrics(races, sel, bet, min_k=2):
    seq = []  # per-race (cost, payout)
    for r in races:
        S = [x for x in sel(r) if x]
        if len(S) < min_k:
            continue
        c, p = bet(S, r)
        if c <= 0:
            continue
        seq.append((c, p, p - c))
    n = len(seq)
    if not n:
        return None
    hits = sum(1 for _, p, _ in seq if p > 0)
    cost = sum(c for c, _, _ in seq)
    payout = sum(p for _, p, _ in seq)
    roi = (payout - cost) / cost
    # ex-big2 robustness
    big2 = sorted((p for _, p, _ in seq), reverse=True)[:2]
    payout_ex = payout - sum(big2)
    # cost stays (still placed those bets); removing the winnings only
    roi_ex = (payout_ex - cost) / cost
    # chronological drawdown (cum pnl trough) in stakes
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    streak = 0
    maxstreak = 0
    for c, p, pnl in seq:
        cum += pnl
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
        if p > 0:
            streak = 0
        else:
            streak += 1
            maxstreak = max(maxstreak, streak)
    avg_stake = cost / n
    return {
        "n": n, "hits": hits, "hit_rate": hits / n, "roi": roi, "roi_ex2": roi_ex,
        "pnl": payout - cost, "mdd_yen": mdd, "mdd_stakes": mdd / avg_stake,
        "maxstreak": maxstreak, "pf": (payout / cost if cost else 0),
        "avg_stake": avg_stake,
    }


def row(label, m):
    if not m:
        return f"  {label:32s} (n/a)"
    grade = "投資級" if (m["roi"] >= 0 and m["roi_ex2"] >= -0.20) else (
        "条件付" if m["roi"] >= -0.10 else "博打")
    return (f"  {label:32s} n={m['n']:>3d} 的中{m['hit_rate']*100:4.1f}% "
            f"ROI{m['roi']*100:+6.1f}% ex2{m['roi_ex2']*100:+6.1f}% "
            f"DD{m['mdd_stakes']:>5.1f}口 連敗{m['maxstreak']:>2d} PF{m['pf']:.2f} [{grade}]")


CANDS = [
    ("単勝 model本命(1点)",        lambda r: sel_model_topk(r, 1), tansho_top1, 1),
    ("単勝 人気1(1点)",            lambda r: sel_favk(r, 1),       tansho_top1, 1),
    ("馬連BOX3点 diversified",     sel_diversified,                umaren_box, 2),
    ("馬連BOX3点 人気1-2-3",       lambda r: sel_favk(r, 3),       umaren_box, 2),
    ("馬連BOX6点 人気1-2-3-4",     lambda r: sel_favk(r, 4),       umaren_box, 2),
    ("馬連BOX10点 人気1-5",        lambda r: sel_favk(r, 5),       umaren_box, 2),
    ("馬単BOX6点 diversified",     sel_diversified,                umatan_box, 2),
    ("馬単1着固定 本命→相手2",     lambda r: sel_model_topk(r, 3), umatan_form1, 2),
]


def main():
    races = load_races()
    g1 = [r for r in races if r["grade"] == "G1"]
    g2 = [r for r in races if r["grade"] == "G2"]
    print("=" * 100)
    print("投資グレード分析 — G1/G2 (results.json 〜2026-04-05, 実JRA配当)")
    print("  判定: [投資級]=ROI≥0かつ一発除外後ex2≥-20%  [条件付]=ROI≥-10%  [博打]=それ未満")
    print("  DD=最大ドローダウン(平均賭け金=1口換算) 連敗=最長連続不的中 PF=払戻/投資")
    print("=" * 100)
    for gname, pool in [("G1", g1), ("G2", g2), ("G1+G2", races)]:
        print(f"\n[{gname}] n={len(pool)}")
        print("-" * 100)
        for label, sel, bet, mk in CANDS:
            print(row(label, metrics(pool, sel, bet, mk)))

    # RACE SELECTION: 単勝 model-top1 gated by win_prob confidence
    print("\n" + "=" * 100)
    print("レース選択の効果: 単勝 model本命を「本命のwin_prob下限」で絞る (G1+G2)")
    print("  -> 全レース買わず、自信のあるレースだけ買うと的中/ROIはどう動くか")
    print("=" * 100)
    for thr in (0.0, 0.20, 0.30, 0.40, 0.50):
        pool = [r for r in races if r["top1_winprob"] >= thr]
        m = metrics(pool, lambda r: sel_model_topk(r, 1), tansho_top1, 1)
        tag = "全R" if thr == 0 else f"prob≥{thr:.2f}"
        print(f"  {tag:10s} " + (row("", m).strip() if m else "(no races)"))


if __name__ == "__main__":
    main()
