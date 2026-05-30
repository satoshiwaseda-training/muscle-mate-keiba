"""Offline verification: 馬連 (quinella) vs 馬単 (exacta) buying methods for G1/G2.

READ-ONLY analysis. Does NOT touch the frozen loose rule, model files, or
live pipeline. Reads only:
  - data/backtest_predictions/*_on.json   (model `ranked` per race)
  - data/results.json                      (real JRA payouts incl. 馬連/馬単)

Goal (user request 2026-05-30): among 馬連/馬単 buying methods, find the
highest 的中率 on G1/G2, and report ROI alongside (constitution: ROI is
the real metric).

Run:  python3 tools/analyze_umaren_umatan_g1g2.py
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


def _norm(n) -> str:
    return (n or "").strip()


def _po(s) -> float:
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return 0.0


def bucket(g: str) -> str:
    g = (g or "").upper()
    if "G1" in g or g == "GI":
        return "G1"
    if "G2" in g or g == "GII":
        return "G2"
    if "G3" in g or g == "GIII":
        return "G3"
    return "OTHER"


def load_races():
    races = []
    skipped = {"no_result": 0, "no_pair": 0, "no_payout": 0, "few_ranked": 0}
    for f in sorted(PRED_DIR.glob("*_on.json")):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        g = bucket(p.get("grade"))
        if g not in ("G1", "G2"):
            continue
        rid = p.get("race_id")
        res = RES.get(f"bt_{rid}") or RES.get(str(rid))
        if not res:
            skipped["no_result"] += 1
            continue
        fo = res.get("finishing_order") or []
        winner = second = None
        for h in fo:
            try:
                r = int(h.get("rank", 0) or 0)
            except (ValueError, TypeError):
                continue
            if r == 1:
                winner = _norm(h.get("name"))
            elif r == 2:
                second = _norm(h.get("name"))
        if not winner or not second:
            skipped["no_pair"] += 1
            continue
        payouts = res.get("payouts") or {}
        p_umaren = _po(payouts.get("馬連"))
        p_umatan = _po(payouts.get("馬単"))
        p_tansho = _po(payouts.get("単勝"))
        if p_umaren <= 0 or p_umatan <= 0:
            skipped["no_payout"] += 1
            continue
        ranked = p.get("ranked") or []
        if len(ranked) < 3:
            skipped["few_ranked"] += 1
            continue
        races.append({
            "race_id": rid, "race_name": p.get("race_name", ""),
            "race_date": p.get("race_date", ""), "grade": g,
            "ranked": ranked, "winner": winner, "second": second,
            "p_umaren": p_umaren, "p_umatan": p_umatan, "p_tansho": p_tansho,
            "market": gs.build_market_rank_map(ranked),
        })
    return races, skipped


# ── selections: ORDERED list of names (S[0] = 本命) ──
def sel_model_topk(race, k):
    return [_norm(h.get("name")) for h in race["ranked"][:k]]


def sel_diversified(race):
    v = gs.build_prediction_variants(race["ranked"], race["grade"])
    return [_norm(h.get("name")) for h in v["primary"]["top3"]]


def sel_favk(race, k):
    inv = sorted(race["market"].items(), key=lambda kv: kv[1])
    return [nm for nm, _ in inv[:k]]


# ── bet evaluators: (cost, payout) ──
def bet_umaren_box(S, race):
    if len(S) < 2:
        return 0.0, 0.0
    cost = len(list(combinations(S, 2))) * TICKET
    hit = any(set(c) == {race["winner"], race["second"]} for c in combinations(S, 2))
    return cost, (race["p_umaren"] if hit else 0.0)


def bet_umatan_box(S, race):
    if len(S) < 2:
        return 0.0, 0.0
    cost = len(list(permutations(S, 2))) * TICKET
    hit = (race["winner"], race["second"]) in set(permutations(S, 2))
    return cost, (race["p_umatan"] if hit else 0.0)


def bet_umatan_formation_1st(S, race):
    if len(S) < 2:
        return 0.0, 0.0
    cost = (len(S) - 1) * TICKET
    hit = (race["winner"] == S[0]) and (race["second"] in S[1:])
    return cost, (race["p_umatan"] if hit else 0.0)


def bet_umatan_nagashi_12(S, race):
    if len(S) < 2:
        return 0.0, 0.0
    cost = 2 * (len(S) - 1) * TICKET
    axis, rest = S[0], set(S[1:])
    hit = ((race["winner"] == axis and race["second"] in rest) or
           (race["second"] == axis and race["winner"] in rest))
    return cost, (race["p_umatan"] if hit else 0.0)


def bet_umaren_nagashi(S, race):
    if len(S) < 2:
        return 0.0, 0.0
    cost = (len(S) - 1) * TICKET
    axis, rest = S[0], set(S[1:])
    pair = {race["winner"], race["second"]}
    other = pair - {axis}
    hit = (axis in pair) and bool(other) and (next(iter(other)) in rest)
    return cost, (race["p_umaren"] if hit else 0.0)


def make_strategies():
    return [
        ("馬連BOX3点 / model top3",        lambda r: sel_model_topk(r, 3), bet_umaren_box, "3"),
        ("馬連BOX3点 / diversified",       sel_diversified,                bet_umaren_box, "3"),
        ("馬連BOX3点 / 人気1-2-3",         lambda r: sel_favk(r, 3),       bet_umaren_box, "3"),
        ("馬連BOX6点 / 人気1-2-3-4",       lambda r: sel_favk(r, 4),       bet_umaren_box, "6"),
        ("馬連BOX10点 / 人気1-5",          lambda r: sel_favk(r, 5),       bet_umaren_box, "10"),
        ("馬連 軸流し(本命-相手2)/model",  lambda r: sel_model_topk(r, 3), bet_umaren_nagashi, "2"),
        ("馬連 軸流し(人気1-相手3)/fav",   lambda r: sel_favk(r, 4),       bet_umaren_nagashi, "3"),
        ("馬単BOX6点 / model top3",        lambda r: sel_model_topk(r, 3), bet_umatan_box, "6"),
        ("馬単BOX6点 / diversified",       sel_diversified,                bet_umatan_box, "6"),
        ("馬単BOX6点 / 人気1-2-3",         lambda r: sel_favk(r, 3),       bet_umatan_box, "6"),
        ("馬単BOX12点 / 人気1-2-3-4",      lambda r: sel_favk(r, 4),       bet_umatan_box, "12"),
        ("馬単1着固定(本命→相手2)/model",  lambda r: sel_model_topk(r, 3), bet_umatan_formation_1st, "2"),
        ("馬単1着固定(人気1→相手2)/fav",   lambda r: sel_favk(r, 3),       bet_umatan_formation_1st, "2"),
        ("馬単1・2着流し(本命⇄相手2)/model",lambda r: sel_model_topk(r, 3), bet_umatan_nagashi_12, "4"),
        ("馬単1・2着流し(人気1⇄相手3)/fav", lambda r: sel_favk(r, 4),       bet_umatan_nagashi_12, "6"),
    ]


def evaluate(races, sel_fn, bet_fn):
    n = hits = 0
    cost = payout = 0.0
    for r in races:
        try:
            S = [s for s in sel_fn(r) if s]
        except Exception:
            continue
        if len(S) < 2:
            continue
        c, p = bet_fn(S, r)
        if c <= 0:
            continue
        n += 1
        cost += c
        payout += p
        if p > 0:
            hits += 1
    return {"n": n, "hits": hits, "hit_rate": (hits / n if n else 0.0),
            "cost": cost, "payout": payout, "pnl": payout - cost,
            "roi": ((payout - cost) / cost if cost else 0.0)}


def fmt_row(label, npts, m):
    if not m["n"]:
        return f"  {label:30s} {npts:>3s}点  (no bets)"
    return (f"  {label:30s} {npts:>3s}点  n={m['n']:>3d}  "
            f"的中{m['hits']:>3d}/{m['n']:<3d}({m['hit_rate']*100:5.1f}%)  "
            f"ROI={m['roi']*100:+7.1f}%  pnl={m['pnl']:+8.0f}")


def main():
    races, skipped = load_races()
    by = {"G1": [r for r in races if r["grade"] == "G1"],
          "G2": [r for r in races if r["grade"] == "G2"]}
    print("=" * 92)
    print("馬連 / 馬単 買い方比較 — G1/G2 バックテスト (results.json 〜2026-04-05, 実JRA配当)")
    print("=" * 92)
    print(f"対象: G1={len(by['G1'])}  G2={len(by['G2'])}  G1+G2={len(races)}  skipped={skipped}")

    strategies = make_strategies()
    for gname, pool in [("G1", by["G1"]), ("G2", by["G2"]), ("G1+G2", races)]:
        print("\n" + "-" * 92 + f"\n[{gname}] n={len(pool)}\n" + "-" * 92)
        results = [(l, npts, evaluate(pool, sf, bf)) for (l, sf, bf, npts) in strategies]
        for label, npts, m in results:
            print(fmt_row(label, npts, m))
        valid = [(l, n, m) for (l, n, m) in results if m["n"] >= 10]
        if valid:
            bh = max(valid, key=lambda x: x[2]["hit_rate"])
            br = max(valid, key=lambda x: x[2]["roi"])
            print(f"  >>> 最高的中率: {bh[0]} = {bh[2]['hit_rate']*100:.1f}% (ROI {bh[2]['roi']*100:+.1f}%)")
            print(f"  >>> 最高ROI  : {br[0]} = {br[2]['roi']*100:+.1f}% (的中 {br[2]['hit_rate']*100:.1f}%)")

    print("\n" + "=" * 92 + "\n参考: モデル素の精度\n" + "=" * 92)
    for gname, pool in [("G1", by["G1"]), ("G2", by["G2"]), ("G1+G2", races)]:
        if not pool:
            continue
        t1 = sum(1 for r in pool if sel_model_topk(r, 1)[0] == r["winner"]) / len(pool)
        t1p = sum(1 for r in pool if sel_model_topk(r, 1)[0] in (r["winner"], r["second"])) / len(pool)
        f1 = sum(1 for r in pool if (sel_favk(r, 1) or [None])[0] == r["winner"]) / len(pool)
        t3 = sum(1 for r in pool if r["winner"] in sel_model_topk(r, 3)) / len(pool)
        print(f"  {gname:6s} n={len(pool):>3d}  本命的中(単勝)={t1*100:4.1f}%  本命連対={t1p*100:4.1f}%  "
              f"人気1の単勝={f1*100:4.1f}%  勝馬がmodel_top3内={t3*100:4.1f}%")


if __name__ == "__main__":
    main()
