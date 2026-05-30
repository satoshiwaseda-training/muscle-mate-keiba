"""Investment-grade backtest for 複勝 (place) & ワイド (quinella-place) on G1/G2.

Uses the NEW data/results.json `payouts_detail` (per-horse 複勝, per-pair
ワイド) produced by the fixed scraper + tools/backfill_payouts_detail.py.
READ-ONLY. Does not touch frozen rules/model/live pipeline.

Metrics (same investment lens as investment_grade_analysis.py):
  hit rate, ROI, ROI_ex_big2 (one-jackpot dependence), max drawdown (in
  stakes), longest losing streak, profit factor.

複勝/ワイド are the high-hit-rate, low-variance tickets — the natural core
of an "investment, not gambling" portfolio. This tool measures whether the
model's picks beat takeout on them.

Run: python3 tools/analyze_fukusho_wide_g1g2.py
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
TICKET = 100


def _norm(n):
    return (n or "").strip()


def bucket(g):
    g = (g or "").upper()
    return "G1" if "G1" in g else ("G2" if "G2" in g else "OTHER")


def _detail_lookup(detail, bet_type, numbers, ordered=False):
    entries = (detail or {}).get(bet_type) or []
    want = [str(numbers)] if isinstance(numbers, (str, int)) else [str(x) for x in numbers]
    for e in entries:
        combo = e.get("combo")
        if not combo:
            continue
        combo = [str(c) for c in combo]
        if (combo == want) if ordered else (sorted(combo) == sorted(want)):
            return int(e.get("yen") or 0)
    return 0


def load_races():
    races = []
    missing_detail = 0
    for f in sorted(PRED_DIR.glob("*_on.json")):
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if bucket(p.get("grade")) not in ("G1", "G2"):
            continue
        rid = str(p.get("race_id"))
        res = RES.get(f"bt_{rid}") or RES.get(rid)
        if not res:
            continue
        detail = res.get("payouts_detail")
        if not detail:
            missing_detail += 1
            continue
        fo = res.get("finishing_order") or []
        # map horse name -> 馬番 (umaban). netkeiba result rows don't store
        # umaban directly here, so we infer combos by NAME via finishing order
        # position: 複勝/ワイド detail uses 馬番, but we matched on rank above.
        # Instead we use the placed NAMES (rank<=3) and map them to detail by
        # the umaban embedded in finishing order if available; fall back to
        # rank-based place set for hit detection and use detail values by slot.
        placed = []  # (rank, name)
        for h in fo:
            try:
                r = int(h.get("rank", 0) or 0)
            except (ValueError, TypeError):
                continue
            if 1 <= r <= 3:
                placed.append((r, _norm(h.get("name"))))
        placed.sort()
        if len(placed) < 3:
            continue
        place_names = [nm for _, nm in placed]
        ranked = p.get("ranked") or []
        if len(ranked) < 3:
            continue
        # Build name<->umaban via finishing order index isn't reliable; the
        # detail combos are 馬番. We instead score by NAME using the slot order
        # of the 複勝 detail (sorted by finishing position == rank order).
        fuku = (detail.get("複勝") or [])
        # place payout by finishing rank slot: slot0=1着,1=2着,2=3着
        place_pay_by_name = {}
        for idx, (r, nm) in enumerate(placed):
            if idx < len(fuku):
                place_pay_by_name[nm] = int(fuku[idx].get("yen") or 0)
        # ワイド: map each placed pair -> yen by matching the 3 combos to the
        # 3 placed pairs (1-2,1-3,2-3) in canonical order.
        wide = (detail.get("ワイド") or [])
        wide_pairs = [frozenset((place_names[a], place_names[b]))
                      for a, b in ((0, 1), (0, 2), (1, 2))]
        wide_pay = {}
        if len(wide) == 3:
            for slot, pr in enumerate(wide_pairs):
                wide_pay[pr] = int(wide[slot].get("yen") or 0)
        races.append({
            "race_id": rid, "race_name": p.get("race_name", ""),
            "race_date": p.get("race_date", ""), "grade": bucket(p.get("grade")),
            "ranked": ranked, "place_names": set(place_names),
            "place_pay_by_name": place_pay_by_name, "wide_pay": wide_pay,
            "market": gs.build_market_rank_map(ranked),
        })
    races.sort(key=lambda r: (r["race_date"], r["race_id"]))
    return races, missing_detail


def sel_topk(r, k):
    return [_norm(h.get("name")) for h in r["ranked"][:k]]


def sel_div(r):
    v = gs.build_prediction_variants(r["ranked"], r["grade"])
    return [_norm(h.get("name")) for h in v["primary"]["top3"]]


def sel_favk(r, k):
    return [nm for nm, _ in sorted(r["market"].items(), key=lambda kv: kv[1])[:k]]


# bets -> (cost, payout)
def fukusho_each(S, r):
    """Buy 複勝 on each of S (len pts). Payout = sum of placed picks."""
    cost = len(S) * TICKET
    payout = sum(r["place_pay_by_name"].get(nm, 0) for nm in S if nm in r["place_names"])
    return cost, payout


def wide_box(S, r):
    """ワイド BOX of S. Payout = sum over picked pairs that both placed."""
    pairs = list(combinations(S, 2))
    cost = len(pairs) * TICKET
    payout = 0
    for a, b in pairs:
        pr = frozenset((a, b))
        if a in r["place_names"] and b in r["place_names"]:
            payout += r["wide_pay"].get(pr, 0)
    return cost, payout


def metrics(races, sel, bet, min_k=1):
    seq = []
    for r in races:
        S = [x for x in sel(r) if x]
        if len(S) < min_k:
            continue
        c, p = bet(S, r)
        if c <= 0:
            continue
        seq.append((c, p))
    n = len(seq)
    if not n:
        return None
    hits = sum(1 for _, p in seq if p > 0)
    cost = sum(c for c, _ in seq)
    payout = sum(p for _, p in seq)
    big2 = sorted((p for _, p in seq), reverse=True)[:2]
    roi = (payout - cost) / cost
    roi_ex = (payout - sum(big2) - cost) / cost
    cum = peak = mdd = 0.0
    streak = maxstreak = 0
    for c, p in seq:
        cum += p - c
        peak = max(peak, cum)
        mdd = min(mdd, cum - peak)
        if p > 0:
            streak = 0
        else:
            streak += 1
            maxstreak = max(maxstreak, streak)
    return {"n": n, "hit": hits / n, "roi": roi, "roi_ex2": roi_ex,
            "mdd": mdd / (cost / n), "streak": maxstreak, "pf": payout / cost}


def row(label, m):
    if not m:
        return f"  {label:30s} (n/a — no payouts_detail yet)"
    tag = "投資級" if (m["roi"] >= 0 and m["roi_ex2"] >= -0.20) else (
        "条件付" if m["roi"] >= -0.10 else "博打")
    return (f"  {label:30s} n={m['n']:>3d} 的中{m['hit']*100:4.1f}% "
            f"ROI{m['roi']*100:+6.1f}% ex2{m['roi_ex2']*100:+6.1f}% "
            f"DD{m['mdd']:>5.1f}口 連敗{m['streak']:>2d} PF{m['pf']:.2f} [{tag}]")


CANDS = [
    ("複勝 model本命1点",          lambda r: sel_topk(r, 1), fukusho_each, 1),
    ("複勝 人気1番1点",            lambda r: sel_favk(r, 1), fukusho_each, 1),
    ("複勝 model上位2頭",          lambda r: sel_topk(r, 2), fukusho_each, 1),
    ("複勝 model上位3頭",          lambda r: sel_topk(r, 3), fukusho_each, 1),
    ("複勝 diversified3頭",        sel_div,                  fukusho_each, 1),
    ("ワイドBOX model top3(3点)",  lambda r: sel_topk(r, 3), wide_box,    2),
    ("ワイドBOX diversified(3点)", sel_div,                  wide_box,    2),
    ("ワイドBOX 人気1-2-3(3点)",   lambda r: sel_favk(r, 3), wide_box,    2),
    ("ワイドBOX 人気1-4(6点)",     lambda r: sel_favk(r, 4), wide_box,    2),
]


def main():
    races, missing = load_races()
    g1 = [r for r in races if r["grade"] == "G1"]
    g2 = [r for r in races if r["grade"] == "G2"]
    print("=" * 100)
    print("複勝・ワイド 投資グレード分析 — G1/G2 (payouts_detail 使用)")
    print("  判定: [投資級]=ROI≥0かつex2≥-20%  [条件付]=ROI≥-10%  [博打]=それ未満")
    print("=" * 100)
    print(f"payouts_detail 有り: {len(races)}R (G1={len(g1)} G2={len(g2)}) | detail未取得でskip: {missing}R")
    if not races:
        print("\n⚠ payouts_detail がまだ無い。先に tools/backfill_payouts_detail.py を実行。")
        return
    for gname, pool in [("G1", g1), ("G2", g2), ("G1+G2", races)]:
        print(f"\n[{gname}] n={len(pool)}")
        print("-" * 100)
        for label, sel, bet, mk in CANDS:
            print(row(label, metrics(pool, sel, bet, mk)))


if __name__ == "__main__":
    main()
