"""Signal discovery — what features separate winners from market favorites?

Loads the 50-race enriched backtest and results, then:

  1. Partitions races into "fav won" vs "fav lost" (market-error races).
  2. For each feature available in structured_features, computes the
     discriminator: E[feature | winner] - E[feature | avg field] and
     separately E[feature | upset_winner] - E[feature | upset_favorite].
  3. Reports which features distinguish upset winners from the horses
     the market mispriced away from.
  4. Also tests whether the model currently deviates from the odds
     favorite in any meaningful way on this data, and whether those
     deviations correlate with upsets.

No network calls. Read-only on saved files.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).parent
ENR = ROOT / "data" / "enriched_backtest_results.json"
RES = ROOT / "data" / "results.json"


def _norm(n): return (n or "").strip()
def _po(s):
    s = str(s or "").strip().replace("---", "").replace("--", "").replace(",", "")
    try: return float(s)
    except ValueError: return 0.0


def load():
    enriched = json.loads(ENR.read_text(encoding="utf-8"))
    results = json.loads(RES.read_text(encoding="utf-8"))
    rows = []
    for er in enriched:
        rid = er["race_id"]
        res = results.get(f"bt_{rid}")
        if not res:
            continue
        fo = res.get("finishing_order") or []
        winner = second = third = None
        for h in fo:
            try:
                rk = int(h.get("rank", 0) or 0)
            except ValueError:
                continue
            nm = _norm(h.get("name"))
            if rk == 1: winner = nm
            elif rk == 2: second = nm
            elif rk == 3: third = nm
        if not winner:
            continue
        odds_map = {}
        for h in fo:
            nm = _norm(h.get("name"))
            od = _po(h.get("odds"))
            if nm and od > 0 and nm not in odds_map:
                odds_map[nm] = od
        if not odds_map:
            continue
        fav = min(odds_map, key=lambda n: odds_map[n])
        sf = er.get("structured_features") or {}
        rows.append({
            "race_id": rid,
            "race_name": er.get("race_name", ""),
            "grade": er.get("grade", ""),
            "winner": winner,
            "second": second,
            "third": third,
            "odds_map": odds_map,
            "favorite": fav,
            "fav_won": fav == winner,
            "sf_horses": sf.get("horses") or {},
            "race_info": sf.get("race") or {},
            "ranked": er.get("ranked", []),
            "selected_top3": er.get("selected_top3", []),
        })
    return rows


FEATURES = [
    "jockey_win_rate",
    "jockey_g1_wins",
    "horse_weight_delta",
    "horse_weight_kg",
    "carried_weight",
    "age",
    "training_acceleration",
    "training_cardio_index",
    "training_coat_gloss",
    "training_stride_quality",
    "paddock_vascularity",
    "paddock_hindquarter",
    "paddock_gait",
    "transport_stress",
    "waku",
    "number",
]


def get_feat(h: dict, key: str) -> float | None:
    v = h.get(key)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def feature_stat(rows, feature_name, subset="all"):
    """Compute mean feature value for (winners, favorites, field-avg) across
    a subset of races: 'all', 'upset' (fav lost), 'chalk' (fav won)."""
    winner_vals = []
    fav_vals = []
    field_vals = []
    for r in rows:
        if subset == "upset" and r["fav_won"]:
            continue
        if subset == "chalk" and not r["fav_won"]:
            continue
        w = r["sf_horses"].get(r["winner"], {})
        f = r["sf_horses"].get(r["favorite"], {})
        wv = get_feat(w, feature_name)
        fv = get_feat(f, feature_name)
        if wv is not None:
            winner_vals.append(wv)
        if fv is not None:
            fav_vals.append(fv)
        for name, h in r["sf_horses"].items():
            hv = get_feat(h, feature_name)
            if hv is not None:
                field_vals.append(hv)
    def m(vs):
        return st.mean(vs) if vs else float("nan")
    return {
        "winner_mean": m(winner_vals),
        "fav_mean": m(fav_vals),
        "field_mean": m(field_vals),
        "n_winner": len(winner_vals),
        "n_fav": len(fav_vals),
        "n_field": len(field_vals),
    }


def main():
    rows = load()
    n = len(rows)
    upset = [r for r in rows if not r["fav_won"]]
    chalk = [r for r in rows if r["fav_won"]]
    print(f"Total races       : {n}")
    print(f"Favorite won      : {len(chalk)} ({len(chalk)/n*100:.1f}%)")
    print(f"Favorite LOST     : {len(upset)} ({len(upset)/n*100:.1f}%) ← market-error races")
    print()

    # 1. Per-feature comparison: upset winner vs upset favorite
    print("="*96)
    print("Feature means — UPSET races (n={}) — winner vs the market favorite they beat".format(len(upset)))
    print("="*96)
    print(f"{'feature':30} {'w_mean':>9} {'f_mean':>9} {'diff':>9} {'field':>9} {'z_vs_field':>11} {'cov':>5}")
    print("-"*96)
    sig = []
    for feat in FEATURES:
        s = feature_stat(upset, feat, subset="all")
        w = s["winner_mean"]; f = s["fav_mean"]; fld = s["field_mean"]
        # z-score of winner above field (rough; no pooled stddev)
        all_vals = []
        for r in upset:
            for h in r["sf_horses"].values():
                v = get_feat(h, feat)
                if v is not None:
                    all_vals.append(v)
        sd = st.pstdev(all_vals) if len(all_vals) > 1 else 0
        z = ((w - fld) / sd) if (sd > 0 and not (w != w)) else 0
        cov = s["n_field"]
        # Coverage — fraction of horses in upset field having this feature
        nonzero = sum(1 for v in all_vals if v != 0)
        cov_pct = nonzero / cov * 100 if cov else 0
        diff = (w - f) if not (w != w or f != f) else float("nan")
        print(f"{feat:30} {w:>9.3f} {f:>9.3f} {diff:>+9.3f} {fld:>9.3f} {z:>+11.3f} {cov_pct:>4.0f}%")
        sig.append((abs(z), feat, z, diff, cov_pct))

    print("\nFeatures ranked by |z| of upset-winner vs field (signal strength):")
    sig.sort(reverse=True)
    for absz, feat, z, diff, cov in sig[:10]:
        print(f"  |z|={absz:5.3f}  z={z:+.3f}  diff(w-f)={diff:+.3f}  coverage={cov:.0f}%  {feat}")

    # 2. Chalk comparison for contrast
    print("\n" + "="*96)
    print("Feature means — CHALK races (n={}) — winner is the favorite".format(len(chalk)))
    print("="*96)
    print(f"{'feature':30} {'w_mean':>9} {'fld_mean':>9} {'z_vs_field':>11}")
    print("-"*96)
    for feat in FEATURES:
        s = feature_stat(chalk, feat, subset="all")
        w = s["winner_mean"]; fld = s["field_mean"]
        all_vals = []
        for r in chalk:
            for h in r["sf_horses"].values():
                v = get_feat(h, feat)
                if v is not None:
                    all_vals.append(v)
        sd = st.pstdev(all_vals) if len(all_vals) > 1 else 0
        z = ((w - fld) / sd) if (sd > 0 and not (w != w)) else 0
        print(f"{feat:30} {w:>9.3f} {fld:>9.3f} {z:>+11.3f}")

    # 3. Model deviation analysis
    print("\n" + "="*96)
    print("Model deviation from odds favorite (how often model disagrees, and with what effect)")
    print("="*96)
    dev_rows = []
    for r in rows:
        mt = r["ranked"][0]["name"] if r["ranked"] else ""
        if mt and mt != r["favorite"]:
            dev_rows.append({
                "race_id": r["race_id"],
                "grade": r["grade"],
                "model_top": mt,
                "mt_odds": r["odds_map"].get(mt, 0),
                "favorite": r["favorite"],
                "fav_odds": r["odds_map"][r["favorite"]],
                "winner": r["winner"],
                "model_right": mt == r["winner"],
                "fav_right": r["favorite"] == r["winner"],
            })
    print(f"Model disagreed with favorite in {len(dev_rows)}/{n} races")
    print(f"  Of those, model won: {sum(1 for d in dev_rows if d['model_right'])}")
    print(f"  Of those, fav  won: {sum(1 for d in dev_rows if d['fav_right'])}")
    print(f"  Of those, neither: {sum(1 for d in dev_rows if not d['model_right'] and not d['fav_right'])}")
    if dev_rows:
        print("\nDeviation details:")
        for d in dev_rows:
            print(f"  {d['race_id']} {d['grade']}  model={d['model_top']}@{d['mt_odds']:.1f}  "
                  f"fav={d['favorite']}@{d['fav_odds']:.1f}  winner={d['winner']}  "
                  f"{'M' if d['model_right'] else ('F' if d['fav_right'] else '-')}")


if __name__ == "__main__":
    main()
