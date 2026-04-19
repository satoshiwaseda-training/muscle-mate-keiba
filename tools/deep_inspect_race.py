"""Deep-inspect a single race — dump every cell of every horse row.

When the live panel shows obviously-wrong odds (like 170.3 / 678.1 / 788.7
for 2026 皐月賞 entries), this tool is the single call that reveals the
true cause. It does NOT go through the consensus layer — it directly
hits the source HTML and prints:

  (a) netkeiba shutuba: every <td> in every row, with class names
  (b) netkeiba JSON odds API: raw payload
  (c) what our scraper actually extracts
  (d) diff between (a)-derived odds and (c)

Usage:
  python tools/deep_inspect_race.py --race-id 202606030811
  python tools/deep_inspect_race.py --race-id 202606030811 --save-html
  python tools/deep_inspect_race.py --race-id 202606030811 --horse フォルテアンジェロ

The output intentionally is large — copy the whole thing into your bug
report so the pipeline can be traced end-to-end without guessing.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests
from bs4 import BeautifulSoup

import scraper  # noqa: E402


SHUTUBA_URL = "https://race.netkeiba.com/race/shutuba.html?race_id={rid}"
ODDS_API_URL = (
    "https://race.netkeiba.com/api/api_get_jra_odds.html"
    "?type=1&locale=ja&race_id={rid}"
)


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/128.0 Safari/537.36"
    ),
    "Accept-Language": "ja,en-US;q=0.7,en;q=0.3",
}


def _get(url: str, *, as_json: bool = False, timeout: int = 15):
    """Plain GET; returns .text or parsed JSON. Raises on any failure."""
    print(f"[deep] GET {url}")
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    print(f"[deep]   → HTTP {resp.status_code}  ({len(resp.content)} bytes)")
    resp.raise_for_status()
    if as_json:
        return resp.json()
    return resp.text


def dump_shutuba(race_id: str, save_html: bool = False,
                 horse_filter: str | None = None) -> None:
    url = SHUTUBA_URL.format(rid=race_id)
    html = _get(url)

    if save_html:
        out = Path(__file__).resolve().parent.parent / "data" / "deep_inspect"
        out.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        p = out / f"shutuba_{race_id}_{ts}.html"
        p.write_text(html, encoding="utf-8")
        print(f"[deep] saved HTML: {p}")

    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("tr.HorseList")
    print(f"\n[deep] rows found: {len(rows)}")
    if not rows:
        # Maybe the selector changed — fall back
        rows = soup.select("table.Shutuba_Table tr")
        print(f"[deep] fallback table.Shutuba_Table tr: {len(rows)}")

    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        # Find the horse name for filtering / display
        name_cell = row.select_one(".HorseInfo a, .HorseName a, td.HorseInfo a")
        name = name_cell.get_text(strip=True) if name_cell else ""
        if horse_filter and horse_filter not in name:
            continue

        print(f"\n── Horse row: {name} ──")
        print(f"  number of <td>: {len(cells)}")
        for i, td in enumerate(cells):
            txt = td.get_text(" ", strip=True)
            classes = td.get("class") or []
            # Short-hand: class names + text (truncated)
            cls_str = ".".join(classes) or "(no class)"
            short = (txt[:60] + "…") if len(txt) > 60 else txt
            print(f"  cells[{i:>2}]  class={cls_str:<30}  text={short!r}")

        # Apply our current parser to THIS row and show both outcomes
        try:
            parsed = scraper._parse_shutuba_odds(row, cells)
            print(f"  → scraper._parse_shutuba_odds() returns: {parsed!r}")
        except Exception as e:
            print(f"  → scraper._parse_shutuba_odds() FAILED: {e}")

        # Also show what each selector attempt would yield
        td_odds = row.select_one("td.Odds")
        txr_odds = row.select_one("td.Txt_R.Odds")
        td_popular = row.select_one("td.Popular")
        print(f"    td.Odds          → {td_odds.get_text(strip=True) if td_odds else '(missing)'}")
        print(f"    td.Txt_R.Odds    → {txr_odds.get_text(strip=True) if txr_odds else '(missing)'}")
        print(f"    td.Popular       → {td_popular.get_text(strip=True) if td_popular else '(missing)'}")

    # Also dump what fetch_entries_netkeiba actually returns (with caching etc.)
    print("\n── scraper.fetch_entries_netkeiba actual result ──")
    try:
        entries = scraper.fetch_entries_netkeiba(race_id)
        for e in entries:
            if horse_filter and horse_filter not in (e.get("name") or ""):
                continue
            print(f"  #{e.get('number'):>2}  {e.get('name','?'):<16}  "
                  f"odds={e.get('odds'):<6}  "
                  f"weight={e.get('horse_weight',''):<10}")
    except Exception as e:
        print(f"  fetch_entries_netkeiba failed: {e}")


def dump_odds_api(race_id: str) -> None:
    url = ODDS_API_URL.format(rid=race_id)
    try:
        payload = _get(url, as_json=True)
    except Exception as e:
        print(f"[deep] odds API fetch failed: {e}")
        return

    # Top-level keys
    print("\n[deep] API top-level keys:", list(payload.keys()))
    print(f"[deep] status:       {payload.get('status')!r}")
    print(f"[deep] update_count: {payload.get('update_count')!r}")
    print(f"[deep] reason:       {payload.get('reason')!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        print(f"[deep] data is not dict: type={type(data).__name__} value={data!r}")
        return
    print(f"[deep] data.official_datetime: {data.get('official_datetime')!r}")
    odds = data.get("odds") or {}
    print(f"[deep] data.odds keys:   {list(odds.keys())}")
    # 単勝 (key "1")
    tan = odds.get("1") or {}
    print(f"[deep] data.odds['1'] n horses: {len(tan)}")
    for um in sorted(tan.keys()):
        arr = tan[um]
        print(f"  馬番{um}:  arr={arr}  → arr[0]={arr[0] if arr else '?'}  "
              f"(this is what scraper parses as 単勝 odds)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--race-id", required=True)
    parser.add_argument("--save-html", action="store_true",
                        help="Save raw shutuba HTML to data/deep_inspect/")
    parser.add_argument("--horse",
                        help="Only show this horse (partial match)")
    parser.add_argument("--skip-api", action="store_true",
                        help="Don't call the odds JSON API")
    args = parser.parse_args()

    print(f"\n========= Deep-inspect race_id={args.race_id} =========")

    print("\n### 1. SHUTUBA HTML (per-row cell dump)")
    try:
        dump_shutuba(args.race_id, save_html=args.save_html,
                     horse_filter=args.horse)
    except Exception as e:
        print(f"[deep] shutuba dump failed: {e}")

    if not args.skip_api:
        print("\n### 2. ODDS JSON API (raw)")
        try:
            dump_odds_api(args.race_id)
        except Exception as e:
            print(f"[deep] api dump failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
