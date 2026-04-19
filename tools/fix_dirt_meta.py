"""Manually fix 15 dirt races that fetch_race_info_netkeiba failed on."""

import json
import os
from pathlib import Path

META_DIR = Path(__file__).resolve().parent.parent / "data" / "race_meta"

# race_id -> (surface, distance) for dirt races that netkeiba parse failed on
FIXES = {
    # 2024 races (added from expand)
    "202406050411": ("dirt", 1200),  # カペラS 2024
    "202407040211": ("dirt", 1800),  # チャンピオンズカップ 2024
    "202505010211": ("dirt", 1400),  # 根岸S 2025
    "202505010811": ("dirt", 1600),  # フェブラリーS 2025
    "202507010911": ("dirt", 1400),  # プロキオンS 2025
    # Original 15
    "202501010511": ("dirt", 1700),  # エルムS
    "202504020607": ("dirt", 1800),  # レパードS
    "202505050311": ("dirt", 1600),  # 武蔵野S
    "202506050411": ("dirt", 1200),  # カペラS
    "202507030207": ("dirt", 1900),  # 東海S
    "202507050211": ("dirt", 1800),  # チャンピオンズカップ
    "202508020311": ("dirt", 1600),  # ユニコーンS
    "202508020911": ("dirt", 1900),  # 平安S
    "202508040211": ("dirt", 1800),  # みやこS
    "202509020711": ("dirt", 1800),  # アンタレスS
    "202509040811": ("dirt", 2000),  # シリウスS
    "202605010211": ("dirt", 1400),  # 根岸S
    "202605010811": ("dirt", 1600),  # フェブラリーS
    "202606030211": ("dirt", 1800),  # マーチS
    "202608010911": ("dirt", 1400),  # プロキオンS
}

fixed = 0
for rid, (surf, dist) in FIXES.items():
    p = META_DIR / f"{rid}.json"
    if not p.exists():
        print(f"  skip: {p.name} not found")
        continue
    with open(p, "r", encoding="utf-8") as f:
        m = json.load(f)
    m["surface"] = surf
    m["distance"] = dist
    m["_note"] = "manually fixed (netkeiba parse returned empty)"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    fixed += 1

print(f"Fixed: {fixed}")
