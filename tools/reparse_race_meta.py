"""Reparse existing race_meta files to fix surface detection bug."""

import json
import os
import re
from pathlib import Path

META_DIR = Path(__file__).resolve().parent.parent / "data" / "race_meta"

fixed = 0
for f in os.listdir(META_DIR):
    p = META_DIR / f
    with open(p, "r", encoding="utf-8") as fp:
        m = json.load(fp)
    raw = m.get("surface_raw", "")
    surf = "unknown"
    if "芝" in raw:
        surf = "turf"
    elif "ダ" in raw:
        surf = "dirt"
    if m.get("surface") != surf:
        m["surface"] = surf
        with open(p, "w", encoding="utf-8") as fp:
            json.dump(m, fp, ensure_ascii=False, indent=2)
        fixed += 1

print(f"Fixed: {fixed}")
