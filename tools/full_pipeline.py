"""End-to-end: rebuild snapshots → predict → analyze.

Run this after refresh_horse_cache.py completes, to regenerate all
outputs based on the refreshed horse cache.

USAGE:
  python tools/full_pipeline.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str]) -> int:
    print(f"\n{'=' * 70}\n>>> {' '.join(cmd)}\n{'=' * 70}")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return r.returncode


def main() -> int:
    steps = [
        # 1. Rebuild all snapshots (reads the now-fresh horse cache)
        [sys.executable, "tools/build_snapshot.py", "--all"],
        # 2. Run predictions both modes for all snapshots
        [sys.executable, "tools/snapshot_predict.py", "--all",
         "--pedigree", "both"],
        # 3. Basic v1 vs v2 comparison
        [sys.executable, "tools/snapshot_compare.py"],
        # 4. Conditional pedigree analysis
        [sys.executable, "tools/analyze_pedigree_conditions.py"],
    ]

    for cmd in steps:
        rc = run(cmd)
        if rc != 0:
            print(f"[FATAL] step exited with {rc}: {cmd}")
            return rc

    print("\n[done] Full pipeline complete.")
    print("       See: data/pedigree_conditional_analysis.json")
    print("       See: data/snapshot_comparison_report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
