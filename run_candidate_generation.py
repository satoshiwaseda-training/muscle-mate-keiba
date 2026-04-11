"""Candidate generation pipeline — formula optimization variant system.

Usage:
    python run_candidate_generation.py

This pipeline evaluates score_runner() from train.py as a candidate
formula against the baseline. The baseline uses Gemini confidence;
the candidate uses structured pre-race features only.

Candidate types (one change at a time):
    - coefficient_only: adjust w_field, w_grade, w_consensus, overround
    - transform_added: apply log/sqrt to odds or field size
    - interaction_added: add one cross-term (e.g., grade * odds_rank)
    - conditional_adjustment: add one if/else rule

NEVER modifies:
    - weights.json
    - prepare.py
    - evaluator.py
"""

import json
import shutil
import inspect
from datetime import datetime
from pathlib import Path

from prepare import load_paired_data
from evaluator import (
    baseline_score_runner,
    evaluate_walk_forward,
    check_adoption,
)
from train import score_runner as candidate_score_runner


OUTPUT_DIR = Path(__file__).parent / "candidates"


def _count_complexity(source: str) -> dict:
    """Count complexity indicators in score_runner source code."""
    lines = source.strip().split("\n")
    code_lines = [l for l in lines if l.strip() and not l.strip().startswith("#")
                  and not l.strip().startswith('"""') and not l.strip().startswith("'''")]
    # Count W_ coefficient declarations (e.g., W_FIELD = 0.06)
    import re
    num_coefficients = len(re.findall(r"W_\w+\s*=", source))
    # Count ix_ interaction variables
    num_interactions = len(re.findall(r"ix_\w+\s*=", source))
    num_conditionals = sum(1 for l in code_lines if l.strip().startswith("if ") or l.strip().startswith("elif "))
    return {
        "code_lines": len(code_lines),
        "num_coefficients": num_coefficients,
        "num_interactions": num_interactions,
        "num_conditionals": num_conditionals,
    }


def _detect_gemini_usage(source: str) -> list[str]:
    """Detect if score_runner uses Gemini-derived fields."""
    violations = []
    # These fields are Gemini outputs and must NOT be used as primary inputs
    gemini_fields = [
        ("confidence", r'\.get\("confidence"'),
        ("ev_gap", r'\.get\("ev_gap"'),
        ("bet", r'\.get\("bet"'),
    ]
    for field_name, pattern in gemini_fields:
        import re
        # Check for actual usage (not just in comments or docstrings)
        for line in source.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'"):
                continue
            if f'.get("{field_name}"' in stripped or f'["{field_name}"]' in stripped:
                # Check it's not in a comment on same line
                code_part = stripped.split("#")[0]
                if f'"{field_name}"' in code_part:
                    violations.append(f"Uses Gemini field: {field_name} in: {stripped}")
    return violations


def run_pipeline(min_races: int = 300) -> dict:
    """Run the full candidate generation pipeline."""
    print("=" * 60)
    print("CANDIDATE GENERATION PIPELINE (Formula Optimization)")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 60)

    # Step 0: Pre-flight checks
    print("\n--- Step 0: Pre-flight Checks ---")

    # Check Gemini independence
    import train
    source = inspect.getsource(train.score_runner)
    gemini_violations = _detect_gemini_usage(source)
    if gemini_violations:
        print("FATAL: Candidate uses Gemini-derived fields:")
        for v in gemini_violations:
            print(f"  - {v}")
        print("DISCARD: Gemini dependency detected.")
        return {"status": "error", "reason": "Gemini dependency in score_runner"}
    print("  Gemini independence: PASS")

    # Check complexity
    complexity = _count_complexity(source)
    print(f"  Complexity: {complexity}")

    # Load data
    data = load_paired_data()
    print(f"  Paired races: {len(data)}")

    if not data:
        print("ERROR: No paired data available.")
        return {"status": "error", "reason": "No data"}

    # Step 1: Baseline evaluation (Gemini confidence passthrough)
    print("\n--- Step 1: Baseline Evaluation (Gemini confidence) ---")
    baseline = evaluate_walk_forward(baseline_score_runner, mask_odds=False)
    if baseline.get("error"):
        print(f"ERROR: {baseline['error']}")
        return {"status": "error", "reason": baseline["error"]}
    _print_metrics("Baseline", baseline)

    # Step 2: Candidate evaluation (structured formula)
    print("\n--- Step 2: Candidate Evaluation (structured formula) ---")
    candidate = evaluate_walk_forward(candidate_score_runner, mask_odds=False)
    if candidate.get("error"):
        print(f"ERROR: {candidate['error']}")
        return {"status": "error", "reason": candidate["error"]}
    _print_metrics("Candidate", candidate)

    # Step 3: Odds-masked evaluation
    print("\n--- Step 3: Odds-Masked Evaluation ---")
    candidate_masked = evaluate_walk_forward(candidate_score_runner, mask_odds=True)
    if candidate_masked.get("error"):
        print(f"WARNING: {candidate_masked['error']}")
    else:
        _print_metrics("Masked", candidate_masked)

    # Step 4: Adoption gate
    print("\n--- Step 4: Adoption Gate ---")
    adoption = check_adoption(candidate, baseline, candidate_masked, min_races=min_races)

    for check_name, check_result in adoption["checks"].items():
        status = "PASS" if check_result["passed"] else "FAIL"
        print(f"  [{status}] {check_name}: {check_result['details']}")

    # Step 5: Output
    print("\n" + "=" * 60)
    if adoption["adopted"]:
        print("RESULT: ALL CHECKS PASSED")
        _generate_candidate_files(baseline, candidate, candidate_masked, adoption, complexity)
        print(f"\nOutput: {OUTPUT_DIR}/")
        print("HUMAN APPROVAL REQUIRED before production merge.")
        return {"status": "adopted", "adoption": adoption}
    else:
        print(f"RESULT: {adoption['reason']}")
        # Still generate a diagnostic report (not candidate_logic.py)
        _generate_diagnostic_report(baseline, candidate, candidate_masked, adoption, complexity)
        return {"status": "discarded", "reason": adoption["reason"], "adoption": adoption}


def _print_metrics(label: str, metrics: dict):
    print(f"  {label} ROI:     {metrics['roi']:.4f}")
    print(f"  {label} Brier:   {metrics['brier']:.4f}")
    print(f"  {label} MaxDD:   {metrics['max_drawdown']:.4f}")
    print(f"  {label} Races:   {metrics['num_races']}")


def _generate_candidate_files(baseline, candidate, candidate_masked, adoption, complexity):
    """Generate candidate_logic.py + candidate_report.md on PASS."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Copy candidate logic
    train_path = Path(__file__).parent / "train.py"
    shutil.copy2(train_path, OUTPUT_DIR / f"candidate_logic_{ts}.py")
    shutil.copy2(train_path, OUTPUT_DIR / "candidate_logic.py")

    # Generate report
    report = _build_report(baseline, candidate, candidate_masked, adoption, complexity, ts, passed=True)
    (OUTPUT_DIR / f"candidate_report_{ts}.md").write_text(report, encoding="utf-8")
    (OUTPUT_DIR / "candidate_report.md").write_text(report, encoding="utf-8")


def _generate_diagnostic_report(baseline, candidate, candidate_masked, adoption, complexity):
    """Generate diagnostic report on FAIL (no candidate_logic.py)."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = _build_report(baseline, candidate, candidate_masked, adoption, complexity, ts, passed=False)
    (OUTPUT_DIR / f"diagnostic_{ts}.md").write_text(report, encoding="utf-8")


def _build_report(baseline, candidate, candidate_masked, adoption, complexity, ts, passed):
    """Build the candidate/diagnostic report."""
    import train
    source = inspect.getsource(train.score_runner)

    lines = []
    status_label = "ADOPTED" if passed else "DISCARDED (diagnostic only)"
    lines.append(f"# Candidate Report — {status_label}")
    lines.append(f"\nGenerated: {ts}")

    # Hypothesis
    lines.append("\n## Hypothesis")
    lines.append("")
    lines.append("### What changed")
    lines.append("Replaced Gemini-confidence-based scoring with a deterministic formula")
    lines.append("that computes win probability from structured pre-race features only:")
    lines.append("- **odds**: market-implied probability with overround correction")
    lines.append("- **field size**: base-rate calibration for number of runners")
    lines.append("- **grade**: G1/G2/G3 market efficiency proxy")
    lines.append("- **odds-rank consensus**: top pick's odds vs field average")
    lines.append("")
    lines.append("### Gemini fields excluded")
    lines.append("- `confidence`: NOT used (Gemini prediction output)")
    lines.append("- `ev_gap`: NOT used (Gemini expected-value calculation)")
    lines.append("- `bet`: NOT used (Gemini recommendation)")
    lines.append("")
    lines.append("### Why this should improve")
    lines.append("- Removes dependency on LLM accuracy for predictions")
    lines.append("- Makes scoring deterministic and reproducible")
    lines.append("- Enables true formula self-improvement via coefficient tuning")
    lines.append("- Odds-implied probability is a strong pre-race signal from aggregated market wisdom")

    # Complexity
    lines.append("\n## Complexity Analysis")
    lines.append("")
    lines.append(f"- Code lines: {complexity['code_lines']}")
    lines.append(f"- Coefficient terms: {complexity['num_coefficient_terms']}")
    lines.append(f"- Interaction operations: {complexity['num_interactions']}")
    lines.append(f"- Conditional branches: {complexity['num_conditionals']}")
    lines.append(f"- Candidate type: **coefficient_only + conditional_adjustment**")

    # Annual Matrix
    lines.append("\n## Annual Matrix")
    lines.append("")
    lines.append("| Year | Races | ROI (B) | ROI (C) | Brier (B) | Brier (C) | MaxDD (B) | MaxDD (C) | Status |")
    lines.append("|------|-------|---------|---------|-----------|-----------|-----------|-----------|--------|")
    b_yearly = baseline.get("yearly_results", {})
    c_yearly = candidate.get("yearly_results", {})
    for yr in sorted(set(list(b_yearly.keys()) + list(c_yearly.keys()))):
        b = b_yearly.get(yr, {})
        c = c_yearly.get(yr, {})
        races = c.get("num_races", b.get("num_races", 0))
        b_roi, c_roi = b.get("roi", 0), c.get("roi", 0)
        b_brier, c_brier = b.get("brier", 1), c.get("brier", 1)
        b_mdd, c_mdd = b.get("max_drawdown", 0), c.get("max_drawdown", 0)
        status = "PASS" if (c_roi >= b_roi and c_brier <= b_brier) else "FAIL"
        lines.append(f"| {yr} | {races} | {b_roi:.4f} | {c_roi:.4f} | {b_brier:.4f} | {c_brier:.4f} | {b_mdd:.4f} | {c_mdd:.4f} | {status} |")

    # Verification Proof
    lines.append("\n## Verification Proof")
    lines.append("")
    lines.append("### Overall Comparison")
    lines.append("")
    lines.append("| Metric | Baseline (Gemini) | Candidate (Formula) | Status |")
    lines.append("|--------|-------------------|---------------------|--------|")
    lines.append(f"| ROI | {baseline['roi']:.4f} | {candidate['roi']:.4f} | {'PASS' if candidate['roi'] >= baseline['roi'] else 'FAIL'} |")
    lines.append(f"| Brier | {baseline['brier']:.4f} | {candidate['brier']:.4f} | {'PASS' if candidate['brier'] <= baseline['brier'] else 'FAIL'} |")
    lines.append(f"| MaxDrawdown | {baseline['max_drawdown']:.4f} | {candidate['max_drawdown']:.4f} | {'PASS' if adoption['checks']['safety']['passed'] else 'FAIL'} |")
    lines.append(f"| Races | {baseline['num_races']} | {candidate['num_races']} | {'PASS' if adoption['checks']['significance']['passed'] else 'FAIL'} |")

    # Fold-level
    lines.append("")
    lines.append("### Fold-Level Results")
    lines.append("")
    lines.append("| Fold | Races | B-ROI | C-ROI | B-Brier | C-Brier |")
    lines.append("|------|-------|-------|-------|---------|---------|")
    b_folds = baseline.get("fold_results", [])
    c_folds = candidate.get("fold_results", [])
    for i in range(max(len(b_folds), len(c_folds))):
        bf = b_folds[i] if i < len(b_folds) else {}
        cf = c_folds[i] if i < len(c_folds) else {}
        lines.append(f"| {i} | {cf.get('num_races', bf.get('num_races', 0))} | "
                     f"{bf.get('roi', 0):.4f} | {cf.get('roi', 0):.4f} | "
                     f"{bf.get('brier', 1):.4f} | {cf.get('brier', 1):.4f} |")

    # Odds mask
    lines.append("")
    lines.append("### Odds Mask Test")
    lines.append("")
    lines.append("| Metric | Normal | Masked | Threshold | Status |")
    lines.append("|--------|--------|--------|-----------|--------|")
    if not candidate_masked.get("error"):
        lines.append(f"| ROI | {candidate['roi']:.4f} | {candidate_masked['roi']:.4f} | >= {baseline['roi'] * 0.9:.4f} | {'PASS' if candidate_masked['roi'] >= baseline['roi'] * 0.9 else 'FAIL'} |")
        lines.append(f"| Brier | {candidate['brier']:.4f} | {candidate_masked['brier']:.4f} | <= {baseline['brier'] * 1.05:.4f} | {'PASS' if candidate_masked['brier'] <= baseline['brier'] * 1.05 else 'FAIL'} |")

    # Gate summary
    lines.append("")
    lines.append("### Adoption Gate")
    lines.append("")
    for name, check in adoption["checks"].items():
        lines.append(f"- **[{'PASS' if check['passed'] else 'FAIL'}] {name}**: {check['details']}")

    # Risk
    lines.append("\n## Risk Disclosure")
    lines.append("")
    lines.append("### Odds dependency")
    lines.append("- Primary signal is odds-implied probability. Without odds, falls back to uniform rate.")
    lines.append("- Anti-bias mask test verifies the formula isn't 100% odds-dependent.")
    lines.append("")
    lines.append("### Feature poverty")
    lines.append("- Current prediction format does not persist structured features from scraper.")
    lines.append("- Only odds, grade, field size, and ordinal position are available.")
    lines.append("- Future improvement: persist horse_weight, jockey_win_rate, training_physics, etc.")
    lines.append("")
    lines.append("### Overfitting risk")
    lines.append("- Low: only 4 coefficients (w_field, w_grade, w_consensus, overround).")
    lines.append("- No learned parameters; coefficients are set by domain knowledge.")
    lines.append("")
    lines.append("### Data volume")
    lines.append(f"- Currently {candidate['num_races']} races. Need 300+ for significance gate.")

    # Next candidates
    lines.append("\n## Suggested Next Candidates")
    lines.append("")
    lines.append("1. **coefficient_only**: tune overround (1.15-1.30), w_field (0.05-0.12)")
    lines.append("2. **transform_added**: log(odds) instead of 1/odds for base probability")
    lines.append("3. **interaction_added**: grade * consensus_ratio cross-term")
    lines.append("4. **conditional_adjustment**: separate coefficients for turf vs dirt")

    # Source
    lines.append("\n## Candidate Source")
    lines.append("")
    lines.append("```python")
    lines.append(source)
    lines.append("```")

    lines.append("\n---")
    if passed:
        lines.append("**HUMAN APPROVAL REQUIRED before merging to production.**")
    else:
        lines.append("**DISCARDED — diagnostic report only. No candidate files generated.**")

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_pipeline()
    print(f"\nPipeline result: {result['status']}")
