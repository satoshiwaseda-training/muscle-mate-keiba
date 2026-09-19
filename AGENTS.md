# Project Constitution Reference

This project has a **live trading constitution** that governs all
production-facing changes:

**`docs/trading_constitution.md`** — MUST be read before touching any of:

- `dual_mode_scoring.py` — especially `trigger_loose_capped`, `LOOSE_*`
- `live_pipeline.py`
- `prediction_log.py` — loose KPI block
- `trigger_evaluation.py`
- `app_live.py` — Loose Trigger panel

## Non-negotiable rules (from the constitution)

1. **ROI is the only success metric.** Not hit rate, not trigger count.
2. **STRICT and LOOSE are two independent layers** — do not mix their
   KPIs, do not let one's behavior justify changing the other.
3. **The LOOSE rule is frozen at:**
   ```
   consensus_count >= 1
   AND composite_condition >= 0.60
   AND odds <= 15.0
   AND strong_negative_present == False
   ```
   Rule version: `cons>=1_comp>=0.60_odds<=15_no_strongneg_v1`.
4. **Do not tune thresholds to chase short-term performance.** The rule
   requires ≥100 bets before its ROI is interpretable.
5. **Log every loose bet** with rule version + full metadata.
   See `prediction_log.recent_loose_bets_table`.
6. **Never touch** `evaluator.py`, `train.py`, or `probability_engine.py`
   for trading-strategy reasons. Those are model-side files.

When in doubt, re-read `docs/trading_constitution.md` — section 8 is the
summary:

> **「ルールを守れるかどうかが、勝てるかどうか」**
