import sys
import shutil
import uuid
import zipfile
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import prediction_log as plog


def _workspace_tmp() -> Path:
    base = Path(__file__).resolve().parent.parent / ".test-output" / "prediction_archive"
    path = base / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sample_prediction(race_id: str = "202604250101") -> dict:
    return {
        "race_id": race_id,
        "race_name": "Codex Stakes",
        "grade": "G3",
        "venue": "Tokyo",
        "race_date": "2026-04-25",
        "is_live": True,
        "prediction_stage": "final",
        "prediction_created_at": "2026-04-25T09:00:00",
        "odds_status_at_prediction": "ok",
        "data_source_version": "test",
        "ranked": [
            {"name": "Alpha", "odds": 2.5, "win_prob": 0.32, "mode": "odds", "odds_score": 75},
            {"name": "Bravo", "odds": 5.1, "win_prob": 0.18, "mode": "odds", "odds_score": 61},
            {"name": "Charlie", "odds": 8.0, "win_prob": 0.12, "mode": "odds", "odds_score": 55},
        ],
        "selected_top3": [
            {"name": "Alpha", "odds": 2.5, "win_prob": 0.32},
            {"name": "Bravo", "odds": 5.1, "win_prob": 0.18},
            {"name": "Charlie", "odds": 8.0, "win_prob": 0.12},
        ],
        "prediction_variants": {
            "primary": {
                "candidate_id": "primary_current",
                "strategy": "win_prob",
                "strategy_version": "test-primary",
                "top3": [
                    {"name": "Alpha", "odds": 2.5, "win_prob": 0.32},
                    {"name": "Bravo", "odds": 5.1, "win_prob": 0.18},
                    {"name": "Charlie", "odds": 8.0, "win_prob": 0.12},
                ],
            },
            "experimental": {
                "candidate_id": "experimental_jockey_trainer",
                "strategy": "feature_only_jockey_trainer_combo",
                "strategy_version": "test-experimental",
                "top3": [
                    {
                        "name": "Delta",
                        "odds": 6.0,
                        "win_prob": 0.16,
                        "market_rank": 2,
                        "experimental_score": 0.0702,
                    },
                    {
                        "name": "Echo",
                        "odds": 9.0,
                        "win_prob": 0.10,
                        "market_rank": 5,
                        "experimental_score": 0.0401,
                    },
                    {
                        "name": "Foxtrot",
                        "odds": 12.0,
                        "win_prob": 0.08,
                        "market_rank": 8,
                        "experimental_score": 0.0301,
                    },
                ],
            },
        },
        "loose_bets": [
            {
                "name": "Alpha",
                "odds": 2.5,
                "consensus_count": 1,
                "composite_condition": 0.7,
                "loose_trigger_reason": "test",
            }
        ],
        "loose_rule_version": "cons>=1_comp>=0.60_odds<=15_no_strongneg_v1",
    }


def test_store_prediction_exports_review_archive():
    old_live = plog.LIVE_FILE
    old_archive = plog.ARCHIVE_DIR
    tmp_path = _workspace_tmp()
    try:
        plog.LIVE_FILE = tmp_path / "live_predictions.json"
        plog.ARCHIVE_DIR = tmp_path / "prediction_archive"

        plog.store_prediction(_sample_prediction())

        race_dir = plog.ARCHIVE_DIR / "2026-04-25" / "202604250101_Codex_Stakes"
        assert (race_dir / "latest.json").exists()
        assert (race_dir / "latest.md").exists()
        assert list((race_dir / "snapshots").glob("*.json"))
        assert (plog.ARCHIVE_DIR / "2026-04-25" / "index.csv").exists()
        assert (plog.ARCHIVE_DIR / "index.csv").exists()

        md = (race_dir / "latest.md").read_text(encoding="utf-8")
        assert "Codex Stakes" in md
        assert "Alpha" in md
        assert "Experimental Candidate Top 3" in md
        assert "Delta" in md
        assert "Loose Bets" in md

        rows = plog.recent_prediction_archive_table(limit=10)
        assert rows[0]["race_id"] == "202604250101"
        assert rows[0]["top1"] == "Alpha"
        assert rows[0]["experimental_top1"] == "Delta"
    finally:
        plog.LIVE_FILE = old_live
        plog.ARCHIVE_DIR = old_archive
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_attach_result_refreshes_archive_markdown():
    old_live = plog.LIVE_FILE
    old_archive = plog.ARCHIVE_DIR
    tmp_path = _workspace_tmp()
    try:
        plog.LIVE_FILE = tmp_path / "live_predictions.json"
        plog.ARCHIVE_DIR = tmp_path / "prediction_archive"

        plog.store_prediction(_sample_prediction("202604250102"))
        assert plog.attach_result(
            "202604250102",
            {
                "finishing_order": [{"rank": 1, "name": "Alpha", "odds": 2.5}],
                "payouts": {"単勝": 250},
            },
        )

        race_dir = plog.ARCHIVE_DIR / "2026-04-25" / "202604250102_Codex_Stakes"
        md = (race_dir / "latest.md").read_text(encoding="utf-8")
        assert "## Result" in md
        assert "winner: **Alpha**" in md
    finally:
        plog.LIVE_FILE = old_live
        plog.ARCHIVE_DIR = old_archive
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_build_prediction_archive_zip_contains_review_files():
    payload = plog.build_prediction_archive_zip([_sample_prediction()])
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = set(zf.namelist())
        assert "prediction_archive/index.csv" in names
        assert (
            "prediction_archive/2026-04-25/"
            "202604250101_Codex_Stakes/latest.json"
        ) in names
        assert (
            "prediction_archive/2026-04-25/"
            "202604250101_Codex_Stakes/latest.md"
        ) in names
        md = zf.read(
            "prediction_archive/2026-04-25/"
            "202604250101_Codex_Stakes/latest.md"
        ).decode("utf-8")
        assert "Codex Stakes" in md
        assert "Experimental Candidate Top 3" in md
        assert "Loose Bets" in md
