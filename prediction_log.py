"""Live prediction persistence + trigger KPIs.

File: data/live_predictions.json — one entry per race keyed by race_id.

Each entry holds the full dual-mode prediction, the per-horse trigger
table, and (after the race runs) the attached result block with winner
and payouts. KPIs are computed on demand.
"""

from __future__ import annotations

import json
import os
import csv
import io
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent
LIVE_FILE = ROOT / "data" / "live_predictions.json"
ARCHIVE_DIR = ROOT / "data" / "prediction_archive"
JOURNAL_DIR = ROOT / "data" / "prediction_journal"
RECOVERY_DIR = ROOT / "data" / "recovery"
PERSISTENCE_STATUS_FILE = ROOT / "data" / "persistence_status.json"


def _now_stamp() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_stamp(value: str | None = None) -> str:
    raw = value or _now_stamp()
    return re.sub(r"[^0-9A-Za-z_-]+", "", raw.replace(":", "").replace("-", ""))


def _backup_corrupt_live_file(reason: str) -> Optional[Path]:
    """Move an unreadable canonical log aside before recovery.

    Returning `{}` on JSON decode failure is convenient but dangerous:
    the next save would overwrite the corrupt file and hide the evidence.
    Keep a timestamped copy under data/recovery so manual recovery remains
    possible even when automatic archive rebuild succeeds.
    """
    if not LIVE_FILE.exists():
        return None
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    dest = RECOVERY_DIR / f"corrupt_live_predictions_{_safe_stamp()}_{reason}.json"
    try:
        dest.write_bytes(LIVE_FILE.read_bytes())
        return dest
    except Exception:
        return None


def _rebuild_live_from_archive() -> dict:
    """Reconstruct the latest per-race log from prediction_archive/latest.json."""
    if not ARCHIVE_DIR.exists():
        return {}
    rebuilt = {}
    latest_files = sorted(
        ARCHIVE_DIR.glob("*/*/latest.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
    )
    for latest in latest_files:
        try:
            entry = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            continue
        rid = entry.get("race_id")
        if rid:
            rebuilt[rid] = entry
    return rebuilt


def _load() -> dict:
    if not LIVE_FILE.exists():
        rebuilt = _rebuild_live_from_archive()
        if rebuilt:
            _save(rebuilt)
        return rebuilt
    try:
        return json.loads(LIVE_FILE.read_text(encoding="utf-8"))
    except Exception:
        _backup_corrupt_live_file("decode")
        rebuilt = _rebuild_live_from_archive()
        if rebuilt:
            _save(rebuilt)
        return rebuilt


def _save(data: dict) -> None:
    """Atomically persist the prediction log.

    Writes to a sibling `.tmp` file first, fsyncs it, then uses
    `os.replace` to swap it into place. `os.replace` is atomic on both
    POSIX and Windows (Python 3.3+), so readers either see the previous
    complete file or the new complete file — never a truncated one.

    This matters because store_prediction is called inside a per-race
    loop that may be killed (Task Scheduler timeout, Ctrl-C, power loss).
    Without atomic write, a kill mid-save would leave the file as the
    partially-written new bytes and every subsequent run would fail to
    decode it — losing the whole week of logs.
    """
    LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = LIVE_FILE.with_name(LIVE_FILE.name + ".tmp")
    # Serialize first so a JSON encoding error doesn't even touch disk
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    # Write + fsync so the bytes actually reach the platter/SSD before
    # we rename. On Windows this is also honoured by os.fsync.
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(payload)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass  # some filesystems (network, OneDrive) don't support fsync
    try:
        os.replace(tmp_path, LIVE_FILE)
    except PermissionError:
        # OneDrive / WindowsApps sandboxes can allow the temp write but
        # deny the atomic rename. Fall back to a direct complete write so
        # logging still works; the JSON was already fully serialized above.
        LIVE_FILE.write_text(payload, encoding="utf-8", newline="\n")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_text_atomic(path: Path, text: str) -> None:
    """Write text using the same replace-on-complete pattern as _save."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        path.write_text(text, encoding="utf-8", newline="\n")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_json_atomic(path: Path, payload: dict) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))


def _append_journal(entry: dict, event: str) -> Optional[Path]:
    """Append an immutable JSONL event for audit/recovery.

    Folder layout:
      data/prediction_journal/YYYY-MM-DD/events.jsonl

    Unlike live_predictions.json, this file is append-only. Even if a later
    overwrite or archive export fails, each save/result attachment leaves a
    recoverable event trail.
    """
    if not entry or not entry.get("race_id"):
        return None
    race_date = _safe_name(entry.get("race_date"), "unknown-date")
    path = JOURNAL_DIR / race_date / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "saved_at": _now_stamp(),
        "race_id": entry.get("race_id"),
        "race_date": entry.get("race_date"),
        "race_name": entry.get("race_name"),
        "prediction_stage": entry.get("prediction_stage"),
        "snapshot_at": entry.get("snapshot_at"),
        "entry": entry,
    }
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    return path


def _write_persistence_status(status: dict) -> None:
    """Best-effort status marker for Streamlit/sidebar diagnostics."""
    try:
        _write_json_atomic(PERSISTENCE_STATUS_FILE, status)
    except Exception:
        pass


def _safe_name(value: object, fallback: str = "unknown") -> str:
    raw = str(value or "").strip() or fallback
    raw = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", raw)
    raw = re.sub(r"\s+", "_", raw)
    raw = raw.strip("._ ")
    return raw[:80] or fallback


def _archive_slug(entry: dict) -> str:
    race_id = _safe_name(entry.get("race_id"), "race")
    race_name = _safe_name(entry.get("race_name"), "prediction")
    return f"{race_id}_{race_name}"


def _archive_timestamp(entry: dict) -> str:
    raw = (
        entry.get("snapshot_at")
        or entry.get("prediction_created_at")
        or datetime.now().isoformat(timespec="seconds")
    )
    return _safe_name(str(raw).replace(":", "").replace("-", ""), "snapshot")


def _fmt_pct(value: object) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_odds(value: object) -> str:
    try:
        v = float(value or 0)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.1f}" if v > 0 else "-"


def _prediction_markdown(entry: dict) -> str:
    """Human-readable race snapshot for later review."""
    title = entry.get("race_name") or entry.get("race_id") or "Prediction"
    lines = [
        f"# {title}",
        "",
        f"- race_id: `{entry.get('race_id', '')}`",
        f"- race_date: `{entry.get('race_date', '')}`",
        f"- venue: `{entry.get('venue', '')}`",
        f"- grade: `{entry.get('grade', '')}`",
        f"- prediction_stage: `{entry.get('prediction_stage', '')}`",
        f"- prediction_created_at: `{entry.get('prediction_created_at', '')}`",
        f"- snapshot_at: `{entry.get('snapshot_at', '')}`",
        f"- data_source_version: `{entry.get('data_source_version', '')}`",
        f"- odds_status_at_prediction: `{entry.get('odds_status_at_prediction', entry.get('odds_status', ''))}`",
        "",
        "## Primary Candidate Top 3",
        "",
        "| Mark | Horse | Odds | Win Prob |",
        "|---|---:|---:|---:|",
    ]
    marks = ["◎", "○", "▲"]
    for i, horse in enumerate(entry.get("selected_top3") or []):
        lines.append(
            "| {mark} | {name} | {odds} | {prob} |".format(
                mark=marks[i] if i < len(marks) else str(i + 1),
                name=horse.get("name", ""),
                odds=_fmt_odds(horse.get("odds")),
                prob=_fmt_pct(horse.get("win_prob")),
            )
        )

    variants = entry.get("prediction_variants") or {}
    experimental = (variants.get("experimental") or {}).get("top3") or []
    if experimental:
        lines.extend([
            "",
            "## Experimental Candidate Top 3",
            "",
            f"- strategy: `{(variants.get('experimental') or {}).get('strategy', '')}`",
            f"- strategy_version: `{(variants.get('experimental') or {}).get('strategy_version', '')}`",
            "",
            "| Mark | Horse | Odds | Win Prob | Jockey WR | Trainer WR | Exp Score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ])
        for i, horse in enumerate(experimental):
            lines.append(
                "| {mark} | {name} | {odds} | {prob} | {jwr} | {twr} | {score} |".format(
                    mark=marks[i] if i < len(marks) else str(i + 1),
                    name=horse.get("name", ""),
                    odds=_fmt_odds(horse.get("odds")),
                    prob=_fmt_pct(horse.get("win_prob")),
                    jwr=_fmt_pct(horse.get("jockey_win_rate")),
                    twr=_fmt_pct(horse.get("trainer_win_rate")),
                    score=f"{float(horse.get('experimental_score', 0) or 0):.4f}",
                )
            )

    lines.extend([
        "",
        "## Loose Bets",
        "",
        "| Horse | Odds | Consensus | Composite | Reason | Rule Version |",
        "|---|---:|---:|---:|---|---|",
    ])
    loose = entry.get("loose_bets") or []
    if loose:
        for bet in loose:
            lines.append(
                "| {name} | {odds} | {cons} | {comp} | {reason} | `{version}` |".format(
                    name=bet.get("name", ""),
                    odds=_fmt_odds(bet.get("odds")),
                    cons=bet.get("consensus_count", ""),
                    comp=f"{float(bet.get('composite_condition', 0) or 0):.2f}",
                    reason=str(bet.get("loose_trigger_reason", "")).replace("|", "/"),
                    version=entry.get("loose_rule_version", ""),
                )
            )
    else:
        lines.append("| - | - | - | - | No loose bet candidates | - |")

    lines.extend([
        "",
        "## Ranking Top 8",
        "",
        "| Rank | Horse | Odds | Win Prob | Mode | Internal Score |",
        "|---:|---|---:|---:|---|---:|",
    ])
    for i, horse in enumerate((entry.get("ranked") or [])[:8], start=1):
        lines.append(
            "| {rank} | {name} | {odds} | {prob} | {mode} | {score} |".format(
                rank=i,
                name=horse.get("name", ""),
                odds=_fmt_odds(horse.get("odds")),
                prob=_fmt_pct(horse.get("win_prob")),
                mode=horse.get("mode", ""),
                score=f"{float(horse.get('odds_score', 0) or 0):.1f}",
            )
        )

    result = entry.get("result") or {}
    fo = result.get("finishing_order") or []
    if fo:
        winner = _race_winner(fo) or "-"
        lines.extend([
            "",
            "## Result",
            "",
            f"- winner: **{winner}**",
            f"- win payout: `{(result.get('payouts') or {}).get('単勝', '')}`",
        ])

    lines.append("")
    return "\n".join(lines)


def _archive_summary_row(entry: dict, md_path: Path) -> dict:
    ranked = entry.get("ranked") or []
    selected = entry.get("selected_top3") or []
    top_source = selected or ranked
    variants = entry.get("prediction_variants") or {}
    exp_top3 = (variants.get("experimental") or {}).get("top3") or []
    return {
        "race_date": entry.get("race_date", ""),
        "race_id": entry.get("race_id", ""),
        "race_name": entry.get("race_name", ""),
        "venue": entry.get("venue", ""),
        "grade": entry.get("grade", ""),
        "prediction_stage": entry.get("prediction_stage", ""),
        "prediction_created_at": entry.get("prediction_created_at", ""),
        "snapshot_at": entry.get("snapshot_at", ""),
        "loose_bet_count": len(entry.get("loose_bets") or []),
        "top1": (top_source[0].get("name", "") if len(top_source) > 0 else ""),
        "top2": (top_source[1].get("name", "") if len(top_source) > 1 else ""),
        "top3": (top_source[2].get("name", "") if len(top_source) > 2 else ""),
        "experimental_top1": (exp_top3[0].get("name", "") if len(exp_top3) > 0 else ""),
        "experimental_top2": (exp_top3[1].get("name", "") if len(exp_top3) > 1 else ""),
        "experimental_top3": (exp_top3[2].get("name", "") if len(exp_top3) > 2 else ""),
        "has_result": bool((entry.get("result") or {}).get("finishing_order")),
        "archive_markdown": str(md_path),
    }


def _write_index(path: Path, rows: list[dict]) -> None:
    columns = [
        "race_date", "race_id", "race_name", "venue", "grade",
        "prediction_stage", "prediction_created_at", "snapshot_at",
        "loose_bet_count", "top1", "top2", "top3",
        "experimental_top1", "experimental_top2", "experimental_top3",
        "has_result",
        "archive_markdown",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})
        f.flush()
        try:
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    try:
        os.replace(tmp_path, path)
    except PermissionError:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass


def rebuild_prediction_archive_index() -> list[dict]:
    """Rebuild date-level and global archive CSV indexes from latest.json."""
    rows = []
    if not ARCHIVE_DIR.exists():
        return []
    by_date: dict[str, list[dict]] = defaultdict(list)
    for latest in ARCHIVE_DIR.glob("*/*/latest.json"):
        try:
            entry = json.loads(latest.read_text(encoding="utf-8"))
        except Exception:
            continue
        md_path = latest.with_name("latest.md")
        row = _archive_summary_row(entry, md_path)
        rows.append(row)
        by_date[str(row.get("race_date") or "unknown")].append(row)

    rows.sort(
        key=lambda r: (r.get("race_date", ""), r.get("snapshot_at", ""), r.get("race_id", "")),
        reverse=True,
    )
    for race_date, date_rows in by_date.items():
        date_rows.sort(key=lambda r: (r.get("snapshot_at", ""), r.get("race_id", "")), reverse=True)
        _write_index(ARCHIVE_DIR / race_date / "index.csv", date_rows)
    _write_index(ARCHIVE_DIR / "index.csv", rows)
    return rows


def export_prediction_archive(entry: dict) -> Optional[dict]:
    """Export the latest prediction into review-friendly folder files.

    The canonical machine log remains data/live_predictions.json. This
    archive is a convenience layer for humans: date folder, race folder,
    latest JSON/Markdown, immutable snapshot copies, and CSV indexes.
    """
    if not entry or not entry.get("race_id"):
        return None

    race_date = _safe_name(entry.get("race_date"), "unknown-date")
    race_dir = ARCHIVE_DIR / race_date / _archive_slug(entry)
    snap_dir = race_dir / "snapshots"
    stamp = _archive_timestamp(entry)
    md = _prediction_markdown(entry)

    latest_json = race_dir / "latest.json"
    latest_md = race_dir / "latest.md"
    snapshot_json = snap_dir / f"{stamp}.json"
    snapshot_md = snap_dir / f"{stamp}.md"

    _write_json_atomic(latest_json, entry)
    _write_text_atomic(latest_md, md)
    _write_json_atomic(snapshot_json, entry)
    _write_text_atomic(snapshot_md, md)
    rebuild_prediction_archive_index()

    return {
        "race_dir": str(race_dir),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
        "snapshot_json": str(snapshot_json),
        "snapshot_markdown": str(snapshot_md),
    }


def recent_prediction_archive_table(limit: int = 50) -> list[dict]:
    rows = rebuild_prediction_archive_index()
    return rows[:limit]


def build_prediction_archive_zip(predictions: list[dict]) -> bytes:
    """Return a ZIP containing review files for the given predictions.

    This is primarily for Streamlit Cloud: the app cannot write directly
    to a user's local PC, but it can generate a downloadable archive with
    the same JSON/Markdown/index files that local runs persist under
    data/prediction_archive.
    """
    rows = []
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for entry in predictions or []:
            if not entry or not entry.get("race_id"):
                continue
            race_date = _safe_name(entry.get("race_date"), "unknown-date")
            race_slug = _archive_slug(entry)
            stamp = _archive_timestamp(entry)
            base = f"prediction_archive/{race_date}/{race_slug}"
            md_text = _prediction_markdown(entry)
            json_text = json.dumps(entry, ensure_ascii=False, indent=2)

            zf.writestr(f"{base}/latest.json", json_text)
            zf.writestr(f"{base}/latest.md", md_text)
            zf.writestr(f"{base}/snapshots/{stamp}.json", json_text)
            zf.writestr(f"{base}/snapshots/{stamp}.md", md_text)
            rows.append(_archive_summary_row(entry, Path(base) / "latest.md"))

        columns = [
            "race_date", "race_id", "race_name", "venue", "grade",
            "prediction_stage", "prediction_created_at", "snapshot_at",
            "loose_bet_count", "top1", "top2", "top3",
            "experimental_top1", "experimental_top2", "experimental_top3",
            "has_result",
            "archive_markdown",
        ]
        csv_buf = io.StringIO()
        writer = csv.DictWriter(csv_buf, fieldnames=columns)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda r: (
                r.get("race_date", ""),
                r.get("snapshot_at", ""),
                r.get("race_id", ""),
            ),
            reverse=True,
        ):
            writer.writerow({c: row.get(c, "") for c in columns})
        zf.writestr("prediction_archive/index.csv", csv_buf.getvalue())

    return buf.getvalue()


# ── Public API ────────────────────────────────────────

_HISTORY_CAP = 20


def _snapshot_of(entry: dict) -> dict:
    """Compact snapshot of a prior prediction for audit history.

    Intentionally lightweight so the live_predictions.json file does
    not blow up when a race is re-predicted many times during the day
    (e.g. 09:00 early → 11:00 early → 13:30 final, plus recovery runs).
    Everything here must be interpretable on its own without needing
    to look up a separate file.
    """
    ranked = entry.get("ranked") or []
    loose = entry.get("loose_bets") or []
    meta = entry.get("odds_api_meta") or {}
    return {
        "snapshot_at":               entry.get("snapshot_at"),
        "prediction_created_at":     entry.get("prediction_created_at"),
        "prediction_stage":          entry.get("prediction_stage"),
        "odds_status":               entry.get("odds_status"),
        "odds_status_at_prediction": entry.get("odds_status_at_prediction"),
        "odds_updated_at":           entry.get("odds_updated_at"),
        "data_source_version":       entry.get("data_source_version"),
        "calibration_k":             entry.get("calibration_k"),
        # Top-5 ranking digest — enough to diff two versions without
        # carrying 18-horse payloads per snapshot
        "top5": [
            {
                "name":     r.get("name"),
                "odds":     r.get("odds"),
                "win_prob": r.get("win_prob"),
                "fact_edge": r.get("fact_edge"),
            }
            for r in ranked[:5]
        ],
        "prediction_variants": {
            key: {
                "candidate_id": (value or {}).get("candidate_id"),
                "strategy": (value or {}).get("strategy"),
                "strategy_version": (value or {}).get("strategy_version"),
                "top3_names": [
                    h.get("name") for h in ((value or {}).get("top3") or [])[:3]
                ],
            }
            for key, value in (entry.get("prediction_variants") or {}).items()
        },
        "loose_bet_names": [b.get("name") for b in loose],
        "loose_bet_count": len(loose),
        "triggers_count":  len(entry.get("triggers") or []),
        "odds_api_meta":   meta,
        # Multi-source consensus digest (2026-04)
        "consensus_primary_source":   meta.get("consensus_primary_source"),
        "consensus_has_disagreement": meta.get("consensus_has_disagreement"),
        "consensus_summary":          meta.get("consensus_summary"),
    }


def store_prediction(prediction: dict) -> None:
    """Persist a prediction keyed by race_id.

    History preservation (ADDED 2026-04):
      - `first_predicted_at` is copied from the earliest save and
        never overwritten, so audits can tell when the race was first
        analysed regardless of how many re-runs happened afterwards.
      - A compact snapshot of the PREVIOUS version is appended to
        `history` before overwriting. This means the first re-run
        after initial save produces history=[initial_snapshot], the
        second produces history=[initial, first-rerun], etc.
      - `history` is capped at `_HISTORY_CAP` entries (oldest dropped).
      - Cross-run invariants: `result` (ground truth), `first_predicted_at`,
        and `history` are NEVER dropped by a re-save of the same race_id.
    """
    if not prediction or "race_id" not in prediction:
        return
    data = _load()
    rid = prediction["race_id"]
    existing = data.get(rid) or {}
    existing_result = existing.get("result")

    entry = dict(prediction)

    # Preserve the earliest timestamp across re-runs
    first_seen = existing.get("first_predicted_at") or entry.get("prediction_created_at")
    if first_seen:
        entry["first_predicted_at"] = first_seen

    # Snapshot the previous version into history BEFORE we overwrite.
    history = list(existing.get("history") or [])
    if existing.get("prediction_created_at") or existing.get("created_at"):
        # Make sure the existing entry has a snapshot_at for clarity
        if "snapshot_at" not in existing:
            existing["snapshot_at"] = (
                existing.get("prediction_created_at")
                or existing.get("created_at")
            )
        history.append(_snapshot_of(existing))
        if len(history) > _HISTORY_CAP:
            history = history[-_HISTORY_CAP:]
    entry["history"] = history

    # `snapshot_at` marks when THIS record was saved (distinct from
    # prediction_created_at which marks when it was computed — typically
    # the same but we keep the distinction for future async workflows).
    entry["snapshot_at"] = datetime.now().isoformat(timespec="seconds")

    # Never drop an attached result on re-save
    if existing_result:
        entry["result"] = existing_result
    else:
        entry.setdefault("result", None)

    data[rid] = entry
    _save(data)
    journal_path = None
    archive_info = None
    archive_error = None
    gist_ok = None
    try:
        journal_path = _append_journal(entry, "store_prediction")
    except Exception as e:
        archive_error = f"journal: {e}"
    try:
        archive_info = export_prediction_archive(entry)
    except Exception as e:
        # The canonical log must remain the source of truth even if the
        # human-readable archive cannot be written (OneDrive lock, etc.).
        archive_error = f"{archive_error}; archive: {e}" if archive_error else f"archive: {e}"
    # v5.9: live_predictions.json を Gist に push (Streamlit Cloud
    # ephemeral filesystem 対策)。failure は silent — ログの正本はあくまで
    # local file。Gist 認証無し環境では _available() が False で no-op。
    try:
        import github_sync as _gs
        if _gs._available():
            gist_ok = _gs.push_file("live_predictions.json", data)
        else:
            gist_ok = False
    except Exception:
        gist_ok = False
    _write_persistence_status({
        "last_event": "store_prediction",
        "last_saved_at": _now_stamp(),
        "race_id": rid,
        "canonical_file": str(LIVE_FILE),
        "journal_file": str(journal_path) if journal_path else "",
        "archive_latest_json": (archive_info or {}).get("latest_json", ""),
        "archive_latest_markdown": (archive_info or {}).get("latest_markdown", ""),
        "archive_error": archive_error or "",
        "gist_sync_ok": gist_ok,
    })


def attach_result(race_id: str, result: dict) -> bool:
    """Attach the actual race outcome to a logged prediction.

    `result` shape (can be produced from scraper.fetch_result_netkeiba):
      {
        "finishing_order": [{"rank": 1, "name": "...", "odds": 2.5}, ...],
        "payouts": {"単勝": int, "馬連": int, ...}
      }
    """
    data = _load()
    if race_id not in data:
        return False
    entry = data[race_id]
    entry["result"] = result
    entry["result_attached_at"] = datetime.now().isoformat()
    data[race_id] = entry
    _save(data)
    journal_path = None
    archive_info = None
    archive_error = None
    gist_ok = None
    try:
        journal_path = _append_journal(entry, "attach_result")
    except Exception as e:
        archive_error = f"journal: {e}"
    try:
        archive_info = export_prediction_archive(entry)
    except Exception as e:
        archive_error = f"{archive_error}; archive: {e}" if archive_error else f"archive: {e}"
    # v5.9: Gist sync (Streamlit Cloud ephemeral filesystem 対策)
    try:
        import github_sync as _gs
        if _gs._available():
            gist_ok = _gs.push_file("live_predictions.json", data)
        else:
            gist_ok = False
    except Exception:
        gist_ok = False
    _write_persistence_status({
        "last_event": "attach_result",
        "last_saved_at": _now_stamp(),
        "race_id": race_id,
        "canonical_file": str(LIVE_FILE),
        "journal_file": str(journal_path) if journal_path else "",
        "archive_latest_json": (archive_info or {}).get("latest_json", ""),
        "archive_latest_markdown": (archive_info or {}).get("latest_markdown", ""),
        "archive_error": archive_error or "",
        "gist_sync_ok": gist_ok,
    })
    return True


def list_predictions(
    only_with_result: bool = False,
    only_live: bool = False,
    since: Optional[str] = None,
) -> list[dict]:
    data = _load()
    out = []
    for rid, entry in data.items():
        if only_with_result and not entry.get("result"):
            continue
        if only_live and not entry.get("is_live"):
            continue
        if since and entry.get("race_date", "") < since:
            continue
        out.append(entry)
    out.sort(key=lambda e: e.get("race_date", ""), reverse=True)
    return out


def get_prediction(race_id: str) -> Optional[dict]:
    return _load().get(race_id)


# ── KPI computation ────────────────────────────────────

def _norm(n): return (n or "").strip()


def _parse_odds(raw) -> float:
    try:
        return float(str(raw).replace("---", "0").replace("--", "0"))
    except (ValueError, TypeError):
        return 0.0


def _race_winner(fo: list) -> Optional[str]:
    for h in fo or []:
        try:
            if int(h.get("rank", 0) or 0) == 1:
                return _norm(h.get("name"))
        except ValueError:
            continue
    return None


def compute_kpis(preds: Optional[list[dict]] = None) -> dict:
    """Return the full KPI panel.

    Contains THREE completely independent blocks:
      • strict-trigger KPIs     — existing audit-grade metric
      • baseline ROI            — odds-favorite + model top-1 for
                                  comparison against any strategy
      • loose-trigger KPIs      — experimental betting rule

    Only predictions with an attached `result` contribute to the
    win-rate / ROI figures. Volume counts (total_triggers,
    loose_bets_total, etc.) use every prediction whether scored or not.
    """
    if preds is None:
        preds = list_predictions()

    n_races = 0
    n_result_races = 0

    # ── STRICT trigger accumulators (unchanged) ──
    total_triggers = 0
    triggers_with_result = 0
    trigger_wins = 0
    trigger_pnl = 0.0
    trigger_cost = 0.0
    trigger_count_histogram = defaultdict(int)

    # ── Baselines on the same pool ──
    odds_fav_wins = 0
    odds_fav_pnl = 0.0
    odds_fav_cost = 0.0
    model_top_wins = 0
    model_top_pnl = 0.0
    model_top_cost = 0.0

    # ── LOOSE bet accumulators (new, parallel) ──
    loose_bets_total = 0
    loose_bet_races_total = 0
    loose_bets_with_result = 0
    loose_bet_win_count = 0
    loose_bet_cost = 0.0
    loose_bet_payout = 0.0

    for e in preds:
        n_races += 1

        triggers = e.get("triggers") or []
        total_triggers += len(triggers)
        trigger_count_histogram[len(triggers)] += 1

        loose = e.get("loose_bets") or []
        if loose:
            loose_bet_races_total += 1
            loose_bets_total += len(loose)

        result = e.get("result") or {}
        fo = result.get("finishing_order") or []
        if not fo:
            continue
        winner = _race_winner(fo)
        if not winner:
            continue

        n_result_races += 1
        win_payout = float((result.get("payouts") or {}).get("単勝", 0) or 0)

        # Baseline: odds-favorite bet on this race
        odds_map = {}
        for h in fo:
            nm = _norm(h.get("name"))
            od = _parse_odds(h.get("odds"))
            if nm and od > 0 and nm not in odds_map:
                odds_map[nm] = od
        odds_fav = min(odds_map, key=lambda n: odds_map[n]) if odds_map else None
        odds_fav_cost += 100.0
        if odds_fav == winner:
            odds_fav_wins += 1
            odds_fav_pnl += win_payout

        # Baseline: flat model top-1 on this race
        ranked = e.get("ranked") or []
        model_top = ranked[0]["name"] if ranked else None
        model_top_cost += 100.0
        if model_top == winner:
            model_top_wins += 1
            model_top_pnl += win_payout

        # STRICT trigger outcomes
        for t in triggers:
            triggers_with_result += 1
            trigger_cost += 100.0
            if t.get("name") == winner:
                trigger_wins += 1
                trigger_pnl += win_payout

        # LOOSE bet outcomes (per-race, per-horse)
        for lb in loose:
            loose_bets_with_result += 1
            loose_bet_cost += 100.0
            if lb.get("name") == winner:
                loose_bet_win_count += 1
                loose_bet_payout += win_payout

    # ── Derived ratios ──
    def _safe_ratio(num, den):
        return (num / den) if den else 0.0

    trigger_rate_per_50 = (total_triggers / n_races * 50) if n_races else 0.0
    trigger_win_rate = _safe_ratio(trigger_wins, triggers_with_result)
    trigger_roi = _safe_ratio(trigger_pnl - trigger_cost, trigger_cost)

    odds_fav_roi = _safe_ratio(odds_fav_pnl - odds_fav_cost, odds_fav_cost)
    model_top_roi = _safe_ratio(model_top_pnl - model_top_cost, model_top_cost)

    loose_win_rate = _safe_ratio(loose_bet_win_count, loose_bets_with_result)
    loose_roi = _safe_ratio(loose_bet_payout - loose_bet_cost, loose_bet_cost)

    return {
        # --- Shared ---
        "n_races": n_races,
        "n_races_with_result": n_result_races,

        # --- STRICT trigger block (unchanged) ---
        "total_triggers": total_triggers,
        "triggers_per_50_races": round(trigger_rate_per_50, 2),
        "triggers_with_result": triggers_with_result,
        "trigger_wins": trigger_wins,
        "trigger_win_rate": round(trigger_win_rate, 4),
        "trigger_cost_yen": trigger_cost,
        "trigger_payout_yen": trigger_pnl,
        "trigger_pnl_yen": trigger_pnl - trigger_cost,
        "trigger_roi": round(trigger_roi, 4),
        "trigger_count_histogram": dict(sorted(trigger_count_histogram.items())),

        # --- Baselines ---
        "odds_favorite_win_rate": round(_safe_ratio(odds_fav_wins, n_result_races), 4),
        "odds_favorite_roi": round(odds_fav_roi, 4),
        "model_top_win_rate": round(_safe_ratio(model_top_wins, n_result_races), 4),
        "model_top_roi": round(model_top_roi, 4),

        # --- LOOSE bet block (new, experimental) ---
        "loose_bets_total": loose_bets_total,
        "loose_bet_races_total": loose_bet_races_total,
        "loose_bets_with_result": loose_bets_with_result,
        "loose_bet_win_count": loose_bet_win_count,
        "loose_bet_win_rate": round(loose_win_rate, 4),
        "loose_bet_cost_yen": loose_bet_cost,
        "loose_bet_payout_yen": loose_bet_payout,
        "loose_bet_pnl_yen": loose_bet_payout - loose_bet_cost,
        "loose_bet_roi": round(loose_roi, 4),
        "loose_vs_odds_roi_delta": round(loose_roi - odds_fav_roi, 4),
        "loose_vs_model_roi_delta": round(loose_roi - model_top_roi, 4),
        "loose_rule_version": "cons>=1_comp>=0.60_odds<=15_no_strongneg_v1",
    }


def recent_trigger_table(limit: int = 20) -> list[dict]:
    """Return the most recent STRICT trigger horses with their outcome
    if known."""
    rows = []
    for e in list_predictions()[:limit * 3]:
        triggers = e.get("triggers") or []
        if not triggers:
            continue
        result = e.get("result") or {}
        winner = _race_winner(result.get("finishing_order") or [])
        for t in triggers:
            rows.append({
                "race_id": e.get("race_id", ""),
                "race_name": e.get("race_name", ""),
                "race_date": e.get("race_date", ""),
                "horse": t["name"],
                "consensus_count": t["consensus_count"],
                "composite_condition": t["composite_condition"],
                "source_count": t["source_count"],
                "odds": t.get("odds", 0),
                "outcome": ("WIN" if winner and t["name"] == winner
                            else "LOSS" if winner else "PENDING"),
                "facts": t.get("facts_preview", []),
            })
    return rows[:limit]


def recent_loose_bets_table(limit: int = 20) -> list[dict]:
    """Return the most recent LOOSE bet horses with their outcome if known.

    Each row also carries payout / pnl so the UI can show 100-yen flat
    betting profitability per bet.
    """
    rows = []
    for e in list_predictions()[:limit * 3]:
        loose = e.get("loose_bets") or []
        if not loose:
            continue
        result = e.get("result") or {}
        fo = result.get("finishing_order") or []
        winner = _race_winner(fo)
        payouts = result.get("payouts") or {}
        win_payout = float(payouts.get("単勝", 0) or 0)
        # 結果ページの確定オッズを name→odds で引けるようにしておく
        # (RC-4: bet-time odds と settled odds を UI で併記するため)
        settled_by_name = {}
        for h in fo:
            nm = _norm(h.get("name"))
            od = _parse_odds(h.get("odds"))
            if nm and od > 0 and nm not in settled_by_name:
                settled_by_name[nm] = od
        for lb in loose:
            is_win = bool(winner and lb["name"] == winner)
            payout = win_payout if is_win else 0
            pnl = payout - 100
            rows.append({
                "race_date": e.get("race_date", ""),
                "race_id": e.get("race_id", ""),
                "race_name": e.get("race_name", ""),
                "horse": lb["name"],
                "odds": lb.get("odds", 0),
                # 馬単位の odds 出典 (FIX-3): "jra-official" / "netkeiba-api"
                # / "yahoo-keiba" / "live-odds-api" / "shutuba" / "result-page"
                # / "none" / None (古いログ)
                "odds_source": lb.get("odds_source"),
                "odds_fetched_at": lb.get("odds_fetched_at"),
                # 確定オッズ (結果ページから)。bet-time と分離して併記
                # することで RC-4 のズレが UI 上で可視化される。
                "odds_settled": settled_by_name.get(lb["name"]),
                # Multi-source cross-check (2026-04)
                # - odds_by_source: {source: float}. bet-time の値
                # - odds_disagreement_pct: 最大相対差 (0.0 〜)
                # - odds_disagreement_flag: >= 20% 差があった場合 True
                # これらが立っている bet は実運用では held になっているはず
                # だが、過去ログの事後監査のために表示する。
                "odds_by_source":         lb.get("odds_by_source") or {},
                "odds_disagreement_pct":  lb.get("odds_disagreement_pct", 0.0),
                "odds_disagreement_flag": lb.get("odds_disagreement_flag", False),
                "consensus_count": lb.get("consensus_count"),
                "composite_condition": lb.get("composite_condition"),
                "source_count": lb.get("source_count"),
                "loose_trigger_reason": lb.get("loose_trigger_reason", ""),
                "outcome": "WIN" if is_win else ("LOSS" if winner else "PENDING"),
                "payout": payout,
                "pnl": pnl if winner else 0,
            })
    return rows[:limit]
