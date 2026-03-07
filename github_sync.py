"""
GitHub Gist を使った data/*.json の永続化。

Streamlit Cloud はリブート時にファイルシステムがリセットされるため、
PDCA で学習した重み・予測・結果・ログを GitHub Gist に保存・復元する。

Gist 内のファイル構成:
  weights.json      — PDCA学習済み重み
  predictions.json  — 過去の予測データ
  results.json      — 過去のレース結果
  pdca_log.json     — PDCAログ

Secrets に以下を設定してください:
  GITHUB_TOKEN = "ghp_xxxx"   # gist スコープのみでOK
  GIST_ID      = "abcdef..."  # 上記ファイルを含む Gist の ID
"""

import json
import os

import requests

# Gist内のファイル名マッピング
_FILES = ["weights.json", "predictions.json", "results.json", "pdca_log.json"]


def _token() -> str:
    try:
        import streamlit as st
        return st.secrets.get("GITHUB_TOKEN", os.getenv("GITHUB_TOKEN", ""))
    except Exception:
        return os.getenv("GITHUB_TOKEN", "")


def _gist_id() -> str:
    try:
        import streamlit as st
        return st.secrets.get("GIST_ID", os.getenv("GIST_ID", ""))
    except Exception:
        return os.getenv("GIST_ID", "")


def _headers() -> dict:
    return {
        "Authorization": f"token {_token()}",
        "Accept": "application/vnd.github.v3+json",
    }


def _available() -> bool:
    return bool(_token() and _gist_id())


# ── 汎用 push/pull ────────────────────────────────────────────

def push_file(filename: str, data) -> bool:
    """data (dict or list) を Gist の filename に保存する。"""
    if not _available():
        return False
    content = json.dumps(data, ensure_ascii=False, indent=2)
    try:
        r = requests.patch(
            f"https://api.github.com/gists/{_gist_id()}",
            headers=_headers(),
            json={"files": {filename: {"content": content}}},
            timeout=10,
        )
        ok = r.status_code == 200
        if not ok:
            print(f"[github_sync] push_file {filename} failed: {r.status_code}")
        return ok
    except Exception as e:
        print(f"[github_sync] push_file {filename} error: {e}")
        return False


def pull_file(filename: str):
    """Gist から filename を取得する。失敗時は None。"""
    if not _available():
        return None
    try:
        r = requests.get(
            f"https://api.github.com/gists/{_gist_id()}",
            headers=_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            return None
        file_info = r.json().get("files", {}).get(filename)
        if not file_info:
            return None
        raw = file_info.get("content") or ""
        return json.loads(raw)
    except Exception as e:
        print(f"[github_sync] pull_file {filename} error: {e}")
        return None


# ── weights 専用 (data_store.py から呼び出し) ────────────────

def push_weights(weights: dict) -> bool:
    return push_file("weights.json", weights)


def pull_weights() -> dict | None:
    return pull_file("weights.json")


# ── 全データ一括 sync ────────────────────────────────────────

def push_all(data_dir) -> dict:
    """
    data_dir 内の全 JSON ファイルを Gist にアップロードする。
    戻り値: {filename: True/False}
    """
    from pathlib import Path
    results = {}
    for fname in _FILES:
        path = Path(data_dir) / fname
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            results[fname] = push_file(fname, data)
    return results


def pull_all(data_dir) -> dict:
    """
    Gist から全 JSON ファイルをダウンロードして data_dir に書き込む。
    ローカルに既存ファイルがある場合は上書きしない（ローカル優先）。
    戻り値: {filename: True/False}
    """
    from pathlib import Path
    results = {}
    for fname in _FILES:
        path = Path(data_dir) / fname
        if path.exists():
            results[fname] = False  # ローカル優先
            continue
        data = pull_file(fname)
        if data is not None:
            path.parent.mkdir(exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            results[fname] = True
        else:
            results[fname] = False
    return results
