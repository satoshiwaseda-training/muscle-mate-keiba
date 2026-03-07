"""
GitHub Gist を使った weights.json の永続化。

Streamlit Cloud はリブート時にファイルシステムがリセットされるため、
PDCA で学習した重みを GitHub Gist に保存・復元する。

Secrets に以下を設定してください:
  GITHUB_TOKEN = "ghp_xxxx"  # gist スコープのみでOK
  GIST_ID      = "abcdef1234567890"  # weights.json を含む Gist の ID
"""

import base64
import json
import os

import requests

_FILENAME = "weights.json"


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


def push_weights(weights: dict) -> bool:
    """重みを GitHub Gist に保存する。成功時 True。"""
    token = _token()
    gist_id = _gist_id()
    if not token or not gist_id:
        return False

    content = json.dumps(weights, ensure_ascii=False, indent=2)
    try:
        r = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers=_headers(),
            json={"files": {_FILENAME: {"content": content}}},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[github_sync] push_weights failed: {e}")
        return False


def pull_weights() -> dict | None:
    """GitHub Gist から重みを取得する。取得失敗時 None。"""
    token = _token()
    gist_id = _gist_id()
    if not token or not gist_id:
        return None

    try:
        r = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers=_headers(),
            timeout=10,
        )
        if r.status_code != 200:
            return None
        file_info = r.json().get("files", {}).get(_FILENAME)
        if not file_info:
            return None
        raw = file_info.get("content") or ""
        return json.loads(raw)
    except Exception as e:
        print(f"[github_sync] pull_weights failed: {e}")
        return None
