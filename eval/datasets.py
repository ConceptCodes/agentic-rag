from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from src.constants import (
    EVAL_DATA_DIR,
    HOTPOT_DATASET_CONFIG,
    HOTPOT_DATASET_NAME,
    HOTPOT_DATASET_SPLIT,
)


def _cache_path() -> Path:
    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{HOTPOT_DATASET_NAME}_{HOTPOT_DATASET_CONFIG}_{HOTPOT_DATASET_SPLIT}.json"
    return EVAL_DATA_DIR / filename


def _download_hotpotqa_rows(limit: int) -> list[dict[str, Any]]:
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": HOTPOT_DATASET_NAME,
        "config": HOTPOT_DATASET_CONFIG,
        "split": HOTPOT_DATASET_SPLIT,
        "offset": 0,
        "length": max(1, limit),
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    rows = payload.get("rows", [])

    items: list[dict[str, Any]] = []
    for row in rows:
        value = row.get("row", {})
        items.append(
            {
                "id": value.get("id", ""),
                "question": value.get("question", ""),
                "answer": value.get("answer", ""),
            }
        )
    return items


def fetch_hotpotqa_subset(limit: int = 10, use_cache: bool = True) -> list[dict[str, Any]]:
    path = _cache_path()
    if use_cache and path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data[: max(1, limit)]

    items = _download_hotpotqa_rows(limit=max(limit, 100))
    path.write_text(json.dumps(items, ensure_ascii=True, indent=2), encoding="utf-8")
    return items[: max(1, limit)]
