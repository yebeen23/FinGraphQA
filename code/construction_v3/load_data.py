from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


def load_finqa_json(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON at {path}, got {type(data)}")
    return data


def load_split(dataset_dir: str | Path, split: str) -> List[Dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")
    return load_finqa_json(path)


def iter_examples(dataset_dir: str | Path, split: str, limit: Optional[int] = None) -> Iterable[Dict[str, Any]]:
    data = load_split(dataset_dir, split)
    if limit is not None and limit >= 0:
        data = data[:limit]
    yield from data


def get_gold_evidence(example: Dict[str, Any]) -> Set[str]:
    qa = example.get("qa", {}) or {}
    return set((qa.get("gold_inds") or {}).keys())


def has_gold_labels(example: Dict[str, Any]) -> bool:
    qa = example.get("qa", {}) or {}
    return bool(qa.get("gold_inds"))


def validate_example_schema(example: Dict[str, Any], require_gold: bool = False) -> Dict[str, Any]:
    required_top = ["pre_text", "post_text", "table", "filename", "qa", "id"]
    required_qa = ["question"]
    gold_qa = ["gold_inds", "program", "exe_ans"]

    missing_top = [k for k in required_top if k not in example]
    qa = example.get("qa", {}) or {}
    missing_qa = [k for k in required_qa if k not in qa]
    missing_gold = [k for k in gold_qa if k not in qa] if require_gold else []

    ok = not missing_top and not missing_qa and not missing_gold
    return {
        "ok": ok,
        "missing_top_level": missing_top,
        "missing_qa": missing_qa,
        "missing_gold": missing_gold,
    }


if __name__ == "__main__":
    data = load_finqa_json("dataset/train.json")
    print(len(data))
    print(data[0].keys())
    print(data[0].get("qa", {}).keys())
