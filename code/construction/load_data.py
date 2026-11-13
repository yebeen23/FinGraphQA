import json
from pathlib import Path
from typing import Any, Dict, List


def load_finqa_json(path: str) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")

    return data


def has_gold_labels(example: Dict[str, Any]) -> bool:
    qa = example.get("qa", {})
    return (
        "gold_inds" in qa
        and "program" in qa
        and "exe_ans" in qa
    )


if __name__ == "__main__":
    data = load_finqa_json("dataset/train.json")
    print(len(data))
    print(data[0].keys())
    print(data[0]["qa"].keys())