import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


REQUIRED_TOP_LEVEL_KEYS = {"pre_text", "post_text", "table", "filename", "qa", "id"}
REQUIRED_QA_KEYS = {"question"}
OPTIONAL_GOLD_QA_KEYS = {"gold_inds", "program", "program_re", "answer", "exe_ans", "steps"}


def load_finqa_json(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of examples, got {type(data)} from {path}")
    return data


def load_split(dataset_dir: str | Path, split: str) -> List[Dict[str, Any]]:
    dataset_dir = Path(dataset_dir)
    path = dataset_dir / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return load_finqa_json(path)


def load_all_splits(dataset_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    dataset_dir = Path(dataset_dir)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for split in ["train", "dev", "test", "private_test"]:
        path = dataset_dir / f"{split}.json"
        if path.exists():
            out[split] = load_finqa_json(path)
    return out


def iter_examples(
    dataset_dir: str | Path,
    split: str,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    data = load_split(dataset_dir, split)
    for i, ex in enumerate(data):
        if limit is not None and i >= limit:
            break
        yield ex


def validate_example_schema(example: Dict[str, Any], allow_missing_gold: bool = True) -> Dict[str, Any]:
    missing_top = sorted(REQUIRED_TOP_LEVEL_KEYS - set(example.keys()))
    qa = example.get("qa", {})
    missing_qa = sorted(REQUIRED_QA_KEYS - set(qa.keys()))

    missing_gold = []
    if not allow_missing_gold:
        missing_gold = sorted(OPTIONAL_GOLD_QA_KEYS - set(qa.keys()))

    return {
        "ok": not missing_top and not missing_qa and not missing_gold,
        "missing_top_level": missing_top,
        "missing_qa": missing_qa,
        "missing_gold": missing_gold,
    }


def has_gold_labels(example: Dict[str, Any]) -> bool:
    qa = example.get("qa", {})
    return bool(qa.get("gold_inds")) and ("program" in qa or "program_re" in qa)


def get_example_id(example: Dict[str, Any]) -> str:
    return str(example.get("id") or example.get("filename") or "unknown")


def get_question(example: Dict[str, Any]) -> str:
    return str(example.get("qa", {}).get("question", ""))


def get_gold_evidence(example: Dict[str, Any]) -> set[str]:
    qa = example.get("qa", {})
    return set(qa.get("gold_inds", {}).keys())


if __name__ == "__main__":
    data = load_finqa_json("dataset/train.json")
    print("examples:", len(data))
    print("schema:", validate_example_schema(data[0]))
    print("example_id:", get_example_id(data[0]))
    print("question:", get_question(data[0]))
