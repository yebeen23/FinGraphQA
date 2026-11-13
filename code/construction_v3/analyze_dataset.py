from __future__ import annotations

from pathlib import Path
from collections import Counter
import argparse
import re
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
sys.path.append(str(CURRENT_DIR))

from load_data import load_split, get_gold_evidence
from preprocess import extract_formula_cues, infer_table_scale, get_all_text, get_financial_numbers


def evidence_modality(gold):
    has_table = any(g.startswith("table_") for g in gold)
    has_text = any(g.startswith("text_") for g in gold)
    if has_table and has_text:
        return "mixed"
    if has_table:
        return "table_only"
    if has_text:
        return "text_only"
    return "unknown"


def op_names(program: str):
    return re.findall(r"([a-zA-Z_]+)\(", program or "")


def analyze(split: str) -> None:
    data = load_split(DATASET_DIR, split)
    companies = Counter()
    years = []
    modality = Counter()
    ops = Counter()
    cues = Counter()
    scales = Counter()
    flags = Counter()
    total_text = 0
    total_rows = 0
    total_cols = 0
    total_prog_len = 0
    total_gold = 0

    for ex in data:
        filename = ex.get("filename", "")
        if "/" in filename:
            companies[filename.split("/")[0]] += 1
        for y in re.findall(r"\b(?:19|20)\d{2}\b", filename + " " + ex.get("id", "")):
            years.append(int(y))
        gold = get_gold_evidence(ex)
        modality[evidence_modality(gold)] += 1
        total_gold += len(gold)
        qa = ex.get("qa", {}) or {}
        ops.update(op_names(qa.get("program", "")))
        cues.update(extract_formula_cues(qa.get("question", "")) or ["direct_lookup"])
        scale = infer_table_scale(ex) or "none"
        scales[scale] += 1
        text = get_all_text(ex)
        total_text += len(text)
        table = ex.get("table", []) or []
        total_rows += len(table)
        total_cols += max((len(r) for r in table), default=0)
        total_prog_len += len(op_names(qa.get("program", "")))
        all_content = " ".join(text + [" ".join(map(str, r)) for r in table])
        for num in get_financial_numbers(all_content, context=all_content, inherited_scale=scale if scale != "none" else None):
            if num.get("currency"):
                flags["currency"] += 1
            if num.get("is_percent"):
                flags["percent"] += 1
            if num.get("value", 0) < 0:
                flags["negative"] += 1
            if num.get("scale"):
                flags[f"scale:{num.get('scale')}"] += 1

    n = len(data)
    print("=" * 80)
    print(f"Dataset Analysis: {split}")
    print("=" * 80)
    print("Examples:", n)
    print("Unique companies:", len(companies))
    if years:
        print(f"Year range: {min(years)} - {max(years)}")
    print("\n[Evidence modality]")
    for k, v in modality.most_common():
        print(f"  {k}: {v} ({v / max(n, 1) * 100:.1f}%)")
    print("\n[Program operations]")
    for k, v in ops.most_common():
        print(f"  {k}: {v}")
    print("\n[Formula cues from question]")
    for k, v in cues.most_common():
        print(f"  {k}: {v}")
    print("\n[Structure]")
    print(f"  avg text sentences: {total_text / max(n, 1):.2f}")
    print(f"  avg table rows: {total_rows / max(n, 1):.2f}")
    print(f"  avg table cols: {total_cols / max(n, 1):.2f}")
    print(f"  avg program length: {total_prog_len / max(n, 1):.2f}")
    print(f"  avg gold evidence count: {total_gold / max(n, 1):.2f}")
    print("\n[Table scale]")
    for k, v in scales.most_common():
        print(f"  {k}: {v}")
    print("\n[Number flags]")
    for k, v in flags.most_common():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "private_test"])
    args = parser.parse_args()
    analyze(args.split)


if __name__ == "__main__":
    main()
