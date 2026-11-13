from pathlib import Path
import argparse
from collections import Counter, defaultdict
import statistics
import sys

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
sys.path.append(str(CURRENT_DIR))

from load_data import load_split
from preprocess import (
    build_evidence_candidates,
    detect_scale,
    extract_formula_cues,
    extract_program_ops,
    get_all_text,
    get_financial_numbers,
    infer_table_scale,
)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def pct(x, n):
    return 100.0 * x / n if n else 0.0


def analyze_split(split: str, limit: int | None = None):
    data = load_split(DATASET_DIR, split)
    if limit is not None:
        data = data[:limit]

    n = len(data)
    companies = Counter()
    years = Counter()
    evidence_type = Counter()
    op_counter = Counter()
    formula_counter = Counter()
    program_len = []
    text_lens = []
    table_rows = []
    table_cols = []
    gold_count = []
    scale_counter = Counter()
    number_flags = Counter()

    for ex in data:
        filename = ex.get("filename", "")
        if "/" in filename:
            parts = filename.split("/")
            companies[parts[0]] += 1
            if len(parts) > 1 and parts[1].isdigit():
                years[parts[1]] += 1

        qa = ex.get("qa", {})
        gold = set(qa.get("gold_inds", {}).keys())
        has_table = any(g.startswith("table_") for g in gold)
        has_text = any(g.startswith("text_") for g in gold)
        if has_table and has_text:
            evidence_type["mixed"] += 1
        elif has_table:
            evidence_type["table_only"] += 1
        elif has_text:
            evidence_type["text_only"] += 1
        else:
            evidence_type["none_or_unlabeled"] += 1
        gold_count.append(len(gold))

        ops = extract_program_ops(qa.get("program_re") or qa.get("program") or "")
        op_counter.update(ops)
        program_len.append(len(ops))
        formula_counter.update(extract_formula_cues(qa.get("question", "")))

        texts = get_all_text(ex)
        text_lens.append(len(texts))
        table = ex.get("table", []) or []
        table_rows.append(len(table))
        table_cols.append(max([len(r) for r in table], default=0))

        scale = infer_table_scale(ex)
        scale_counter[scale or "none"] += 1

        contexts = list(texts)
        if table:
            contexts.extend(" ".join(str(x) for x in row) for row in table)
        for c in contexts:
            for num in get_financial_numbers(c, context=c, inherited_scale=scale):
                if num.get("currency"):
                    number_flags["currency"] += 1
                if num.get("is_percent"):
                    number_flags["percent"] += 1
                if num.get("sign") == "negative":
                    number_flags["negative"] += 1
                if num.get("scale"):
                    number_flags[f"scale:{num.get('scale')}"] += 1

    print("=" * 80)
    print(f"Dataset Analysis: {split}")
    print("=" * 80)
    print("Examples:", n)
    print("Unique companies:", len(companies))
    if years:
        print("Year range:", min(years), "-", max(years))

    print("\n[Evidence modality]")
    for k, v in evidence_type.most_common():
        print(f"  {k}: {v} ({pct(v, n):.1f}%)")

    print("\n[Program operations]")
    for k, v in op_counter.most_common(20):
        print(f"  {k}: {v}")

    print("\n[Formula cues from question]")
    for k, v in formula_counter.most_common():
        print(f"  {k}: {v}")

    print("\n[Structure]")
    print(f"  avg text sentences: {mean(text_lens):.2f}")
    print(f"  avg table rows: {mean(table_rows):.2f}")
    print(f"  avg table cols: {mean(table_cols):.2f}")
    print(f"  avg program length: {mean(program_len):.2f}")
    print(f"  avg gold evidence count: {mean(gold_count):.2f}")

    print("\n[Table scale]")
    for k, v in scale_counter.most_common():
        print(f"  {k}: {v}")

    print("\n[Number flags]")
    for k, v in number_flags.most_common():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "private_test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    analyze_split(args.split, args.limit)


if __name__ == "__main__":
    main()
