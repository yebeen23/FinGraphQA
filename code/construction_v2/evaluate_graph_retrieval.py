from __future__ import annotations

from pathlib import Path
import argparse
from collections import defaultdict
import json
import sys
from typing import Any, Dict, List, Optional, Set

import networkx as nx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.append(str(CURRENT_DIR))

from graph_builder import build_inference_graph
from load_data import get_gold_evidence, load_split

try:
    from preprocess import extract_formula_cues
except Exception:
    def extract_formula_cues(question: str) -> list[str]:
        return []

from evidence_set_ranker import (
    rank_evidence_sources,
    relaxed_all_covered,
    relaxed_any_covered,
)


def evidence_modality(gold: Set[str]) -> str:
    has_table = any(g.startswith("table_") for g in gold)
    has_text = any(g.startswith("text_") for g in gold)
    if has_table and has_text:
        return "mixed"
    if has_table:
        return "table_only"
    if has_text:
        return "text_only"
    return "unknown"


def compute_reachability(G: nx.MultiDiGraph, q_id: str, gold_sources: Set[str]) -> float:
    if not gold_sources or q_id not in G.nodes:
        return 0.0
    UG = G.to_undirected()
    reachable = 0
    for gold in gold_sources:
        if gold not in UG:
            continue
        try:
            nx.shortest_path_length(UG, q_id, gold)
            reachable += 1
        except nx.NetworkXNoPath:
            pass
    return reachable / len(gold_sources)


def _update_group_stats(group_stats, group_name: str, gold: Set[str], ranked_sources: List[str], G: nx.MultiDiGraph, ks: List[int]):
    group_stats[group_name]["n"] += 1
    for k in ks:
        selected = set(ranked_sources[:k])
        group_stats[group_name][f"strict_any@{k}"] += float(bool(gold & selected))
        group_stats[group_name][f"strict_all@{k}"] += float(gold <= selected)
        group_stats[group_name][f"relaxed_any@{k}"] += float(relaxed_any_covered(G, gold, selected))
        group_stats[group_name][f"relaxed_all@{k}"] += float(relaxed_all_covered(G, gold, selected))


def _print_group_stats(title: str, group_stats, ks: List[int]):
    print(f"\n[{title}]")
    if not group_stats:
        print("  EMPTY")
        return
    for group_name in sorted(group_stats.keys()):
        n = int(group_stats[group_name]["n"])
        if n <= 0:
            continue
        parts = [f"n={n}"]
        for k in ks:
            parts.append(f"StrictAll@{k}={group_stats[group_name][f'strict_all@{k}'] / n:.4f}")
            parts.append(f"RelaxedAll@{k}={group_stats[group_name][f'relaxed_all@{k}'] / n:.4f}")
        print(f"  {group_name}: " + " | ".join(parts))


def _detail_to_jsonable(detail: Dict[str, Any]) -> Dict[str, Any]:
    best = detail.get("best_fact") or {}
    return {
        "source": detail.get("source"),
        "kind": detail.get("kind"),
        "score": detail.get("score", detail.get("best_score")),
        "best_score": detail.get("best_score"),
        "base_score": detail.get("base_score"),
        "fact_count": detail.get("fact_count"),
        "expanded": detail.get("expanded"),
        "expanded_from": detail.get("expanded_from"),
        "bridge_rel": detail.get("bridge_rel"),
        "best_fact": {
            "fact_id": best.get("fact_id"),
            "metric": best.get("metric"),
            "period": best.get("period"),
            "value": best.get("value"),
            "score": best.get("score"),
            "is_date_number": best.get("is_date_number"),
        },
    }


def save_missed_cases(path: Path, missed_cases: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for case in missed_cases:
            row = dict(case)
            row["top_details"] = [_detail_to_jsonable(d) for d in case.get("top_details", [])]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate(
    split: str,
    limit: Optional[int],
    ks: List[int],
    verbose_missed: bool,
    save_missed: bool,
    missed_path: Optional[Path],
    max_set_size: int,
):
    data = load_split(DATASET_DIR, split)
    if limit is not None:
        data = data[:limit]

    ks = sorted(set(ks))
    max_k = max(ks)

    totals = defaultdict(int)
    sums = defaultdict(float)
    strict_missed: List[Dict[str, Any]] = []
    relaxed_missed: List[Dict[str, Any]] = []
    modality_missed = defaultdict(list)
    mixed_missed: List[Dict[str, Any]] = []

    modality_stats = defaultdict(lambda: defaultdict(float))
    cue_stats = defaultdict(lambda: defaultdict(float))

    for idx, example in enumerate(data):
        gold = get_gold_evidence(example)
        if not gold:
            continue

        question = example.get("qa", {}).get("question", "")
        modality = evidence_modality(gold)
        cues = extract_formula_cues(question) or ["direct_lookup"]

        G, q_id = build_inference_graph(example)
        ranked_sources, ranked_details = rank_evidence_sources(
            G,
            q_id,
            question=question,
            top_k=max_k,
            max_set_size=max_set_size,
        )

        totals["examples"] += 1
        sums["gold_reachability"] += compute_reachability(G, q_id, gold)

        for k in ks:
            selected = set(ranked_sources[:k])
            sums[f"strict_any@{k}"] += float(bool(gold & selected))
            sums[f"strict_all@{k}"] += float(gold <= selected)
            sums[f"relaxed_any@{k}"] += float(relaxed_any_covered(G, gold, selected))
            sums[f"relaxed_all@{k}"] += float(relaxed_all_covered(G, gold, selected))

        _update_group_stats(modality_stats, modality, gold, ranked_sources, G, ks)
        for cue in cues:
            _update_group_stats(cue_stats, cue, gold, ranked_sources, G, ks)

        strict_ok_at_max = gold <= set(ranked_sources[:max_k])
        relaxed_ok_at_max = relaxed_all_covered(G, gold, set(ranked_sources[:max_k]))

        if not strict_ok_at_max:
            case = {
                "idx": idx,
                "id": example.get("id"),
                "question": question,
                "gold": sorted(gold),
                "modality": modality,
                "cues": cues,
                "top_sources": ranked_sources[:max_k],
                "top_details": ranked_details[:max_k],
            }
            strict_missed.append(case)
            modality_missed[modality].append(case)
            if modality == "mixed":
                mixed_missed.append(case)

        if not relaxed_ok_at_max:
            relaxed_missed.append({
                "idx": idx,
                "id": example.get("id"),
                "question": question,
                "gold": sorted(gold),
                "modality": modality,
                "cues": cues,
                "top_sources": ranked_sources[:max_k],
                "top_details": ranked_details[:max_k],
            })

    n = totals["examples"]
    print("=" * 80)
    print(f"Graph Retrieval Evaluation: {split}")
    print("=" * 80)
    print("Evaluated examples with gold:", n)

    print("\n[Strict source recall]")
    for k in ks:
        print(f"Any gold source Recall@{k}: {sums[f'strict_any@{k}'] / max(n, 1):.4f}")
        print(f"All gold source Recall@{k}: {sums[f'strict_all@{k}'] / max(n, 1):.4f}")

    print("\n[Relaxed source recall]")
    for k in ks:
        print(f"Relaxed Any source Recall@{k}: {sums[f'relaxed_any@{k}'] / max(n, 1):.4f}")
        print(f"Relaxed All source Recall@{k}: {sums[f'relaxed_all@{k}'] / max(n, 1):.4f}")

    print(f"\nQuestion-to-gold reachability: {sums['gold_reachability'] / max(n, 1):.4f}")

    _print_group_stats("Recall by evidence modality", modality_stats, ks=[k for k in ks if k in {5, 10}] or ks)
    _print_group_stats("Recall by formula cue", cue_stats, ks=[k for k in ks if k in {5, 10}] or ks)

    print("\n[Missed count by modality, strict@max_k]")
    for mod in sorted(modality_missed.keys()):
        print(f"  {mod}: {len(modality_missed[mod])}")

    print("\n[Mixed-only missed examples sample]")
    for m in mixed_missed[:10]:
        print(f"  idx={m['idx']} id={m['id']}")
        print(f"    q={m['question']}")
        print(f"    gold={m['gold']}")
        print(f"    cues={m['cues']}")
        print(f"    top_sources={m['top_sources']}")

    print("\n[Strict missed examples sample]")
    for m in strict_missed[:10]:
        print(f"  idx={m['idx']} id={m['id']} modality={m['modality']}")
        print(f"    q={m['question']}")
        print(f"    gold={m['gold']}")
        print(f"    top_sources={m['top_sources']}")
        if verbose_missed:
            print("    top source details:")
            for d in m["top_details"]:
                best = d.get("best_fact") or {}
                print(
                    "      "
                    f"source={d.get('source')} "
                    f"score={d.get('score', d.get('best_score', 0.0)):.4f} "
                    f"base={d.get('base_score', 0.0):.4f} "
                    f"count={d.get('fact_count')} "
                    f"expanded={d.get('expanded')} "
                    f"from={d.get('expanded_from')} "
                    f"fact={best.get('fact_id')} "
                    f"metric={best.get('metric')} "
                    f"period={best.get('period')} "
                    f"value={best.get('value')}"
                )

    if save_missed:
        if missed_path is None:
            lim = "all" if limit is None else str(limit)
            missed_path = OUTPUT_DIR / f"graph_retrieval_missed_{split}_{lim}.jsonl"
        save_missed_cases(missed_path, strict_missed)
        print(f"\nSaved strict missed cases to: {missed_path}")
        mixed_path = missed_path.with_name(missed_path.stem + "_mixed_only" + missed_path.suffix)
        save_missed_cases(mixed_path, mixed_missed)
        print(f"Saved mixed-only missed cases to: {mixed_path}")


def parse_ks(raw: str) -> List[int]:
    ks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"k must be positive, got: {value}")
        ks.append(value)
    if not ks:
        raise ValueError("At least one k value is required.")
    return ks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "private_test"])
    parser.add_argument("--limit", type=int, default=200, help="Number of examples to evaluate. Use -1 for all examples.")
    parser.add_argument("--ks", default="1,3,5,10,20", help="Comma-separated Recall@k values.")
    parser.add_argument("--verbose_missed", "--verbose-missed", action="store_true")
    parser.add_argument("--save_missed", "--save-missed", action="store_true")
    parser.add_argument("--missed_path", type=str, default=None)
    parser.add_argument("--max_set_size", type=int, default=5)
    args = parser.parse_args()

    limit = None if args.limit is not None and args.limit < 0 else args.limit
    missed_path = Path(args.missed_path) if args.missed_path else None

    evaluate(
        split=args.split,
        limit=limit,
        ks=parse_ks(args.ks),
        verbose_missed=args.verbose_missed,
        save_missed=args.save_missed,
        missed_path=missed_path,
        max_set_size=args.max_set_size,
    )


if __name__ == "__main__":
    main()
