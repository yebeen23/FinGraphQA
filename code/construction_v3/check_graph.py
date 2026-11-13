from __future__ import annotations

from pathlib import Path
from collections import Counter
import argparse
import sys
import networkx as nx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
sys.path.append(str(CURRENT_DIR))

from load_data import load_split, validate_example_schema, get_gold_evidence
from graph_builder import build_inference_graph
from schema import *


def summarize_node_types(G: nx.MultiDiGraph) -> Counter:
    return Counter(attrs.get("type", "unknown") for _, attrs in G.nodes(data=True))


def summarize_edge_types(G: nx.MultiDiGraph) -> Counter:
    return Counter(attrs.get("rel", "unknown") for _, _, attrs in G.edges(data=True))


def print_counter(title: str, counter: Counter) -> None:
    print(f"\n[{title}]")
    if not counter:
        print("  EMPTY")
        return
    for k, v in counter.most_common():
        print(f"  {k}: {v}")


def check_required_node_types(G: nx.MultiDiGraph) -> None:
    required = {NODE_PAGE, NODE_TEXT, NODE_TABLE_ROW, NODE_TABLE_CELL, NODE_QUANTITY, NODE_PERIOD, NODE_UNIT, NODE_METRIC, NODE_FINANCIAL_FACT, NODE_QUESTION, NODE_OPERATION}
    node_types = summarize_node_types(G)
    missing = sorted(t for t in required if node_types.get(t, 0) == 0)
    print("\n[Required node type check]")
    if missing:
        print("  STATUS: WARN")
        print("  Missing:", missing)
    else:
        print("  STATUS: PASS")


def check_required_edge_types(G: nx.MultiDiGraph) -> None:
    required = {REL_CONTAINS_TEXT, REL_HAS_CELL, REL_BELONGS_TO_ROW, REL_HAS_QUANTITY, REL_APPEARS_IN, REL_SOURCE_EVIDENCE, REL_CANDIDATE_OPERAND}
    edge_types = summarize_edge_types(G)
    missing = sorted(t for t in required if edge_types.get(t, 0) == 0)
    print("\n[Required edge type check]")
    if missing:
        print("  STATUS: WARN")
        print("  Missing:", missing)
    else:
        print("  STATUS: PASS")


def check_edge_explosion(G: nx.MultiDiGraph) -> None:
    ratio = G.number_of_edges() / max(G.number_of_nodes(), 1)
    print("\n[Edge explosion check]")
    print(f"  edge/node ratio: {ratio:.2f}")
    print("  STATUS:", "WARN" if ratio > 25 else "PASS")


def check_fact_nodes(G: nx.MultiDiGraph) -> None:
    table_numeric_cells = 0
    for n, a in G.nodes(data=True):
        if a.get("type") == NODE_TABLE_CELL and str(a.get("value", "")).strip():
            if any(ch.isdigit() for ch in str(a.get("value", ""))):
                table_numeric_cells += 1
    facts = [(n, a) for n, a in G.nodes(data=True) if a.get("type") == NODE_FINANCIAL_FACT]
    print("\n[Financial fact check]")
    print("  numeric table cells:", table_numeric_cells)
    print("  financial_fact nodes:", len(facts))
    if table_numeric_cells:
        table_facts = [a for _, a in facts if a.get("source_kind") == "table"]
        print(f"  table fact coverage approx: {len(table_facts) / table_numeric_cells * 100:.2f}%")
    print("  facts with metric:", sum(bool(a.get("metric")) for _, a in facts))
    print("  facts with period:", sum(bool(a.get("period")) for _, a in facts))
    print("  facts with unit/scale/currency:", sum(bool(a.get("unit") or a.get("scale") or a.get("currency")) for _, a in facts))
    print("  date-like facts:", sum(bool(a.get("is_date_number")) for _, a in facts))
    print("  STATUS:", "PASS" if facts else "FAIL")


def inspect_top_candidates(G: nx.MultiDiGraph, q_id: str, top_n: int = 10) -> None:
    rows = []
    for _, dst, attrs in G.out_edges(q_id, data=True):
        if attrs.get("rel") != REL_CANDIDATE_OPERAND:
            continue
        fa = G.nodes[dst]
        rows.append((attrs.get("rank", 999), dst, attrs.get("score", 0.0), fa))
    rows.sort(key=lambda x: x[0])
    print("\n[Top candidate facts]")
    if not rows:
        print("  EMPTY")
        return
    for rank, fid, score, fa in rows[:top_n]:
        print(f"  #{rank} {fid} score={score} metric={fa.get('metric')} period={fa.get('period')} value={fa.get('raw_value')} source={fa.get('evidence_id')} date_like={fa.get('is_date_number')}")


def check_gold_evidence_nodes(G: nx.MultiDiGraph, example: dict) -> None:
    gold = get_gold_evidence(example)
    print("\n[Gold evidence node check]")
    if not gold:
        print("  No gold evidence in this example.")
        return
    missing = sorted(g for g in gold if g not in G.nodes)
    print("  Gold:", sorted(gold))
    print("  Missing:", missing)
    print("  STATUS:", "FAIL" if missing else "PASS")


def check_gold_fact_coverage(G: nx.MultiDiGraph, example: dict) -> None:
    gold = get_gold_evidence(example)
    covered = []
    missing = []
    for g in gold:
        has_fact = False
        for n, a in G.nodes(data=True):
            if a.get("type") == NODE_FINANCIAL_FACT and a.get("evidence_id") == g:
                has_fact = True
                break
        (covered if has_fact else missing).append(g)
    print("\n[Gold fact coverage]")
    print("  Gold evidence with at least one fact:", sorted(covered))
    print("  Missing fact coverage:", sorted(missing))
    print(f"  Coverage: {len(covered)}/{len(gold)}")
    print("  STATUS:", "FAIL" if missing else "PASS")


def check_question_to_gold_paths(G: nx.MultiDiGraph, q_id: str, example: dict, max_hops: int = 4) -> None:
    print("\n[Question-to-gold path check]")
    gold = get_gold_evidence(example)
    UG = G.to_undirected()
    for g in sorted(gold):
        if q_id not in UG or g not in UG:
            print(f"  {g}: missing node STATUS=FAIL")
            continue
        try:
            dist = nx.shortest_path_length(UG, q_id, g)
            status = "PASS" if dist <= max_hops else "WARN"
            print(f"  {g}: distance={dist} STATUS={status}")
        except nx.NetworkXNoPath:
            print(f"  {g}: no path STATUS=FAIL")


def check_single_example(split: str = "train", idx: int = 0) -> None:
    data = load_split(DATASET_DIR, split)
    if idx < 0 or idx >= len(data):
        raise IndexError(f"idx={idx} out of range. Dataset size={len(data)}")
    example = data[idx]
    G, q_id = build_inference_graph(example)

    print("=" * 80)
    print("FinGraphQA Graph Check")
    print("=" * 80)
    print("Dataset file:", DATASET_DIR / f"{split}.json")
    print("Split/index:", split, idx)
    print("Schema:", validate_example_schema(example, require_gold=True))
    print("Example ID:", example.get("id"))
    print("Filename:", example.get("filename"))
    print("Question:", example.get("qa", {}).get("question"))
    print("Question node:", q_id)

    print("\n[Graph size]")
    print("  Nodes:", G.number_of_nodes())
    print("  Edges:", G.number_of_edges())
    print_counter("Node types", summarize_node_types(G))
    print_counter("Edge types", summarize_edge_types(G))
    check_required_node_types(G)
    check_required_edge_types(G)
    check_edge_explosion(G)
    check_fact_nodes(G)
    inspect_top_candidates(G, q_id)
    check_gold_evidence_nodes(G, example)
    check_gold_fact_coverage(G, example)
    check_question_to_gold_paths(G, q_id, example)
    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "private_test"])
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()
    check_single_example(args.split, args.idx)


if __name__ == "__main__":
    main()
