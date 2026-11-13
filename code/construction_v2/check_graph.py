from pathlib import Path
from collections import Counter
import argparse
import sys
import networkx as nx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
sys.path.append(str(CURRENT_DIR))

from graph_builder import build_inference_graph
from load_data import get_gold_evidence, load_split, validate_example_schema
from schema import *


def summarize_node_types(G: nx.MultiDiGraph) -> Counter:
    return Counter(attrs.get("type", "unknown") for _, attrs in G.nodes(data=True))


def summarize_edge_types(G: nx.MultiDiGraph) -> Counter:
    return Counter(attrs.get("rel", "unknown") for _, _, attrs in G.edges(data=True))


def print_counter(title: str, counter: Counter, limit: int | None = None):
    print(f"\n[{title}]")
    items = counter.most_common(limit)
    if not items:
        print("  EMPTY")
        return
    for k, v in items:
        print(f"  {k}: {v}")


def check_required_node_types(G: nx.MultiDiGraph):
    required = {
        NODE_PAGE, NODE_TEXT, NODE_TABLE_ROW, NODE_TABLE_COLUMN, NODE_TABLE_CELL,
        NODE_QUANTITY, NODE_PERIOD, NODE_UNIT, NODE_METRIC, NODE_FINANCIAL_FACT,
        NODE_QUESTION, NODE_OPERATION,
    }
    node_types = summarize_node_types(G)
    missing = sorted(t for t in required if node_types.get(t, 0) == 0)
    print("\n[Required node type check]")
    if missing:
        print("  STATUS: WARN")
        print("  Missing:", missing)
    else:
        print("  STATUS: PASS")


def check_required_edge_types(G: nx.MultiDiGraph):
    required = {
        REL_CONTAINS_TEXT, REL_CONTAINS_ROW, REL_CONTAINS_COLUMN, REL_HAS_CELL,
        REL_BELONGS_TO_ROW, REL_BELONGS_TO_COLUMN, REL_HAS_QUANTITY,
        REL_SOURCE_EVIDENCE, REL_CANDIDATE_OPERAND,
    }
    edge_types = summarize_edge_types(G)
    missing = sorted(t for t in required if edge_types.get(t, 0) == 0)
    print("\n[Required edge type check]")
    if missing:
        print("  STATUS: WARN")
        print("  Missing:", missing)
    else:
        print("  STATUS: PASS")


def check_edge_explosion(G: nx.MultiDiGraph, max_ratio: float = 20.0):
    ratio = G.number_of_edges() / max(G.number_of_nodes(), 1)
    edge_types = summarize_edge_types(G)
    print("\n[Edge explosion check]")
    print(f"  edge/node ratio: {ratio:.2f}")
    if ratio > max_ratio:
        print("  STATUS: WARN")
        print(f"  Ratio exceeds {max_ratio}. Top edge types may be too dense:")
        for rel, count in edge_types.most_common(5):
            print(f"    {rel}: {count}")
    else:
        print("  STATUS: PASS")


def check_fact_nodes(G: nx.MultiDiGraph):
    facts = [(n, a) for n, a in G.nodes(data=True) if a.get("type") == NODE_FINANCIAL_FACT]
    numeric_cells = [n for n, a in G.nodes(data=True) if a.get("type") == NODE_TABLE_CELL and any(ch.isdigit() for ch in str(a.get("value", "")))]

    with_metric = sum(1 for _, a in facts if a.get("metric"))
    with_period = sum(1 for _, a in facts if a.get("period") or a.get("period_year"))
    with_unit_scale = sum(1 for _, a in facts if a.get("unit") or a.get("scale") or a.get("currency"))

    print("\n[Financial fact check]")
    print("  numeric table cells:", len(numeric_cells))
    print("  financial_fact nodes:", len(facts))
    if numeric_cells:
        print(f"  table fact coverage approx: {len([1 for _, a in facts if a.get('source_kind') == 'table']) / len(numeric_cells):.2%}")
    print("  facts with metric:", with_metric)
    print("  facts with period:", with_period)
    print("  facts with unit/scale/currency:", with_unit_scale)

    if not facts:
        print("  STATUS: FAIL")
    elif with_metric == 0 or with_period == 0:
        print("  STATUS: WARN")
    else:
        print("  STATUS: PASS")


def inspect_top_candidate_facts(G: nx.MultiDiGraph, q_id: str, limit: int = 10):
    print("\n[Top candidate facts]")
    if q_id not in G:
        print("  No question node.")
        return
    edges = []
    for _, dst, attrs in G.out_edges(q_id, data=True):
        if attrs.get("rel") == REL_CANDIDATE_OPERAND:
            edges.append((attrs.get("rank", 999), dst, attrs))
    edges.sort(key=lambda x: x[0])
    if not edges:
        print("  EMPTY")
        return
    for rank, fact_id, eattrs in edges[:limit]:
        f = G.nodes[fact_id]
        print(
            f"  #{rank} {fact_id} score={eattrs.get('score')} "
            f"metric={f.get('metric')} period={f.get('period')} "
            f"value={f.get('raw_value')} source={f.get('evidence_id')}"
        )


def check_gold_evidence_nodes(G: nx.MultiDiGraph, example: dict):
    gold = get_gold_evidence(example)
    print("\n[Gold evidence node check]")
    if not gold:
        print("  No gold evidence available.")
        return
    missing = sorted(g for g in gold if g not in G.nodes)
    print("  Gold:", sorted(gold))
    print("  Missing:", missing)
    print("  STATUS:", "FAIL" if missing else "PASS")


def check_gold_fact_coverage(G: nx.MultiDiGraph, example: dict):
    gold = get_gold_evidence(example)
    print("\n[Gold fact coverage]")
    if not gold:
        print("  No gold evidence available.")
        return
    facts = [(n, a) for n, a in G.nodes(data=True) if a.get("type") == NODE_FINANCIAL_FACT]
    by_evidence = Counter(a.get("evidence_id") for _, a in facts)
    covered = {g for g in gold if by_evidence.get(g, 0) > 0}
    print("  Gold evidence with at least one fact:", sorted(covered))
    print("  Missing fact coverage:", sorted(gold - covered))
    print(f"  Coverage: {len(covered)}/{len(gold)}")
    print("  STATUS:", "PASS" if covered == gold else "WARN")


def check_question_to_gold_paths(G: nx.MultiDiGraph, q_id: str, example: dict, max_hops: int = 4):
    gold = get_gold_evidence(example)
    print("\n[Question-to-gold path check]")
    if not gold:
        print("  No gold evidence available.")
        return
    if q_id not in G:
        print("  Question node missing.")
        return
    UG = G.to_undirected()
    for g in sorted(gold):
        if g not in UG:
            print(f"  {g}: NOT FOUND")
            continue
        try:
            length = nx.shortest_path_length(UG, q_id, g)
            status = "PASS" if length <= max_hops else "WARN"
            print(f"  {g}: distance={length} STATUS={status}")
        except nx.NetworkXNoPath:
            print(f"  {g}: UNREACHABLE STATUS=FAIL")


def check_single_example(split: str, idx: int, candidate_k: int, include_comparison: bool):
    data = load_split(DATASET_DIR, split)
    if idx < 0 or idx >= len(data):
        raise IndexError(f"idx={idx} out of range. Dataset size={len(data)}")
    example = data[idx]
    schema = validate_example_schema(example)
    G, q_id = build_inference_graph(example, top_k_candidate_facts=candidate_k, include_contextual_comparison=include_comparison)

    print("=" * 80)
    print("FinGraphQA Graph Check")
    print("=" * 80)
    print("Dataset file:", DATASET_DIR / f"{split}.json")
    print("Split/index:", split, idx)
    print("Schema:", schema)
    print("Example ID:", example.get("id"))
    print("Filename:", example.get("filename"))
    print("Question:", example.get("qa", {}).get("question"))
    print("Question node:", q_id)
    print("\n[Graph size]")
    print("  Nodes:", G.number_of_nodes())
    print("  Edges:", G.number_of_edges())

    print_counter("Node types", summarize_node_types(G))
    print_counter("Edge types", summarize_edge_types(G), limit=30)
    check_required_node_types(G)
    check_required_edge_types(G)
    check_edge_explosion(G)
    check_fact_nodes(G)
    inspect_top_candidate_facts(G, q_id)
    check_gold_evidence_nodes(G, example)
    check_gold_fact_coverage(G, example)
    check_question_to_gold_paths(G, q_id, example)
    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "private_test"])
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--include-comparison", action="store_true")
    args = parser.parse_args()
    check_single_example(args.split, args.idx, args.candidate_k, args.include_comparison)


if __name__ == "__main__":
    main()
