from pathlib import Path
from collections import Counter
import sys
import networkx as nx

# 현재 파일 위치:
# FINGRAPHQA/code/construction/check_graph.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

# 같은 폴더에 있는 graph_builder.py, load_data.py, preprocess.py import
sys.path.append(str(CURRENT_DIR))

from load_data import load_finqa_json

# build_inference_graph가 아직 없을 수도 있으므로 안전하게 처리
try:
    from graph_builder import build_inference_graph
    HAS_INFERENCE_BUILDER = True
except ImportError:
    from graph_builder import build_page_graph
    HAS_INFERENCE_BUILDER = False


def build_graph(example):
    """
    graph_builder.py에 build_inference_graph가 있으면 그것을 사용.
    없으면 build_page_graph만 사용해서 page graph 상태를 점검.
    """
    if HAS_INFERENCE_BUILDER:
        G, q_id = build_inference_graph(example)
        return G, q_id

    G = build_page_graph(example)
    q_id = None
    return G, q_id


def summarize_node_types(G: nx.MultiDiGraph) -> Counter:
    node_types = Counter()
    for _, attrs in G.nodes(data=True):
        node_types[attrs.get("type", "unknown")] += 1
    return node_types


def summarize_edge_types(G: nx.MultiDiGraph) -> Counter:
    edge_types = Counter()
    for _, _, attrs in G.edges(data=True):
        edge_types[attrs.get("rel", "unknown")] += 1
    return edge_types


def print_counter(title: str, counter: Counter):
    print(f"\n[{title}]")
    if not counter:
        print("  EMPTY")
        return

    for k, v in counter.most_common():
        print(f"  {k}: {v}")


def check_required_node_types(G: nx.MultiDiGraph):
    required = {
        "page",
        "text_sentence",
        "table_row",
        "table_cell",
        "quantity",
        "date",
        "unit",
        "metric",
    }

    # build_inference_graph가 있으면 question/formula_cue도 기대
    if HAS_INFERENCE_BUILDER:
        required.add("question")
        # formula_cue는 모든 질문에서 반드시 나오진 않으므로 hard fail로 보지 않음

    node_types = summarize_node_types(G)
    missing = sorted([t for t in required if node_types.get(t, 0) == 0])

    print("\n[Required node type check]")
    if missing:
        print("  STATUS: WARN")
        print("  Missing node types:", missing)
    else:
        print("  STATUS: PASS")


def check_required_edge_types(G: nx.MultiDiGraph):
    required = {
        "contains_text",
        "contains_table_row",
        "has_cell",
        "belongs_to_row",
        "same_row",
        "same_column",
        "has_quantity",
        "appears_in",
    }

    if HAS_INFERENCE_BUILDER:
        # question-aware graph에서 기대하는 edge
        # 단, 질문에 year/metric/formula가 없으면 일부는 없을 수 있음
        optional_question_edges = {
            "asks_year",
            "asks_unit",
            "asks_metric",
            "asks_formula",
            "greater_or_equal_quantity",
            "later_or_equal_date",
        }
    else:
        optional_question_edges = set()

    edge_types = summarize_edge_types(G)
    missing = sorted([t for t in required if edge_types.get(t, 0) == 0])

    print("\n[Required edge type check]")
    if missing:
        print("  STATUS: WARN")
        print("  Missing edge types:", missing)
    else:
        print("  STATUS: PASS")

    if optional_question_edges:
        print("\n[Optional question-aware edge types]")
        for edge_type in sorted(optional_question_edges):
            print(f"  {edge_type}: {edge_types.get(edge_type, 0)}")


def check_gold_evidence_nodes(G: nx.MultiDiGraph, example: dict):
    qa = example.get("qa", {})
    gold = set(qa.get("gold_inds", {}).keys())

    print("\n[Gold evidence node check]")

    if not gold:
        print("  No gold evidence in this example.")
        return

    missing = sorted([node_id for node_id in gold if node_id not in G.nodes])

    print("  Gold evidence:", sorted(gold))
    print("  Missing:", missing)

    if missing:
        print("  STATUS: FAIL")
    else:
        print("  STATUS: PASS")


def inspect_gold_evidence(G: nx.MultiDiGraph, example: dict):
    qa = example.get("qa", {})
    gold = set(qa.get("gold_inds", {}).keys())

    print("\n[Gold evidence detail]")

    for node_id in sorted(gold):
        if node_id not in G.nodes:
            print(f"\n  {node_id}: NOT FOUND")
            continue

        attrs = G.nodes[node_id]
        print(f"\n  {node_id}")
        print(f"    type: {attrs.get('type')}")
        print(f"    text: {attrs.get('text', '')[:250]}")

        neighbors = list(G.neighbors(node_id))
        print(f"    outgoing neighbors: {len(neighbors)}")

        shown = 0
        for _, dst, edge_attrs in G.out_edges(node_id, data=True):
            print(f"      --{edge_attrs.get('rel')}--> {dst}")
            shown += 1
            if shown >= 10:
                print("      ...")
                break


def inspect_question_node(G: nx.MultiDiGraph, q_id: str | None):
    print("\n[Question node check]")

    if q_id is None:
        print("  STATUS: WARN")
        print("  No question node. build_inference_graph() is probably not implemented yet.")
        return

    if q_id not in G.nodes:
        print("  STATUS: FAIL")
        print(f"  Question node not found: {q_id}")
        return

    print("  STATUS: PASS")
    print("  Question node:", q_id)
    print("  Question text:", G.nodes[q_id].get("text", ""))

    print("\n  Question outgoing edges:")
    count = 0
    for _, dst, attrs in G.out_edges(q_id, data=True):
        print(f"    --{attrs.get('rel')}--> {dst}")
        count += 1

    if count == 0:
        print("    No outgoing edges from question node.")


def check_single_example(split: str = "train", idx: int = 0):
    data_path = DATASET_DIR / f"{split}.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    data = load_finqa_json(str(data_path))

    if idx < 0 or idx >= len(data):
        raise IndexError(f"idx={idx} out of range. Dataset size={len(data)}")

    example = data[idx]

    G, q_id = build_graph(example)

    print("=" * 80)
    print("Graph Check")
    print("=" * 80)
    print("Project root:", PROJECT_ROOT)
    print("Dataset file:", data_path)
    print("Split:", split)
    print("Index:", idx)
    print("Example ID:", example.get("id"))
    print("Filename:", example.get("filename"))
    print("Question:", example.get("qa", {}).get("question"))
    print("Using build_inference_graph:", HAS_INFERENCE_BUILDER)

    print("\n[Graph size]")
    print("  Nodes:", G.number_of_nodes())
    print("  Edges:", G.number_of_edges())

    print_counter("Node types", summarize_node_types(G))
    print_counter("Edge types", summarize_edge_types(G))

    check_required_node_types(G)
    check_required_edge_types(G)
    inspect_question_node(G, q_id)
    check_gold_evidence_nodes(G, example)
    inspect_gold_evidence(G, example)

    print("\nDone.")


if __name__ == "__main__":
    # 여기서 보고 싶은 split/index 바꾸면 됨
    check_single_example(split="train", idx=0)