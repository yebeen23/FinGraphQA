from pathlib import Path
import sys
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx

# 현재 파일 위치:
# FINGRAPHQA/code/construction/visualize_graph_subset.py
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"

sys.path.append(str(CURRENT_DIR))

from load_data import load_finqa_json
from graph_builder import build_inference_graph


NODE_COLORS = {
    "page": "#BDBDBD",
    "question": "#FFD54F",
    "text_sentence": "#90CAF9",
    "table_row": "#A5D6A7",
    "table_cell": "#C8E6C9",
    "quantity": "#CE93D8",
    "date": "#EF9A9A",
    "unit": "#FFAB91",
    "metric": "#80CBC4",
    "formula_cue": "#F48FB1",
}


EDGE_RELATIONS_TO_HIDE = {
    # 너무 많아서 기본 시각화에서는 숨김
    "greater_or_equal_quantity",
    "later_or_equal_date",
}


def make_filtered_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Visualization용 graph.
    너무 많은 comparison edge는 제거한다.
    """
    H = nx.MultiDiGraph()

    for node_id, attrs in G.nodes(data=True):
        H.add_node(node_id, **attrs)

    for src, dst, key, attrs in G.edges(keys=True, data=True):
        rel = attrs.get("rel", "")

        if rel in EDGE_RELATIONS_TO_HIDE:
            continue

        H.add_edge(src, dst, key=key, **attrs)

    return H


def get_subgraph_around_nodes(
    G: nx.MultiDiGraph,
    center_nodes: set[str],
    radius: int = 2,
) -> nx.MultiDiGraph:
    """
    center_nodes 주변 radius-hop subgraph 추출.
    """
    UG = G.to_undirected()

    selected = set()
    for node in center_nodes:
        if node not in UG:
            continue
        selected.update(nx.ego_graph(UG, node, radius=radius).nodes())

    return G.subgraph(selected).copy()


def shorten(text: str, max_len: int = 28) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


def node_label(G: nx.MultiDiGraph, node_id: str) -> str:
    attrs = G.nodes[node_id]
    node_type = attrs.get("type", "")

    if node_type == "question":
        return "QUESTION"

    if node_type == "text_sentence":
        return node_id

    if node_type == "table_row":
        return node_id

    if node_type == "table_cell":
        row = attrs.get("row_index", "")
        col = attrs.get("col_index", "")
        return f"cell[{row},{col}]"

    if node_type == "quantity":
        return shorten(attrs.get("raw", node_id), 18)

    if node_type == "metric":
        return shorten(attrs.get("name", node_id), 24)

    if node_type == "date":
        return str(attrs.get("value", node_id))

    if node_type == "unit":
        return shorten(attrs.get("name", node_id), 18)

    if node_type == "formula_cue":
        return shorten(attrs.get("name", node_id), 24)

    return shorten(node_id, 24)


def draw_graph(
    G: nx.MultiDiGraph,
    title: str,
    highlight_nodes: set[str] | None = None,
):
    highlight_nodes = highlight_nodes or set()

    # MultiDiGraph는 matplotlib edge label이 지저분해져서 DiGraph로 압축
    D = nx.DiGraph()

    for node_id, attrs in G.nodes(data=True):
        D.add_node(node_id, **attrs)

    edge_labels = {}

    for src, dst, attrs in G.edges(data=True):
        rel = attrs.get("rel", "")

        if D.has_edge(src, dst):
            old = edge_labels.get((src, dst), "")
            if rel and rel not in old:
                edge_labels[(src, dst)] = old + "|" + rel
        else:
            D.add_edge(src, dst)
            edge_labels[(src, dst)] = rel

    plt.figure(figsize=(18, 11))

    pos = nx.spring_layout(D, seed=42, k=1.2, iterations=100)

    node_colors = []
    node_sizes = []

    for node_id in D.nodes():
        node_type = D.nodes[node_id].get("type", "unknown")

        if node_id in highlight_nodes:
            node_colors.append("#FF3333")
            node_sizes.append(1600)
        else:
            node_colors.append(NODE_COLORS.get(node_type, "#EEEEEE"))
            node_sizes.append(900)

    nx.draw_networkx_nodes(
        D,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="black",
        linewidths=0.8,
    )

    nx.draw_networkx_edges(
        D,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=12,
        width=0.9,
        alpha=0.65,
    )

    labels = {node_id: node_label(D, node_id) for node_id in D.nodes()}

    nx.draw_networkx_labels(
        D,
        pos,
        labels=labels,
        font_size=8,
        font_family="sans-serif",
    )

    # edge label이 너무 많으면 그림이 망가짐.
    # 80개 이하일 때만 표시.
    if D.number_of_edges() <= 80:
        nx.draw_networkx_edge_labels(
            D,
            pos,
            edge_labels=edge_labels,
            font_size=6,
        )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_graph_summary(G: nx.MultiDiGraph):
    node_types = Counter()
    edge_types = Counter()

    for _, attrs in G.nodes(data=True):
        node_types[attrs.get("type", "unknown")] += 1

    for _, _, attrs in G.edges(data=True):
        edge_types[attrs.get("rel", "unknown")] += 1

    print("\n[Subgraph size]")
    print("  Nodes:", G.number_of_nodes())
    print("  Edges:", G.number_of_edges())

    print("\n[Node types]")
    for k, v in node_types.most_common():
        print(f"  {k}: {v}")

    print("\n[Edge types]")
    for k, v in edge_types.most_common():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test"])
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument(
        "--mode",
        default="question",
        choices=["question", "gold", "question_gold"],
        help="Which part of the graph to visualize.",
    )
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    data_path = DATASET_DIR / f"{args.split}.json"
    data = load_finqa_json(str(data_path))

    example = data[args.idx]
    G, q_id = build_inference_graph(example)

    # comparison edge 제거한 visualization graph
    G_vis = make_filtered_graph(G)

    gold_nodes = set(example.get("qa", {}).get("gold_inds", {}).keys())

    if args.mode == "question":
        centers = {q_id}
        highlight = {q_id}
    elif args.mode == "gold":
        centers = gold_nodes
        highlight = gold_nodes
    else:
        centers = {q_id} | gold_nodes
        highlight = {q_id} | gold_nodes

    subG = get_subgraph_around_nodes(
        G_vis,
        center_nodes=centers,
        radius=args.radius,
    )

    print("=" * 80)
    print("Graph Visualization")
    print("=" * 80)
    print("Example ID:", example.get("id"))
    print("Filename:", example.get("filename"))
    print("Question:", example.get("qa", {}).get("question"))
    print("Gold evidence:", sorted(gold_nodes))
    print("Mode:", args.mode)
    print("Radius:", args.radius)

    print_graph_summary(subG)

    draw_graph(
        subG,
        title=f"{args.split}[{args.idx}] | {args.mode} | radius={args.radius}",
        highlight_nodes=highlight,
    )


if __name__ == "__main__":
    main()