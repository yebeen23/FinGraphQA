from pathlib import Path
import sys
import argparse
from collections import Counter

import matplotlib.pyplot as plt
import networkx as nx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
sys.path.append(str(CURRENT_DIR))

from graph_builder import build_inference_graph
from load_data import get_gold_evidence, load_split
from schema import *

NODE_COLORS = {
    NODE_PAGE: "#BDBDBD",
    NODE_QUESTION: "#FFD54F",
    NODE_TEXT: "#90CAF9",
    NODE_TABLE: "#B0BEC5",
    NODE_TABLE_ROW: "#A5D6A7",
    NODE_TABLE_COLUMN: "#AED581",
    NODE_TABLE_CELL: "#C8E6C9",
    NODE_FINANCIAL_FACT: "#FFCC80",
    NODE_QUANTITY: "#CE93D8",
    NODE_PERIOD: "#EF9A9A",
    NODE_UNIT: "#FFAB91",
    NODE_SCALE: "#FFE082",
    NODE_CURRENCY: "#BCAAA4",
    NODE_METRIC: "#80CBC4",
    NODE_OPERATION: "#F48FB1",
}

EDGE_RELATIONS_TO_HIDE = {
    REL_GREATER_OR_EQUAL_QUANTITY,
    REL_LATER_OR_EQUAL_PERIOD,
    REL_SAME_ROW_FACT,
    REL_SAME_COLUMN_FACT,
}


def make_filtered_graph(G: nx.MultiDiGraph, keep_dense_edges: bool = False) -> nx.MultiDiGraph:
    H = nx.MultiDiGraph()
    for node_id, attrs in G.nodes(data=True):
        H.add_node(node_id, **attrs)
    for src, dst, key, attrs in G.edges(keys=True, data=True):
        rel = attrs.get("rel", "")
        if not keep_dense_edges and rel in EDGE_RELATIONS_TO_HIDE:
            continue
        H.add_edge(src, dst, key=key, **attrs)
    return H


def get_subgraph_around_nodes(G: nx.MultiDiGraph, center_nodes: set[str], radius: int = 2) -> nx.MultiDiGraph:
    UG = G.to_undirected()
    selected = set()
    for node in center_nodes:
        if node not in UG:
            continue
        selected.update(nx.ego_graph(UG, node, radius=radius).nodes())
    return G.subgraph(selected).copy()


def get_candidate_operand_subgraph(G: nx.MultiDiGraph, q_id: str, top_k: int = 8) -> nx.MultiDiGraph:
    selected = {q_id}
    candidate_facts = []
    for _, dst, attrs in G.out_edges(q_id, data=True):
        if attrs.get("rel") == REL_CANDIDATE_OPERAND:
            candidate_facts.append((attrs.get("rank", 999), dst))
    candidate_facts.sort()
    for _, fact_id in candidate_facts[:top_k]:
        selected.add(fact_id)
        for _, dst, attrs in G.out_edges(fact_id, data=True):
            if attrs.get("rel") in {REL_SOURCE_CELL, REL_SOURCE_TEXT, REL_SOURCE_EVIDENCE, REL_HAS_QUANTITY, REL_HAS_METRIC, REL_HAS_PERIOD, REL_HAS_UNIT, REL_HAS_SCALE, REL_HAS_CURRENCY}:
                selected.add(dst)
        for src, _, attrs in G.in_edges(fact_id, data=True):
            if attrs.get("rel") in {REL_MENTIONED_IN, REL_APPEARS_IN}:
                selected.add(src)
    return G.subgraph(selected).copy()


def get_reasoning_trace_subgraph(G: nx.MultiDiGraph, q_id: str, gold_nodes: set[str], top_k: int = 8) -> nx.MultiDiGraph:
    selected = {q_id} | set(gold_nodes)
    selected.update(get_candidate_operand_subgraph(G, q_id, top_k=top_k).nodes())
    for gold in gold_nodes:
        if gold in G:
            for _, dst, attrs in G.out_edges(gold, data=True):
                if attrs.get("rel") in {REL_HAS_CELL, REL_HAS_QUANTITY, REL_HAS_METRIC, REL_HAS_PERIOD}:
                    selected.add(dst)
            for src, _, attrs in G.in_edges(gold, data=True):
                if attrs.get("rel") in {REL_SOURCE_EVIDENCE, REL_CONTAINS_ROW, REL_CONTAINS_TEXT}:
                    selected.add(src)
    return G.subgraph(selected).copy()


def shorten(text: str, max_len: int = 32) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[:max_len] + "..."


def node_label(G: nx.MultiDiGraph, node_id: str) -> str:
    attrs = G.nodes[node_id]
    node_type = attrs.get("type", "")
    if node_type == NODE_QUESTION:
        return "QUESTION"
    if node_type == NODE_FINANCIAL_FACT:
        metric = attrs.get("metric") or "?"
        period = attrs.get("period") or attrs.get("period_year") or "?"
        value = attrs.get("raw_value") or attrs.get("value")
        return shorten(f"FACT\n{metric}\n{period}: {value}", 55)
    if node_type == NODE_TEXT:
        return node_id
    if node_type == NODE_TABLE_ROW:
        return node_id
    if node_type == NODE_TABLE_COLUMN:
        return f"col[{attrs.get('col_index')}]\n{shorten(attrs.get('header', ''), 22)}"
    if node_type == NODE_TABLE_CELL:
        return f"cell[{attrs.get('row_index')},{attrs.get('col_index')}]"
    if node_type == NODE_QUANTITY:
        return shorten(attrs.get("raw", node_id), 18)
    if node_type == NODE_METRIC:
        return shorten(attrs.get("name", node_id), 28)
    if node_type == NODE_PERIOD:
        return str(attrs.get("value", node_id))
    if node_type == NODE_UNIT:
        return shorten(attrs.get("name", node_id), 18)
    if node_type == NODE_SCALE:
        return f"scale:{attrs.get('name', '')}"
    if node_type == NODE_OPERATION:
        return shorten(attrs.get("name", node_id), 24)
    return shorten(node_id, 24)


def draw_graph(G: nx.MultiDiGraph, title: str, highlight_nodes: set[str] | None = None):
    highlight_nodes = highlight_nodes or set()
    D = nx.DiGraph()
    edge_labels = {}
    for node_id, attrs in G.nodes(data=True):
        D.add_node(node_id, **attrs)
    for src, dst, attrs in G.edges(data=True):
        rel = attrs.get("rel", "")
        if D.has_edge(src, dst):
            old = edge_labels.get((src, dst), "")
            if rel and rel not in old:
                edge_labels[(src, dst)] = old + "|" + rel
        else:
            D.add_edge(src, dst)
            edge_labels[(src, dst)] = rel

    plt.figure(figsize=(20, 12))
    pos = nx.spring_layout(D, seed=42, k=1.3, iterations=120)

    node_colors = []
    node_sizes = []
    for node_id in D.nodes():
        node_type = D.nodes[node_id].get("type", "unknown")
        if node_id in highlight_nodes:
            node_colors.append("#FF3333")
            node_sizes.append(1800)
        elif node_type == NODE_FINANCIAL_FACT:
            node_colors.append(NODE_COLORS.get(node_type, "#EEEEEE"))
            node_sizes.append(1400)
        else:
            node_colors.append(NODE_COLORS.get(node_type, "#EEEEEE"))
            node_sizes.append(950)

    nx.draw_networkx_nodes(D, pos, node_color=node_colors, node_size=node_sizes, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_edges(D, pos, arrows=True, arrowstyle="-|>", arrowsize=12, width=0.9, alpha=0.65)
    labels = {node_id: node_label(D, node_id) for node_id in D.nodes()}
    nx.draw_networkx_labels(D, pos, labels=labels, font_size=8, font_family="sans-serif")
    if D.number_of_edges() <= 90:
        nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels, font_size=6)
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_graph_summary(G: nx.MultiDiGraph):
    node_types = Counter(attrs.get("type", "unknown") for _, attrs in G.nodes(data=True))
    edge_types = Counter(attrs.get("rel", "unknown") for _, _, attrs in G.edges(data=True))
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
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "private_test"])
    parser.add_argument("--idx", type=int, default=0)
    parser.add_argument(
        "--mode",
        default="candidate_operands",
        choices=["question", "gold", "question_gold", "candidate_operands", "reasoning_trace", "missed_gold"],
    )
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--keep-dense-edges", action="store_true")
    args = parser.parse_args()

    data = load_split(DATASET_DIR, args.split)
    example = data[args.idx]
    G, q_id = build_inference_graph(example)
    G_vis = make_filtered_graph(G, keep_dense_edges=args.keep_dense_edges)
    gold_nodes = get_gold_evidence(example)

    if args.mode == "question":
        subG = get_subgraph_around_nodes(G_vis, {q_id}, radius=args.radius)
        highlight = {q_id}
    elif args.mode == "gold":
        subG = get_subgraph_around_nodes(G_vis, gold_nodes, radius=args.radius)
        highlight = gold_nodes
    elif args.mode == "question_gold":
        subG = get_subgraph_around_nodes(G_vis, {q_id} | gold_nodes, radius=args.radius)
        highlight = {q_id} | gold_nodes
    elif args.mode == "candidate_operands":
        subG = get_candidate_operand_subgraph(G_vis, q_id, top_k=args.top_k)
        highlight = {q_id}
    elif args.mode == "reasoning_trace":
        subG = get_reasoning_trace_subgraph(G_vis, q_id, gold_nodes, top_k=args.top_k)
        highlight = {q_id} | gold_nodes
    else:
        # Gold nodes that are not reachable from the question within 4 hops.
        UG = G_vis.to_undirected()
        missed = set()
        for g in gold_nodes:
            try:
                if nx.shortest_path_length(UG, q_id, g) > 4:
                    missed.add(g)
            except Exception:
                missed.add(g)
        subG = get_subgraph_around_nodes(G_vis, {q_id} | missed, radius=args.radius)
        highlight = {q_id} | missed

    print("=" * 80)
    print("FinGraphQA Graph Visualization")
    print("=" * 80)
    print("Example ID:", example.get("id"))
    print("Filename:", example.get("filename"))
    print("Question:", example.get("qa", {}).get("question"))
    print("Gold evidence:", sorted(gold_nodes))
    print("Mode:", args.mode)
    print_graph_summary(subG)
    draw_graph(subG, title=f"{args.split}[{args.idx}] | {args.mode}", highlight_nodes=highlight)


if __name__ == "__main__":
    main()
