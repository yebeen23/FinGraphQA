import networkx as nx
from typing import Any, Dict, List

from preprocess import (
    get_all_text,
    normalize_text,
    extract_numbers,
    parse_number,
    extract_years,
    detect_units,
    extract_metrics,
    extract_formula_cues,
    linearize_table_row,
)


def add_node_safe(G: nx.MultiDiGraph, node_id: str, **attrs) -> None:
    if node_id not in G:
        G.add_node(node_id, **attrs)
    else:
        G.nodes[node_id].update(attrs)


def add_edge_safe(G: nx.MultiDiGraph, src: str, dst: str, rel: str, **attrs) -> None:
    G.add_edge(src, dst, rel=rel, **attrs)


def build_page_graph(example: Dict[str, Any]) -> nx.MultiDiGraph:
    """
    Builds inference graph from:
    - filename
    - pre_text
    - post_text
    - table

    Does NOT use gold evidence/program/answer.
    """
    G = nx.MultiDiGraph()

    filename = example.get("filename", "unknown")
    page_id = f"page:{filename}"

    add_node_safe(
        G,
        page_id,
        type="page",
        filename=filename,
    )

    # 1. Text sentence nodes
    all_text = get_all_text(example)

    for i, sent in enumerate(all_text):
        text_id = f"text_{i}"
        sent_norm = normalize_text(sent)

        add_node_safe(
            G,
            text_id,
            type="text_sentence",
            text=sent_norm,
            raw_text=sent,
            position=i,
        )

        add_edge_safe(G, page_id, text_id, "contains_text")

        # Adjacent text edges
        if i > 0:
            prev_id = f"text_{i-1}"
            add_edge_safe(G, prev_id, text_id, "adjacent_text")
            add_edge_safe(G, text_id, prev_id, "adjacent_text")

        # Extract quantities, years, units, metrics
        _attach_text_features(G, text_id, sent_norm)

    # 2. Table row and cell nodes
    table = example.get("table", [])
    if table:
        _attach_table_graph(G, page_id, table)

    return G


def _attach_text_features(G: nx.MultiDiGraph, text_id: str, text: str) -> None:
    # numbers
    for raw_num in extract_numbers(text):
        value = parse_number(raw_num)
        if value is None:
            continue

        num_id = f"num:{text_id}:{raw_num}"
        add_node_safe(
            G,
            num_id,
            type="quantity",
            raw=raw_num,
            value=value,
            source=text_id,
        )
        add_edge_safe(G, text_id, num_id, "has_quantity")
        add_edge_safe(G, num_id, text_id, "appears_in")

    # years
    for year in extract_years(text):
        year_id = f"year:{year}"
        add_node_safe(G, year_id, type="date", value=int(year))
        add_edge_safe(G, text_id, year_id, "mentions_year")

    # units
    for unit in detect_units(text):
        unit_id = f"unit:{unit}"
        add_node_safe(G, unit_id, type="unit", name=unit)
        add_edge_safe(G, text_id, unit_id, "mentions_unit")

    # metrics
    for metric in extract_metrics(text):
        metric_id = f"metric:{metric}"
        add_node_safe(G, metric_id, type="metric", name=metric)
        add_edge_safe(G, text_id, metric_id, "mentions_metric")
        add_edge_safe(G, metric_id, text_id, "mentioned_in")


def _attach_table_graph(G: nx.MultiDiGraph, page_id: str, table: List[List[str]]) -> None:
    headers = table[0]

    # Header row node
    for row_idx, row in enumerate(table):
        row_id = f"table_{row_idx}"
        row_text = normalize_text(linearize_table_row(table, row_idx))

        add_node_safe(
            G,
            row_id,
            type="table_row",
            row_index=row_idx,
            text=row_text,
            raw_row=row,
        )

        add_edge_safe(G, page_id, row_id, "contains_table_row")

    # Cell nodes
    for row_idx, row in enumerate(table):
        row_id = f"table_{row_idx}"
        row_header = row[0] if row else ""

        for col_idx, value in enumerate(row):
            col_header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"

            cell_id = f"cell:table_{row_idx}:{col_idx}"
            cell_text = normalize_text(f"{row_header} {col_header} {value}")

            add_node_safe(
                G,
                cell_id,
                type="table_cell",
                row_index=row_idx,
                col_index=col_idx,
                row_header=normalize_text(row_header),
                col_header=normalize_text(col_header),
                value=value,
                text=cell_text,
            )

            add_edge_safe(G, row_id, cell_id, "has_cell")
            add_edge_safe(G, cell_id, row_id, "belongs_to_row")

            # same row edges
            for other_col_idx in range(len(row)):
                if other_col_idx == col_idx:
                    continue
                other_cell_id = f"cell:table_{row_idx}:{other_col_idx}"
                add_edge_safe(G, cell_id, other_cell_id, "same_row")

            # same column edges
            for other_row_idx in range(len(table)):
                if other_row_idx == row_idx:
                    continue
                if col_idx < len(table[other_row_idx]):
                    other_cell_id = f"cell:table_{other_row_idx}:{col_idx}"
                    add_edge_safe(G, cell_id, other_cell_id, "same_column")

            # Attach extracted features to cell
            _attach_cell_features(G, cell_id, cell_text, value, row_header, col_header)


def _attach_cell_features(
    G: nx.MultiDiGraph,
    cell_id: str,
    cell_text: str,
    value: str,
    row_header: str,
    col_header: str,
) -> None:
    # numbers from value and cell context
    for raw_num in extract_numbers(str(value)):
        parsed = parse_number(raw_num)
        if parsed is None:
            continue

        num_id = f"num:{cell_id}:{raw_num}"
        add_node_safe(
            G,
            num_id,
            type="quantity",
            raw=raw_num,
            value=parsed,
            source=cell_id,
        )

        add_edge_safe(G, cell_id, num_id, "has_quantity")
        add_edge_safe(G, num_id, cell_id, "appears_in")

    # years from row/header/cell
    for year in extract_years(cell_text):
        year_id = f"year:{year}"
        add_node_safe(G, year_id, type="date", value=int(year))
        add_edge_safe(G, cell_id, year_id, "has_year")
        add_edge_safe(G, year_id, cell_id, "year_of")

    # units
    unit_context = f"{row_header} {col_header} {value}"
    for unit in detect_units(unit_context):
        unit_id = f"unit:{unit}"
        add_node_safe(G, unit_id, type="unit", name=unit)
        add_edge_safe(G, cell_id, unit_id, "has_unit")

    # metrics from row header and col header
    metric_context = f"{row_header} {col_header}"
    for metric in extract_metrics(metric_context):
        metric_id = f"metric:{metric}"
        add_node_safe(G, metric_id, type="metric", name=metric)
        add_edge_safe(G, cell_id, metric_id, "mentions_metric")
        add_edge_safe(G, metric_id, cell_id, "mentioned_in")


def attach_question_node(G: nx.MultiDiGraph, example: Dict[str, Any]) -> str:
    """
    Attach question node and question-derived feature nodes.
    This uses only qa.question, not gold evidence/program/answer.
    """
    qa = example.get("qa", {})
    question = qa.get("question", "")
    example_id = example.get("id", "unknown")

    q_id = f"question:{example_id}"
    q_norm = normalize_text(question)

    add_node_safe(
        G,
        q_id,
        type="question",
        text=q_norm,
        raw_text=question,
    )

    # Question -> year/date
    for year in extract_years(q_norm):
        year_id = f"year:{year}"
        add_node_safe(G, year_id, type="date", value=int(year))
        add_edge_safe(G, q_id, year_id, "asks_year")
        add_edge_safe(G, year_id, q_id, "year_asked_by")

    # Question -> unit
    for unit in detect_units(q_norm):
        unit_id = f"unit:{unit}"
        add_node_safe(G, unit_id, type="unit", name=unit)
        add_edge_safe(G, q_id, unit_id, "asks_unit")
        add_edge_safe(G, unit_id, q_id, "unit_asked_by")

    # Question -> metric
    for metric in extract_metrics(q_norm):
        metric_id = f"metric:{metric}"
        add_node_safe(G, metric_id, type="metric", name=metric)
        add_edge_safe(G, q_id, metric_id, "asks_metric")
        add_edge_safe(G, metric_id, q_id, "metric_asked_by")

    # Question -> formula cue
    for cue in extract_formula_cues(q_norm):
        cue_id = f"formula:{cue}"
        add_node_safe(G, cue_id, type="formula_cue", name=cue)
        add_edge_safe(G, q_id, cue_id, "asks_formula")
        add_edge_safe(G, cue_id, q_id, "formula_asked_by")

    return q_id


def add_quantity_comparison_edges(G: nx.MultiDiGraph) -> None:
    """
    Add directed edges between quantity nodes.
    If value_i >= value_j, add value_i -> value_j.
    """
    quantities = [
        (node_id, attrs)
        for node_id, attrs in G.nodes(data=True)
        if attrs.get("type") == "quantity" and attrs.get("value") is not None
    ]

    for i, (id_i, attrs_i) in enumerate(quantities):
        for j, (id_j, attrs_j) in enumerate(quantities):
            if i == j:
                continue

            if attrs_i["value"] >= attrs_j["value"]:
                add_edge_safe(
                    G,
                    id_i,
                    id_j,
                    "greater_or_equal_quantity",
                )


def add_date_comparison_edges(G: nx.MultiDiGraph) -> None:
    """
    Add directed edges between date/year nodes.
    If date_i >= date_j, add date_i -> date_j.
    """
    dates = [
        (node_id, attrs)
        for node_id, attrs in G.nodes(data=True)
        if attrs.get("type") == "date" and attrs.get("value") is not None
    ]

    for i, (id_i, attrs_i) in enumerate(dates):
        for j, (id_j, attrs_j) in enumerate(dates):
            if i == j:
                continue

            if attrs_i["value"] >= attrs_j["value"]:
                add_edge_safe(
                    G,
                    id_i,
                    id_j,
                    "later_or_equal_date",
                )


def build_inference_graph(example: Dict[str, Any]) -> tuple[nx.MultiDiGraph, str]:
    """
    Build complete inference graph for one FinQA example.

    This graph includes:
    - page/text/table/cell/quantity/date/unit/metric nodes
    - question node
    - formula cue nodes
    - quantity/date comparison edges

    It does NOT include gold evidence, gold program, or answer.
    """
    G = build_page_graph(example)
    q_id = attach_question_node(G, example)

    add_quantity_comparison_edges(G)
    add_date_comparison_edges(G)

    return G, q_id