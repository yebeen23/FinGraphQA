import math
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from preprocess import (
    build_evidence_candidates,
    detect_currency,
    detect_scale,
    detect_units,
    extract_formula_cues,
    extract_metric_candidates_from_header,
    extract_metric_candidates_from_question,
    extract_metrics,
    extract_periods,
    get_all_text,
    get_financial_numbers,
    infer_table_scale,
    is_period_like,
    jaccard,
    linearize_table_row,
    normalize_metric_phrase,
    normalize_text,
    token_set,
)
from schema import *


def _safe_id(text: Any, max_len: int = 80) -> str:
    s = normalize_text(text)
    keep = []
    for ch in s:
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_", ":", "."}:
            keep.append("_")
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out[:max_len] or "empty"


def add_node_safe(G: nx.MultiDiGraph, node_id: str, **attrs) -> None:
    if node_id not in G:
        G.add_node(node_id, **attrs)
    else:
        # Preserve existing useful fields while allowing newer fields to update.
        G.nodes[node_id].update({k: v for k, v in attrs.items() if v is not None})


def add_edge_safe(G: nx.MultiDiGraph, src: str, dst: str, rel: str, **attrs) -> None:
    G.add_edge(src, dst, rel=rel, **attrs)


def add_metric_node(G: nx.MultiDiGraph, metric: str) -> str:
    metric_norm = normalize_metric_phrase(metric) or normalize_text(metric)
    node_id = f"metric:{_safe_id(metric_norm)}"
    add_node_safe(G, node_id, type=NODE_METRIC, name=metric_norm, text=metric_norm)
    return node_id


def add_period_node(G: nx.MultiDiGraph, period: Dict[str, Any]) -> str:
    p_type = period.get("type", "period")
    value = period.get("value")
    node_id = f"period:{p_type}:{_safe_id(value)}"
    add_node_safe(
        G,
        node_id,
        type=NODE_PERIOD,
        period_type=p_type,
        value=value,
        year=period.get("year"),
        raw=period.get("raw"),
        text=str(value),
    )
    return node_id


def add_unit_node(G: nx.MultiDiGraph, unit: str) -> str:
    node_id = f"unit:{_safe_id(unit)}"
    add_node_safe(G, node_id, type=NODE_UNIT, name=unit, text=unit)
    return node_id


def add_scale_node(G: nx.MultiDiGraph, scale: str) -> str:
    node_id = f"scale:{_safe_id(scale)}"
    add_node_safe(G, node_id, type=NODE_SCALE, name=scale, text=scale)
    return node_id


def add_currency_node(G: nx.MultiDiGraph, currency: str) -> str:
    node_id = f"currency:{_safe_id(currency)}"
    add_node_safe(G, node_id, type=NODE_CURRENCY, name=currency, text=currency)
    return node_id



GENERIC_FINANCE_TERMS = {
    "income", "expense", "expenses", "asset", "assets", "liability", "liabilities",
    "cash", "revenue", "cost", "debt", "shares", "stock", "total", "net",
    "value", "fair", "amount", "payment", "payments", "basis", "managed",
    "gain", "loss", "price", "tax", "taxes", "rate", "rates", "year", "years",
}


def _metric_tokens(text: Optional[str]) -> set[str]:
    if not text:
        return set()
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]+", normalize_text(text))
        if len(t) > 2
    }


def _informative_tokens(text: Optional[str]) -> set[str]:
    return _metric_tokens(text) - GENERIC_FINANCE_TERMS


def _raw_float(raw: Any) -> Optional[float]:
    if raw is None:
        return None
    s = str(raw).lower()
    s = s.replace("$", "").replace(",", "").replace("%", "")
    s = s.replace("(", "").replace(")", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def _is_integer_value(x: float) -> bool:
    return float(x).is_integer()


def _is_probably_date_number(raw: Any, period_value: Any, text: str = "") -> bool:
    """
    Detect facts whose numeric value is only a date component, not a financial operand.

    Examples to penalize:
      March 31, 2012 -> raw 31 or 2012
      December 30, 2017 -> raw 30 or 2017
      table/text facts whose value equals their period year
    """
    if raw is None:
        return False

    raw_s = str(raw)

    # Currency and percentages are usually real financial operands.
    if "$" in raw_s or "%" in raw_s:
        return False

    value = _raw_float(raw_s)
    if value is None:
        return False

    # 4-digit years are periods, not operands, unless explicitly currency/percent.
    if 1900 <= value <= 2099 and _is_integer_value(value):
        return True

    period_s = str(period_value or "")
    if period_s:
        iv = str(int(value)) if _is_integer_value(value) else str(value)
        # YYYY-MM-DD contains day/month/year; matching value is suspicious.
        if re.search(r"\d{4}-\d{2}-\d{2}", period_s) and iv in period_s:
            return True

    # Day-like values near date-style period are suspicious.
    if 1 <= value <= 31 and _is_integer_value(value):
        if re.search(r"\d{4}-\d{2}-\d{2}", period_s):
            return True

    return False


def _metric_match_score(question_metric: str, fact_metric: str) -> float:
    """
    More conservative metric matching:
    - exact and phrase matches are strong
    - informative-token overlap is useful
    - generic-only overlap such as "income" or "expense" is weak
    """
    if not question_metric or not fact_metric:
        return 0.0

    q = normalize_text(question_metric)
    f = normalize_text(fact_metric)

    if not q or not f:
        return 0.0

    q_tokens = _metric_tokens(q)
    f_tokens = _metric_tokens(f)

    if not q_tokens or not f_tokens:
        return 0.0

    q_info = q_tokens - GENERIC_FINANCE_TERMS
    f_info = f_tokens - GENERIC_FINANCE_TERMS
    overlap = q_tokens & f_tokens
    info_overlap = q_info & f_info

    if q == f:
        return 1.0

    if f in q or q in f:
        if info_overlap:
            return 0.90
        return 0.30

    if info_overlap:
        denom = max(len(q_info), len(f_info), 1)
        return min(0.80, 0.25 + 0.55 * (len(info_overlap) / denom))

    # Generic-only overlap should not dominate.
    if overlap:
        return 0.12

    return 0.0


def _best_metric_score(q_text: str, q_metrics: List[str], fact_metric: str, fact_text: str) -> float:
    scores = []

    if fact_metric:
        for qm in q_metrics:
            scores.append(_metric_match_score(qm, fact_metric))
        scores.append(_metric_match_score(q_text, fact_metric))

    metric_score = max(scores) if scores else 0.0

    # Weak fallback: fact text has lexical overlap with question.
    # Keep this weak so date/context sentences do not dominate real operands.
    q_tokens = token_set(q_text)
    f_text_tokens = token_set(fact_text)
    if q_tokens and f_text_tokens:
        overlap = q_tokens & f_text_tokens
        informative_overlap = overlap - GENERIC_FINANCE_TERMS
        if informative_overlap:
            metric_score = max(metric_score, min(0.18, 0.04 * len(informative_overlap)))
        elif overlap:
            metric_score = max(metric_score, 0.04)

    return metric_score

def add_quantity_node(
    G: nx.MultiDiGraph,
    source_id: str,
    parsed: Dict[str, Any],
    occurrence: int,
) -> str:
    raw_id = _safe_id(parsed.get("raw", "num"), max_len=40)
    node_id = f"quantity:{source_id}:{occurrence}:{raw_id}"
    add_node_safe(
        G,
        node_id,
        type=NODE_QUANTITY,
        raw=parsed.get("raw"),
        value=parsed.get("value"),
        normalized_value=parsed.get("normalized_value"),
        display_value=parsed.get("display_value"),
        is_percent=parsed.get("is_percent"),
        currency=parsed.get("currency"),
        scale=parsed.get("scale"),
        unit=parsed.get("unit"),
        sign=parsed.get("sign"),
        source=source_id,
    )
    add_edge_safe(G, source_id, node_id, REL_HAS_QUANTITY)
    add_edge_safe(G, node_id, source_id, REL_APPEARS_IN)
    return node_id


def build_source_graph(example: Dict[str, Any]) -> nx.MultiDiGraph:
    """
    Build only the source structure: page, text sentences, table, rows, columns, cells.
    No gold evidence/program/answer is used.
    """
    G = nx.MultiDiGraph()
    filename = example.get("filename", "unknown")
    page_id = f"page:{filename}"

    add_node_safe(G, page_id, type=NODE_PAGE, filename=filename, text=str(filename))

    all_text = get_all_text(example)
    for i, sent in enumerate(all_text):
        text_id = f"text_{i}"
        sent_norm = normalize_text(sent)
        add_node_safe(
            G,
            text_id,
            type=NODE_TEXT,
            text=sent_norm,
            raw_text=sent,
            position=i,
            evidence_id=text_id,
        )
        add_edge_safe(G, page_id, text_id, REL_CONTAINS_TEXT)
        if i > 0:
            prev_id = f"text_{i-1}"
            add_edge_safe(G, prev_id, text_id, REL_ADJACENT_TEXT)
            add_edge_safe(G, text_id, prev_id, REL_ADJACENT_TEXT)

    table = example.get("table", []) or []
    if table:
        table_id = "table"
        add_node_safe(G, table_id, type=NODE_TABLE, text="table", n_rows=len(table), n_cols=max(len(r) for r in table))
        add_edge_safe(G, page_id, table_id, REL_CONTAINS_TABLE)
        _attach_table_source_graph(G, table_id, table)

    return G


def _attach_table_source_graph(G: nx.MultiDiGraph, table_id: str, table: List[List[str]]) -> None:
    headers = table[0] if table else []

    # Row nodes preserve FinQA evidence ids: table_0, table_1, ...
    for row_idx, row in enumerate(table):
        row_id = f"table_{row_idx}"
        row_text = normalize_text(linearize_table_row(table, row_idx))
        add_node_safe(
            G,
            row_id,
            type=NODE_TABLE_ROW,
            row_index=row_idx,
            text=row_text,
            raw_row=row,
            evidence_id=row_id,
        )
        add_edge_safe(G, table_id, row_id, REL_CONTAINS_ROW)
        if row_idx > 0:
            add_edge_safe(G, f"table_{row_idx-1}", row_id, REL_NEXT_ROW)

    # Column nodes avoid dense same-column cell-to-cell edges.
    for col_idx, header in enumerate(headers):
        col_id = f"column_{col_idx}"
        header_norm = normalize_text(header)
        add_node_safe(
            G,
            col_id,
            type=NODE_TABLE_COLUMN,
            col_index=col_idx,
            header=header_norm,
            raw_header=header,
            text=header_norm,
        )
        add_edge_safe(G, table_id, col_id, REL_CONTAINS_COLUMN)
        if col_idx > 0:
            add_edge_safe(G, f"column_{col_idx-1}", col_id, REL_NEXT_COLUMN)

    # Cell nodes.
    for row_idx, row in enumerate(table):
        row_id = f"table_{row_idx}"
        row_header = row[0] if row else ""
        for col_idx, value in enumerate(row):
            col_header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
            col_id = f"column_{col_idx}"
            cell_id = f"cell:table_{row_idx}:{col_idx}"
            cell_text = normalize_text(f"{row_header} {col_header} {value}")
            add_node_safe(
                G,
                cell_id,
                type=NODE_TABLE_CELL,
                row_index=row_idx,
                col_index=col_idx,
                row_header=normalize_text(row_header),
                col_header=normalize_text(col_header),
                value=value,
                text=cell_text,
                evidence_id=row_id,
            )
            add_edge_safe(G, row_id, cell_id, REL_HAS_CELL)
            add_edge_safe(G, cell_id, row_id, REL_BELONGS_TO_ROW)
            add_edge_safe(G, col_id, cell_id, REL_HAS_CELL)
            add_edge_safe(G, cell_id, col_id, REL_BELONGS_TO_COLUMN)


def build_fact_graph(G: nx.MultiDiGraph, example: Dict[str, Any]) -> None:
    """
    Add financial facts from text and table sources.
    A financial_fact node is the main reasoning unit: metric + period + quantity + unit/scale + source.
    """
    inherited_table_scale = infer_table_scale(example)
    _attach_text_facts(G, example, inherited_table_scale)
    _attach_table_facts(G, example, inherited_table_scale)
    add_temporal_edges(G)
    add_contextual_fact_edges(G)
    add_table_text_bridge_edges(G, example)


def _attach_text_facts(G: nx.MultiDiGraph, example: Dict[str, Any], inherited_scale: Optional[str]) -> None:
    all_text = get_all_text(example)
    for i, sent in enumerate(all_text):
        text_id = f"text_{i}"
        sent_norm = normalize_text(sent)
        if text_id not in G:
            continue

        periods = extract_periods(sent_norm)
        units = detect_units(sent_norm)
        scale = detect_scale(sent_norm) or inherited_scale
        currency = detect_currency(sent_norm)
        metrics = extract_metrics(sent_norm)
        nums = get_financial_numbers(sent_norm, context=sent_norm, inherited_scale=scale)

        # Attach entities mentioned by text, even when no fact is created.
        for p in periods:
            pid = add_period_node(G, p)
            add_edge_safe(G, text_id, pid, REL_HAS_PERIOD)
            add_edge_safe(G, pid, text_id, REL_MENTIONED_IN)
        for unit in units:
            uid = add_unit_node(G, unit)
            add_edge_safe(G, text_id, uid, REL_HAS_UNIT)
        if scale:
            sid = add_scale_node(G, scale)
            add_edge_safe(G, text_id, sid, REL_HAS_SCALE)
        if currency:
            cid = add_currency_node(G, currency)
            add_edge_safe(G, text_id, cid, REL_HAS_CURRENCY)
        for metric in metrics:
            mid = add_metric_node(G, metric)
            add_edge_safe(G, text_id, mid, REL_HAS_METRIC)
            add_edge_safe(G, mid, text_id, REL_MENTIONED_IN)

        seen_values = set()
        for occ, parsed in enumerate(nums):
            value_key = (parsed.get("normalized_value"), parsed.get("is_percent"))
            if value_key in seen_values:
                continue
            seen_values.add(value_key)
            raw_digits = str(parsed.get("raw", "")).strip().replace(",", "").replace("$", "")
            is_plain_year = raw_digits.isdigit() and len(raw_digits) == 4 and 1900 <= int(raw_digits) <= 2099
            metric = _choose_text_metric(metrics, sent_norm)
            if is_plain_year and not metric:
                continue
            qid = add_quantity_node(G, text_id, parsed, occ)
            period = periods[0] if periods else None
            fact_id = f"fact:text_{i}:{occ}"
            _add_financial_fact(
                G=G,
                fact_id=fact_id,
                source_kind="text",
                source_node=text_id,
                evidence_id=text_id,
                quantity_node=qid,
                parsed_number=parsed,
                metric=metric,
                period=period,
                unit=parsed.get("unit") or (units[0] if units else None),
                scale=parsed.get("scale") or scale,
                currency=parsed.get("currency") or currency,
                text=sent_norm,
            )


def _choose_text_metric(metrics: List[str], text: str) -> Optional[str]:
    if metrics:
        # Prefer the longest specific metric phrase.
        return sorted(metrics, key=lambda x: len(x), reverse=True)[0]
    cand = normalize_metric_phrase(text)
    if cand and len(cand.split()) <= 8:
        return cand
    return None


def _attach_table_facts(G: nx.MultiDiGraph, example: Dict[str, Any], inherited_scale: Optional[str]) -> None:
    table = example.get("table", []) or []
    if not table:
        return
    headers = table[0]

    for row_idx, row in enumerate(table):
        row_header = row[0] if row else ""
        row_id = f"table_{row_idx}"
        row_periods = extract_periods(row_header)
        row_scale = detect_scale(" ".join(str(x) for x in row)) or inherited_scale

        for col_idx, value in enumerate(row):
            if row_idx == 0 or col_idx == 0:
                continue
            col_header = headers[col_idx] if col_idx < len(headers) else f"col_{col_idx}"
            cell_id = f"cell:table_{row_idx}:{col_idx}"
            if cell_id not in G:
                continue

            context = f"{row_header} {col_header} {value}"
            nums = get_financial_numbers(value, context=context, inherited_scale=row_scale)
            if not nums:
                continue

            col_periods = extract_periods(col_header)
            cell_periods = extract_periods(context)
            period = _choose_fact_period(row_periods, col_periods, cell_periods)
            metrics = extract_metric_candidates_from_header(row_header, col_header)
            metric = _choose_table_metric(row_header, col_header, metrics)
            scale = detect_scale(context) or row_scale
            currency = detect_currency(context)
            units = detect_units(context)

            # Add header-derived metric and period nodes to the cell.
            if metric:
                mid = add_metric_node(G, metric)
                add_edge_safe(G, cell_id, mid, REL_HAS_METRIC)
            if period:
                pid = add_period_node(G, period)
                add_edge_safe(G, cell_id, pid, REL_HAS_PERIOD)
            for unit in units:
                uid = add_unit_node(G, unit)
                add_edge_safe(G, cell_id, uid, REL_HAS_UNIT)
            if scale:
                sid = add_scale_node(G, scale)
                add_edge_safe(G, cell_id, sid, REL_HAS_SCALE)
            if currency:
                cid = add_currency_node(G, currency)
                add_edge_safe(G, cell_id, cid, REL_HAS_CURRENCY)

            seen_values = set()
            for occ, parsed in enumerate(nums):
                value_key = (parsed.get("normalized_value"), parsed.get("is_percent"))
                if value_key in seen_values:
                    continue
                seen_values.add(value_key)
                qid = add_quantity_node(G, cell_id, parsed, occ)
                fact_id = f"fact:table_{row_idx}:{col_idx}:{occ}"
                _add_financial_fact(
                    G=G,
                    fact_id=fact_id,
                    source_kind="table",
                    source_node=cell_id,
                    evidence_id=row_id,
                    quantity_node=qid,
                    parsed_number=parsed,
                    metric=metric,
                    period=period,
                    unit=parsed.get("unit") or (units[0] if units else None),
                    scale=parsed.get("scale") or scale,
                    currency=parsed.get("currency") or currency,
                    text=normalize_text(f"{row_header} {col_header} {value}"),
                    row_index=row_idx,
                    col_index=col_idx,
                    row_header=normalize_text(row_header),
                    col_header=normalize_text(col_header),
                )


def _choose_fact_period(
    row_periods: List[Dict[str, Any]],
    col_periods: List[Dict[str, Any]],
    cell_periods: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    # If row header is a year, prefer row period. If columns are dates/years, prefer column period.
    if row_periods and col_periods:
        # Column dates are often period columns in financial tables; row years are often data rows.
        # But if row is exactly a year, that is usually stronger for year-indexed tables.
        return row_periods[0]
    if col_periods:
        return col_periods[0]
    if row_periods:
        return row_periods[0]
    return cell_periods[0] if cell_periods else None


def _choose_table_metric(row_header: Any, col_header: Any, metrics: List[str]) -> Optional[str]:
    row_h = normalize_text(row_header)
    col_h = normalize_text(col_header)
    if not metrics:
        return None

    # If row is period-like, metric is usually the column header.
    if is_period_like(row_h):
        col_metric = normalize_metric_phrase(col_h)
        if col_metric:
            return col_metric

    # If column is period-like or generic amount, metric is usually the row header.
    if is_period_like(col_h) or normalize_metric_phrase(col_h) in {"", "amount", "millions", "thousands", "billions"}:
        row_metric = normalize_metric_phrase(row_h)
        if row_metric:
            return row_metric

    # Otherwise prefer the most specific candidate.
    return sorted(metrics, key=lambda x: len(x), reverse=True)[0]


def _add_financial_fact(
    G: nx.MultiDiGraph,
    fact_id: str,
    source_kind: str,
    source_node: str,
    evidence_id: str,
    quantity_node: str,
    parsed_number: Dict[str, Any],
    metric: Optional[str],
    period: Optional[Dict[str, Any]],
    unit: Optional[str],
    scale: Optional[str],
    currency: Optional[str],
    text: str,
    row_index: Optional[int] = None,
    col_index: Optional[int] = None,
    row_header: Optional[str] = None,
    col_header: Optional[str] = None,
) -> None:
    metric_norm = normalize_metric_phrase(metric) if metric else None
    period_value = period.get("value") if period else None
    is_date_number = _is_probably_date_number(parsed_number.get("raw"), period_value, text)
    add_node_safe(
        G,
        fact_id,
        type=NODE_FINANCIAL_FACT,
        source_kind=source_kind,
        source_node=source_node,
        evidence_id=evidence_id,
        quantity_node=quantity_node,
        metric=metric_norm,
        period=period_value,
        period_year=period.get("year") if period else None,
        value=parsed_number.get("value"),
        normalized_value=parsed_number.get("normalized_value"),
        raw_value=parsed_number.get("raw"),
        is_percent=parsed_number.get("is_percent"),
        is_date_number=is_date_number,
        unit=unit,
        scale=scale,
        currency=currency,
        row_index=row_index,
        col_index=col_index,
        row_header=row_header,
        col_header=col_header,
        text=text,
    )

    if source_kind == "table":
        add_edge_safe(G, fact_id, source_node, REL_SOURCE_CELL)
    else:
        add_edge_safe(G, fact_id, source_node, REL_SOURCE_TEXT)
    add_edge_safe(G, fact_id, evidence_id, REL_SOURCE_EVIDENCE)
    add_edge_safe(G, fact_id, quantity_node, REL_HAS_QUANTITY)
    add_edge_safe(G, quantity_node, fact_id, REL_APPEARS_IN)

    if metric_norm:
        mid = add_metric_node(G, metric_norm)
        add_edge_safe(G, fact_id, mid, REL_HAS_METRIC)
        add_edge_safe(G, mid, fact_id, REL_MENTIONED_IN)
    if period:
        pid = add_period_node(G, period)
        add_edge_safe(G, fact_id, pid, REL_HAS_PERIOD)
        add_edge_safe(G, pid, fact_id, REL_MENTIONED_IN)
    if unit:
        uid = add_unit_node(G, unit)
        add_edge_safe(G, fact_id, uid, REL_HAS_UNIT)
    if scale:
        sid = add_scale_node(G, scale)
        add_edge_safe(G, fact_id, sid, REL_HAS_SCALE)
    if currency:
        cid = add_currency_node(G, currency)
        add_edge_safe(G, fact_id, cid, REL_HAS_CURRENCY)


def attach_question_node(G: nx.MultiDiGraph, example: Dict[str, Any]) -> str:
    qa = example.get("qa", {})
    question = qa.get("question", "")
    example_id = example.get("id", "unknown")
    q_id = f"question:{example_id}"
    q_norm = normalize_text(question)

    q_metrics = extract_metric_candidates_from_question(q_norm)
    q_periods = extract_periods(q_norm)
    q_units = detect_units(q_norm)
    q_ops = extract_formula_cues(q_norm)
    if not q_ops:
        q_ops = ["direct_lookup"]

    add_node_safe(
        G,
        q_id,
        type=NODE_QUESTION,
        text=q_norm,
        raw_text=question,
        metrics=q_metrics,
        periods=[p.get("value") for p in q_periods],
        units=q_units,
        operations=q_ops,
    )

    for metric in q_metrics:
        mid = add_metric_node(G, metric)
        add_edge_safe(G, q_id, mid, REL_ASKS_METRIC)
    for period in q_periods:
        pid = add_period_node(G, period)
        add_edge_safe(G, q_id, pid, REL_ASKS_PERIOD)
    for unit in q_units:
        uid = add_unit_node(G, unit)
        add_edge_safe(G, q_id, uid, REL_ASKS_UNIT)
    for op in q_ops:
        op_id = f"operation:{op}"
        add_node_safe(G, op_id, type=NODE_OPERATION, name=op, text=op)
        add_edge_safe(G, q_id, op_id, REL_ASKS_OPERATION)

    return q_id


def link_question_to_facts(G: nx.MultiDiGraph, q_id: str, top_k: int = 30, min_score: float = 0.05) -> None:
    if q_id not in G:
        return
    q_attrs = G.nodes[q_id]
    q_text = q_attrs.get("text", "")
    q_metrics = q_attrs.get("metrics", []) or []
    q_period_values = set(q_attrs.get("periods", []) or [])
    q_units = set(q_attrs.get("units", []) or [])
    q_ops = set(q_attrs.get("operations", []) or [])

    fact_scores: List[Tuple[float, str, Dict[str, float]]] = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") != NODE_FINANCIAL_FACT:
            continue
        score_parts = _score_fact_for_question(attrs, q_text, q_metrics, q_period_values, q_units, q_ops)
        total = score_parts["total_score"]
        if total >= min_score:
            fact_scores.append((total, node_id, score_parts))

    fact_scores.sort(key=lambda x: x[0], reverse=True)
    for rank, (score, fact_id, parts) in enumerate(fact_scores[:top_k], start=1):
        add_edge_safe(G, q_id, fact_id, REL_CANDIDATE_OPERAND, rank=rank, score=score, **parts)


def _score_fact_for_question(
    fact: Dict[str, Any],
    q_text: str,
    q_metrics: List[str],
    q_period_values: set,
    q_units: set,
    q_ops: set,
) -> Dict[str, float]:
    fact_metric = fact.get("metric") or ""
    fact_period = fact.get("period")
    fact_year = fact.get("period_year")
    fact_unit = fact.get("unit")
    fact_scale = fact.get("scale")
    fact_text = fact.get("text", "") or ""
    evidence_id = fact.get("evidence_id", "") or ""
    source_kind = fact.get("source_kind", "") or ""

    metric_score = _best_metric_score(q_text, q_metrics, fact_metric, fact_text)

    period_score = 0.0
    q_period_strs = {str(x) for x in q_period_values}
    if not q_period_values:
        period_score = 0.20
    else:
        if str(fact_period) in q_period_strs or str(fact_year) in q_period_strs:
            period_score = 1.00
        elif fact_year and any(str(fact_year) in str(x) for x in q_period_values):
            period_score = 0.85
        elif fact_metric and metric_score >= 0.60:
            # Some text facts inherit period from neighboring context.
            period_score = 0.30

    unit_score = 0.0
    if not q_units:
        unit_score = 0.12
    else:
        if fact_unit in q_units or fact_scale in q_units:
            unit_score = 1.00
        elif fact.get("currency") and "dollar" in q_units:
            unit_score = 1.00
        elif fact_scale and fact_scale in q_units:
            unit_score = 0.80

    op_score = 0.0
    if not q_ops or "direct_lookup" in q_ops:
        op_score = max(op_score, 0.15)

    if "part_to_whole_ratio" in q_ops or "inverse_percent_total" in q_ops:
        if "total" in fact_metric:
            op_score = max(op_score, 0.75)
        if fact.get("is_percent") or "percent" in str(fact_metric):
            op_score = max(op_score, 0.70)
        if metric_score >= 0.45:
            op_score = max(op_score, 0.45)

    if "growth_rate" in q_ops or "difference" in q_ops:
        if fact_period or fact_year:
            op_score = max(op_score, 0.65)
        if source_kind == "table":
            op_score = max(op_score, 0.45)

    if "comparison" in q_ops:
        # Comparison questions often need one table operand and one text operand.
        op_score = max(op_score, 0.40)
        if source_kind == "table":
            op_score = max(op_score, 0.55)

    if "annual_amortization" in q_ops:
        if "amortization" in fact_metric or "intangible" in fact_metric or "year" in str(fact_unit):
            op_score = max(op_score, 0.85)

    if "sum_or_total" in q_ops:
        if "total" in fact_metric:
            op_score = max(op_score, 0.70)
        elif metric_score >= 0.35:
            op_score = max(op_score, 0.45)

    # Base score: metric and period dominate. Operation is only a secondary cue.
    total = 0.62 * metric_score + 0.20 * period_score + 0.08 * unit_score + 0.10 * op_score

    # Strong penalty: date components are not operands.
    if fact.get("is_date_number") or _is_probably_date_number(fact.get("raw_value"), fact_period, fact_text):
        total -= 0.35

    # Generic-only metrics should not outrank specific financial facts.
    fact_info = _informative_tokens(fact_metric)
    q_info = _informative_tokens(q_text)
    if fact_metric and not fact_info and (_metric_tokens(fact_metric) & _metric_tokens(q_text)):
        total -= 0.12

    # If metric is extremely generic and value is a year/day, suppress harder.
    if fact.get("is_date_number") and metric_score < 0.80:
        total -= 0.10

    # Mild preference for table facts in numeric operation questions.
    if source_kind == "table" and q_ops and "direct_lookup" not in q_ops:
        total += 0.03

    total = max(0.0, min(1.0, total))

    return {
        "metric_score": round(metric_score, 4),
        "period_score": round(period_score, 4),
        "unit_score": round(unit_score, 4),
        "operation_score": round(op_score, 4),
        "total_score": round(total, 4),
    }

def add_temporal_edges(G: nx.MultiDiGraph) -> None:
    periods = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == NODE_PERIOD:
            year = attrs.get("year")
            value = attrs.get("value")
            if year is not None:
                periods.append((int(year), str(value), node_id))
    periods = sorted(set(periods))
    for i in range(len(periods) - 1):
        _, _, src = periods[i]
        _, _, dst = periods[i + 1]
        add_edge_safe(G, src, dst, REL_PREVIOUS_PERIOD_OF)
        add_edge_safe(G, dst, src, REL_LATER_OR_EQUAL_PERIOD)


def add_contextual_fact_edges(G: nx.MultiDiGraph) -> None:
    facts = [(n, a) for n, a in G.nodes(data=True) if a.get("type") == NODE_FINANCIAL_FACT]

    for i, (id_i, a_i) in enumerate(facts):
        for id_j, a_j in facts[i + 1:]:
            same_metric = a_i.get("metric") and a_i.get("metric") == a_j.get("metric")
            same_period = a_i.get("period") and a_i.get("period") == a_j.get("period")
            same_row = a_i.get("row_index") is not None and a_i.get("row_index") == a_j.get("row_index")
            same_col = a_i.get("col_index") is not None and a_i.get("col_index") == a_j.get("col_index")

            if same_metric:
                add_edge_safe(G, id_i, id_j, REL_SAME_METRIC_AS)
                add_edge_safe(G, id_j, id_i, REL_SAME_METRIC_AS)
            if same_period:
                add_edge_safe(G, id_i, id_j, REL_SAME_PERIOD_AS)
                add_edge_safe(G, id_j, id_i, REL_SAME_PERIOD_AS)
            if same_row:
                add_edge_safe(G, id_i, id_j, REL_SAME_ROW_FACT)
                add_edge_safe(G, id_j, id_i, REL_SAME_ROW_FACT)
            if same_col:
                add_edge_safe(G, id_i, id_j, REL_SAME_COLUMN_FACT)
                add_edge_safe(G, id_j, id_i, REL_SAME_COLUMN_FACT)

            if _looks_part_of_total(a_i, a_j):
                add_edge_safe(G, id_i, id_j, REL_PART_OF_TOTAL)
                add_edge_safe(G, id_j, id_i, REL_PART_OF_TOTAL)

            if same_metric and _period_precedes(a_i, a_j):
                add_edge_safe(G, id_i, id_j, REL_PREVIOUS_PERIOD_OF)
            elif same_metric and _period_precedes(a_j, a_i):
                add_edge_safe(G, id_j, id_i, REL_PREVIOUS_PERIOD_OF)


def _looks_part_of_total(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if not (a.get("period") and a.get("period") == b.get("period")):
        return False
    text = f"{a.get('metric', '')} {b.get('metric', '')}"
    return "percent" in text and "total" in text


def _period_precedes(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    ya = a.get("period_year")
    yb = b.get("period_year")
    if ya is None or yb is None:
        return False
    return int(ya) < int(yb)


def add_contextual_quantity_comparison_edges(G: nx.MultiDiGraph) -> None:
    """
    Optional local comparison edges. This deliberately avoids global all-pairs quantity comparison.
    Only comparable facts with same metric or same period+unit are connected.
    """
    facts = [(n, a) for n, a in G.nodes(data=True) if a.get("type") == NODE_FINANCIAL_FACT]
    for i, (id_i, a_i) in enumerate(facts):
        for id_j, a_j in facts[i + 1:]:
            comparable = (
                (a_i.get("metric") and a_i.get("metric") == a_j.get("metric"))
                or (
                    a_i.get("period") and a_i.get("period") == a_j.get("period")
                    and a_i.get("unit") == a_j.get("unit")
                    and a_i.get("scale") == a_j.get("scale")
                )
            )
            if not comparable:
                continue
            vi = a_i.get("normalized_value")
            vj = a_j.get("normalized_value")
            if vi is None or vj is None:
                continue
            if vi >= vj:
                add_edge_safe(G, id_i, id_j, REL_GREATER_OR_EQUAL_QUANTITY)
            else:
                add_edge_safe(G, id_j, id_i, REL_GREATER_OR_EQUAL_QUANTITY)



# ---------------------------------------------------------------------------
# Table-text bridge and source-expansion support
# ---------------------------------------------------------------------------

REL_SUPPORTED_BY_TEXT_LOCAL = "supported_by_text"
REL_DESCRIBES_TABLE_LOCAL = "describes_table"
REL_TABLE_TEXT_BRIDGE_LOCAL = "table_text_bridge"


def _source_kind(source_id: str) -> str:
    if source_id.startswith("table_"):
        return "table"
    if source_id.startswith("text_"):
        return "text"
    return "unknown"


def _source_index(source_id: str) -> Optional[int]:
    try:
        return int(source_id.split("_", 1)[1])
    except Exception:
        return None


def _source_fact_attrs(G: nx.MultiDiGraph, evidence_id: str) -> List[Dict[str, Any]]:
    return [
        attrs
        for _, attrs in G.nodes(data=True)
        if attrs.get("type") == NODE_FINANCIAL_FACT and attrs.get("evidence_id") == evidence_id
    ]


def _source_metrics_periods_text(G: nx.MultiDiGraph, evidence_id: str) -> Tuple[set, set, str]:
    facts = _source_fact_attrs(G, evidence_id)
    metrics = {normalize_text(f.get("metric") or "") for f in facts if f.get("metric")}
    periods = {str(f.get("period")) for f in facts if f.get("period")}
    text = normalize_text(G.nodes[evidence_id].get("text", "")) if evidence_id in G.nodes else ""
    if not text:
        text = " ".join(normalize_text(f.get("text", "")) for f in facts)
    return metrics, periods, text


def _bridge_score_between_sources(
    G: nx.MultiDiGraph,
    table_source: str,
    text_source: str,
    pre_text_len: int,
) -> float:
    table_metrics, table_periods, table_text = _source_metrics_periods_text(G, table_source)
    text_metrics, text_periods, text_text = _source_metrics_periods_text(G, text_source)

    score = 0.0

    # Layout prior: FinQA pages put the table between pre_text and post_text.
    ti = _source_index(text_source)
    if ti is not None:
        dist_to_boundary = min(abs(ti - (pre_text_len - 1)), abs(ti - pre_text_len))
        if dist_to_boundary <= 1:
            score += 0.30
        elif dist_to_boundary <= 3:
            score += 0.20
        elif dist_to_boundary <= 6:
            score += 0.10

    # Metric overlap. Use informative-token overlap so generic words do not dominate.
    metric_score = 0.0
    for tm in table_metrics:
        for xm in text_metrics:
            metric_score = max(metric_score, _metric_match_score(tm, xm))
    if metric_score == 0.0 and table_text and text_text:
        t_info = _informative_tokens(table_text)
        x_info = _informative_tokens(text_text)
        if t_info and x_info:
            metric_score = len(t_info & x_info) / max(min(len(t_info), len(x_info)), 1)
    score += min(0.35, 0.35 * metric_score)

    # Period overlap.
    if table_periods and text_periods and (table_periods & text_periods):
        score += 0.20

    # Text/caption cue: sentences saying "following table", "as follows", etc.
    caption_cues = [
        "following table", "table shows", "table presents", "as follows", "following is",
        "summarizes", "set forth", "shown below", "in the table",
    ]
    if any(cue in text_text for cue in caption_cues):
        score += 0.20

    # Do not over-connect totally empty sources.
    return round(min(score, 1.0), 4)


def add_table_text_bridge_edges(
    G: nx.MultiDiGraph,
    example: Dict[str, Any],
    min_score: float = 0.18,
    max_texts_per_row: int = 5,
) -> None:
    """
    Add source-level bridge edges between table rows and nearby/related text.

    These edges are intentionally source-level, not fact-level:
      table_i --supported_by_text--> text_j
      text_j  --describes_table--> table_i

    They are used later by evidence_set_ranker and by source expansion to recover
    mixed table-text evidence sets.
    """
    pre_text_len = len(example.get("pre_text", []) or [])
    table_sources = [n for n, a in G.nodes(data=True) if a.get("type") == NODE_TABLE_ROW]
    text_sources = [n for n, a in G.nodes(data=True) if a.get("type") == NODE_TEXT]

    for table_src in table_sources:
        if table_src == "table_0":
            continue
        candidates = []
        for text_src in text_sources:
            score = _bridge_score_between_sources(G, table_src, text_src, pre_text_len)
            if score >= min_score:
                candidates.append((score, text_src))
        candidates.sort(key=lambda x: (-x[0], _source_index(x[1]) if _source_index(x[1]) is not None else 10**9))
        for score, text_src in candidates[:max_texts_per_row]:
            add_edge_safe(G, table_src, text_src, REL_SUPPORTED_BY_TEXT_LOCAL, score=score)
            add_edge_safe(G, text_src, table_src, REL_DESCRIBES_TABLE_LOCAL, score=score)
            add_edge_safe(G, table_src, text_src, REL_TABLE_TEXT_BRIDGE_LOCAL, score=score)
            add_edge_safe(G, text_src, table_src, REL_TABLE_TEXT_BRIDGE_LOCAL, score=score)


def _fact_ids_by_evidence(G: nx.MultiDiGraph, evidence_id: str) -> List[str]:
    out = []
    for node_id, attrs in G.nodes(data=True):
        if attrs.get("type") == NODE_FINANCIAL_FACT and attrs.get("evidence_id") == evidence_id:
            out.append(node_id)
    return out


def _existing_candidate_fact_ids(G: nx.MultiDiGraph, q_id: str) -> set:
    out = set()
    if q_id not in G.nodes:
        return out
    for _, dst, attrs in G.out_edges(q_id, data=True):
        if attrs.get("rel") == REL_CANDIDATE_OPERAND:
            out.add(dst)
    return out


def _current_candidate_source_scores(G: nx.MultiDiGraph, q_id: str) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if q_id not in G.nodes:
        return scores
    for _, fact_id, attrs in G.out_edges(q_id, data=True):
        if attrs.get("rel") != REL_CANDIDATE_OPERAND or fact_id not in G.nodes:
            continue
        source = G.nodes[fact_id].get("evidence_id")
        if not source:
            continue
        score = float(attrs.get("score", attrs.get("total_score", 0.0)) or 0.0)
        scores[source] = max(scores.get(source, 0.0), score)
    return scores


def expand_question_candidates_from_bridges(
    G: nx.MultiDiGraph,
    q_id: str,
    top_k: int = 30,
    max_extra_sources: int = 8,
    min_bridge_score: float = 0.18,
) -> None:
    """
    Add low-priority candidate_operand edges for facts located in bridge-connected
    table/text sources. This is not a replacement for source-set ranking; it only
    prevents mixed evidence candidates from being invisible to downstream rankers.
    """
    if q_id not in G.nodes:
        return
    source_scores = _current_candidate_source_scores(G, q_id)
    if not source_scores:
        return

    existing = _existing_candidate_fact_ids(G, q_id)
    proposed_sources: Dict[str, Tuple[float, str]] = {}

    seed_sources = sorted(source_scores.items(), key=lambda x: -x[1])[:max(10, max_extra_sources)]
    for source, src_score in seed_sources:
        if source not in G.nodes:
            continue
        for _, dst, attrs in G.out_edges(source, data=True):
            if attrs.get("rel") not in {REL_SUPPORTED_BY_TEXT_LOCAL, REL_DESCRIBES_TABLE_LOCAL, REL_TABLE_TEXT_BRIDGE_LOCAL}:
                continue
            if not isinstance(dst, str) or not (dst.startswith("text_") or dst.startswith("table_")):
                continue
            bridge_score = float(attrs.get("score", 0.0) or 0.0)
            if bridge_score < min_bridge_score:
                continue
            proposed = 0.55 * src_score + 0.20 * bridge_score
            if dst not in proposed_sources or proposed > proposed_sources[dst][0]:
                proposed_sources[dst] = (proposed, source)

    rank_offset = top_k + 1
    added = 0
    for evidence_id, (score, expanded_from) in sorted(proposed_sources.items(), key=lambda x: -x[1][0]):
        if added >= max_extra_sources:
            break
        for fact_id in _fact_ids_by_evidence(G, evidence_id)[:3]:
            if fact_id in existing:
                continue
            add_edge_safe(
                G,
                q_id,
                fact_id,
                REL_CANDIDATE_OPERAND,
                rank=rank_offset + added,
                score=round(max(0.01, min(score, 1.0)), 4),
                total_score=round(max(0.01, min(score, 1.0)), 4),
                expanded=True,
                expanded_from=expanded_from,
                bridge_source=evidence_id,
                metric_score=0.0,
                period_score=0.0,
                unit_score=0.0,
                operation_score=0.0,
            )
            existing.add(fact_id)
        added += 1

def build_page_graph(example: Dict[str, Any]) -> nx.MultiDiGraph:
    """Backward-compatible alias: source graph only."""
    return build_source_graph(example)


def build_inference_graph(
    example: Dict[str, Any],
    top_k_candidate_facts: int = 30,
    include_contextual_comparison: bool = False,
) -> tuple[nx.MultiDiGraph, str]:
    """
    Full question-conditioned financial fact graph.

    Does not use gold evidence, gold program, or answer.
    """
    G = build_source_graph(example)
    build_fact_graph(G, example)
    q_id = attach_question_node(G, example)
    link_question_to_facts(G, q_id, top_k=top_k_candidate_facts)
    expand_question_candidates_from_bridges(G, q_id, top_k=top_k_candidate_facts)
    if include_contextual_comparison:
        add_contextual_quantity_comparison_edges(G)
    return G, q_id
