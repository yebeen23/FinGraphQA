"""
Operation-aware evidence-source ranking for FinGraphQA.

This module sits after graph construction. It converts question->candidate_operand
fact edges into a compact, source-level evidence ranking and applies conservative
operation-aware source-set expansion. It does not use gold labels, programs, or
answers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import re

import networkx as nx

try:
    from schema import NODE_FINANCIAL_FACT, REL_CANDIDATE_OPERAND
except Exception:  # pragma: no cover
    NODE_FINANCIAL_FACT = "financial_fact"
    REL_CANDIDATE_OPERAND = "candidate_operand"

try:
    from preprocess import extract_formula_cues, normalize_text, token_set
except Exception:  # pragma: no cover
    def normalize_text(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).lower()).strip()

    def token_set(s: str) -> Set[str]:
        return set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", normalize_text(s)))

    def extract_formula_cues(question: str) -> List[str]:
        q = normalize_text(question)
        cues = []
        if any(x in q for x in ["growth rate", "percentage change", "percent change"]):
            cues.append("growth_rate")
        if any(x in q for x in ["what percentage", "what percent", "comprised of", "portion of", "as part of"]):
            cues.append("part_to_whole_ratio")
        if any(x in q for x in ["average", "per "]):
            cues.append("average_or_ratio")
        if any(x in q for x in ["change", "increase", "decrease", "difference"]):
            cues.append("difference")
        if any(x in q for x in ["exceed", "greater", "larger", "did "]):
            cues.append("comparison")
        if any(x in q for x in ["total", "sum"]):
            cues.append("sum_or_total")
        return sorted(set(cues))


BRIDGE_RELS = {"supported_by_text", "describes_table", "table_text_bridge"}

GENERIC_TERMS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "was", "were",
    "are", "how", "much", "many", "percent", "percentage", "change", "rate",
    "total", "amount", "value", "fair", "income", "expense", "expenses",
    "asset", "assets", "cash", "net", "basis", "year", "years", "million",
    "millions", "thousand", "thousands", "billion", "billions",
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _source_sort_key(src: str) -> Tuple[int, int, str]:
    """Stable sort: table_1 before table_10; text_2 before text_11."""
    if isinstance(src, str) and "_" in src:
        prefix, suffix = src.split("_", 1)
        if suffix.isdigit():
            order = 0 if prefix == "table" else 1 if prefix == "text" else 2
            return (order, int(suffix), src)
    return (9, 10**9, str(src))


def source_kind(source: Optional[str]) -> str:
    if not source:
        return "unknown"
    if source.startswith("table_"):
        return "table"
    if source.startswith("text_"):
        return "text"
    return "unknown"


def get_fact_source(G: nx.MultiDiGraph, fact_id: str) -> Optional[str]:
    """Return FinQA-compatible source id: text_i or table_i."""
    if fact_id not in G.nodes:
        return None

    attrs = G.nodes[fact_id]

    for key in ["evidence_id", "source_evidence", "source"]:
        value = attrs.get(key)
        if isinstance(value, str) and (value.startswith("text_") or value.startswith("table_")):
            return value

    for _, dst, edge_attrs in G.out_edges(fact_id, data=True):
        if edge_attrs.get("rel") == "source_evidence":
            if isinstance(dst, str) and (dst.startswith("text_") or dst.startswith("table_")):
                return dst

    source_node = attrs.get("source_node")
    if isinstance(source_node, str):
        if source_node.startswith("text_"):
            return source_node
        if source_node.startswith("table_"):
            return source_node
        if source_node.startswith("cell:table_"):
            parts = source_node.split(":")
            if len(parts) >= 2 and parts[1].startswith("table_"):
                return parts[1]

    if isinstance(fact_id, str) and fact_id.startswith("fact:"):
        parts = fact_id.split(":")
        if len(parts) >= 2 and (parts[1].startswith("text_") or parts[1].startswith("table_")):
            return parts[1]

    return None


def candidate_fact_rows(G: nx.MultiDiGraph, q_id: str) -> List[Dict[str, Any]]:
    """Question->candidate_operand fact rows, already scored by graph_builder."""
    rows: List[Dict[str, Any]] = []
    if q_id not in G.nodes:
        return rows

    for _, fact_id, edge_attrs in G.out_edges(q_id, data=True):
        if edge_attrs.get("rel") != REL_CANDIDATE_OPERAND:
            continue
        if fact_id not in G.nodes:
            continue
        fact_attrs = G.nodes[fact_id]
        if fact_attrs.get("type") != NODE_FINANCIAL_FACT:
            continue

        source = get_fact_source(G, fact_id)
        if source is None:
            continue

        rank = edge_attrs.get("rank", 999999)
        try:
            rank = int(rank)
        except Exception:
            rank = 999999

        score = _safe_float(edge_attrs.get("score", edge_attrs.get("total_score", 0.0)))
        rows.append({
            "rank": rank,
            "fact_id": fact_id,
            "source": source,
            "score": score,
            "metric": fact_attrs.get("metric"),
            "period": fact_attrs.get("period"),
            "period_year": fact_attrs.get("period_year"),
            "value": fact_attrs.get("raw_value", fact_attrs.get("value")),
            "normalized_value": fact_attrs.get("normalized_value"),
            "is_percent": fact_attrs.get("is_percent"),
            "is_date_number": fact_attrs.get("is_date_number"),
            "source_kind": fact_attrs.get("source_kind"),
            "text": fact_attrs.get("text", ""),
        })

    rows.sort(key=lambda r: (-r["score"], r["rank"], r["fact_id"]))
    return rows


def source_details_from_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate fact-level rows into source-level details."""
    details: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        source = row["source"]
        score = _safe_float(row.get("score"))
        if source not in details:
            details[source] = {
                "source": source,
                "kind": source_kind(source),
                "best_score": score,
                "base_score": score,
                "best_fact": row,
                "fact_count": 1,
                "metrics": set([row.get("metric")]) if row.get("metric") else set(),
                "periods": set([row.get("period")]) if row.get("period") else set(),
            }
        else:
            details[source]["fact_count"] += 1
            if row.get("metric"):
                details[source]["metrics"].add(row.get("metric"))
            if row.get("period"):
                details[source]["periods"].add(row.get("period"))
            if score > details[source]["best_score"]:
                details[source]["best_score"] = score
                details[source]["base_score"] = score
                details[source]["best_fact"] = row

    for d in details.values():
    # Periods may mix int years, e.g., 2012, and string dates, e.g., "2012-03-31".
    # Python 3 cannot sort int and str directly.
        d["metrics"] = sorted(str(x) for x in d["metrics"] if x is not None)
        d["periods"] = sorted(str(x) for x in d["periods"] if x is not None)
    return details


def get_bridge_neighbors(G: nx.MultiDiGraph, source: str, target_kind: Optional[str] = None) -> List[Tuple[str, float, str]]:
    """Return bridge-connected source nodes with bridge score and relation."""
    if source not in G.nodes:
        return []
    out = []
    for _, dst, attrs in G.out_edges(source, data=True):
        rel = attrs.get("rel")
        if rel not in BRIDGE_RELS:
            continue
        if not isinstance(dst, str) or not (dst.startswith("text_") or dst.startswith("table_")):
            continue
        if target_kind and source_kind(dst) != target_kind:
            continue
        out.append((dst, _safe_float(attrs.get("score", attrs.get("weight", 0.0))), rel))
    for src, _, attrs in G.in_edges(source, data=True):
        rel = attrs.get("rel")
        if rel not in BRIDGE_RELS:
            continue
        if not isinstance(src, str) or not (src.startswith("text_") or src.startswith("table_")):
            continue
        if target_kind and source_kind(src) != target_kind:
            continue
        out.append((src, _safe_float(attrs.get("score", attrs.get("weight", 0.0))), rel))
    # Deduplicate by best score.
    best = {}
    for n, s, r in out:
        if n not in best or s > best[n][0]:
            best[n] = (s, r)
    return sorted([(n, s, r) for n, (s, r) in best.items()], key=lambda x: (-x[1], _source_sort_key(x[0])))


def _question_tokens(question: str) -> Set[str]:
    return token_set(question) - GENERIC_TERMS


def _source_text(G: nx.MultiDiGraph, source: str) -> str:
    if source not in G.nodes:
        return ""
    attrs = G.nodes[source]
    return normalize_text(attrs.get("text") or attrs.get("raw_text") or "")


def _textual_relevance(G: nx.MultiDiGraph, source: str, question: str) -> float:
    qt = _question_tokens(question)
    st = token_set(_source_text(G, source)) - GENERIC_TERMS
    if not qt or not st:
        return 0.0
    return len(qt & st) / max(len(qt), 1)


def _needs_mixed_expansion(question: str, cues: Set[str]) -> bool:
    q = normalize_text(question)
    if cues & {"growth_rate", "difference", "part_to_whole_ratio", "comparison", "average_or_ratio", "sum_or_total"}:
        return True
    return any(x in q for x in ["what percent", "what percentage", "portion", "as part of", "compared", "from", "between"])


def _operation_source_bonus(source: str, detail: Dict[str, Any], question: str, cues: Set[str]) -> float:
    bonus = 0.0
    kind = detail.get("kind") or source_kind(source)
    best_fact = detail.get("best_fact", {}) or {}
    metric = normalize_text(best_fact.get("metric") or "")
    value = str(best_fact.get("value") or "")

    if cues & {"growth_rate", "difference", "part_to_whole_ratio", "average_or_ratio", "comparison", "sum_or_total"}:
        if kind == "table":
            bonus += 0.03

    if cues & {"part_to_whole_ratio"}:
        if "total" in metric:
            bonus += 0.08
        if "%" in value or best_fact.get("is_percent"):
            bonus += 0.04

    if cues & {"comparison"} and kind == "table":
        bonus += 0.04

    # Do not let date-only text facts dominate source sets.
    if best_fact.get("is_date_number"):
        bonus -= 0.08

    return bonus


def rank_evidence_sources(
    G: nx.MultiDiGraph,
    q_id: str,
    question: str,
    top_k: int = 20,
    max_set_size: int = 5,
    bridge_bonus: float = 0.18,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Produce an operation-aware source ranking.

    Steps:
    1. Aggregate candidate facts into unique source scores.
    2. Add conservative operation bonuses.
    3. Expand across table-text bridge edges when a mixed evidence set is likely.
    4. Return a compact ranking suitable for Recall@k evaluation.
    """
    rows = candidate_fact_rows(G, q_id)
    details = source_details_from_rows(rows)
    cues = set(extract_formula_cues(question) or ["direct_lookup"])

    for source, detail in details.items():
        detail["score"] = detail["best_score"] + _operation_source_bonus(source, detail, question, cues)
        detail["expanded"] = False
        detail["expanded_from"] = None
        detail["bridge_rel"] = None

    # Add bridge-neighbor sources even when they had no candidate facts.
    if _needs_mixed_expansion(question, cues):
        seeds = sorted(details.values(), key=lambda d: -d["score"])[:max(10, max_set_size * 2)]
        for d in seeds:
            source = d["source"]
            want = "text" if source_kind(source) == "table" else "table" if source_kind(source) == "text" else None
            if not want:
                continue
            for neigh, bscore, rel in get_bridge_neighbors(G, source, target_kind=want)[:3]:
                rel_score = _textual_relevance(G, neigh, question)
                proposed = d["score"] * 0.72 + bridge_bonus * bscore + 0.10 * rel_score
                if neigh not in details:
                    details[neigh] = {
                        "source": neigh,
                        "kind": source_kind(neigh),
                        "best_score": proposed,
                        "base_score": 0.0,
                        "best_fact": {"fact_id": None, "source": neigh, "score": 0.0, "metric": None, "period": None, "value": None},
                        "fact_count": 0,
                        "metrics": [],
                        "periods": [],
                        "score": proposed,
                        "expanded": True,
                        "expanded_from": source,
                        "bridge_rel": rel,
                    }
                elif proposed > details[neigh].get("score", 0.0):
                    details[neigh]["score"] = proposed
                    details[neigh]["expanded"] = True
                    details[neigh]["expanded_from"] = source
                    details[neigh]["bridge_rel"] = rel

    ranked = sorted(
        details.values(),
        key=lambda d: (-d.get("score", 0.0), -min(d.get("fact_count", 0), 3), _source_sort_key(d["source"])),
    )

    # Set construction: interleave table/text when mixed reasoning is likely.
    if _needs_mixed_expansion(question, cues):
        selected: List[Dict[str, Any]] = []
        seen = set()
        tables = [d for d in ranked if d["kind"] == "table"]
        texts = [d for d in ranked if d["kind"] == "text"]

        # Start with the best source, then force the best opposite-modality bridge if available.
        for d in ranked[:2]:
            if d["source"] not in seen:
                selected.append(d); seen.add(d["source"])

        has_table = any(d["kind"] == "table" for d in selected)
        has_text = any(d["kind"] == "text" for d in selected)
        if has_table and not has_text and texts:
            selected.append(texts[0]); seen.add(texts[0]["source"])
        if has_text and not has_table and tables:
            selected.append(tables[0]); seen.add(tables[0]["source"])

        for d in ranked:
            if len(selected) >= top_k:
                break
            if d["source"] not in seen:
                selected.append(d); seen.add(d["source"])
        ranked = selected

    ranked = ranked[:top_k]
    return [d["source"] for d in ranked], ranked


def relaxed_source_covered(G: nx.MultiDiGraph, gold_source: str, selected_sources: Set[str]) -> bool:
    """
    Relaxed source coverage for contextual evidence.

    Strict source recall is still the main metric. Relaxed recall gives credit when:
    - exact gold source selected,
    - adjacent text sentence selected for a text gold,
    - a selected source is bridge-connected to the gold source.
    """
    if gold_source in selected_sources:
        return True

    if gold_source.startswith("text_"):
        try:
            idx = int(gold_source.split("_", 1)[1])
            if f"text_{idx - 1}" in selected_sources or f"text_{idx + 1}" in selected_sources:
                return True
        except Exception:
            pass

    if gold_source in G.nodes:
        for sel in selected_sources:
            if sel not in G.nodes:
                continue
            # direct bridge either direction
            for _, dst, attrs in G.out_edges(sel, data=True):
                if dst == gold_source and attrs.get("rel") in BRIDGE_RELS:
                    return True
            for src, _, attrs in G.in_edges(sel, data=True):
                if src == gold_source and attrs.get("rel") in BRIDGE_RELS:
                    return True
            for _, dst, attrs in G.out_edges(gold_source, data=True):
                if dst == sel and attrs.get("rel") in BRIDGE_RELS:
                    return True
            for src, _, attrs in G.in_edges(gold_source, data=True):
                if src == sel and attrs.get("rel") in BRIDGE_RELS:
                    return True

    return False


def relaxed_all_covered(G: nx.MultiDiGraph, gold_sources: Set[str], selected_sources: Set[str]) -> bool:
    return all(relaxed_source_covered(G, g, selected_sources) for g in gold_sources)


def relaxed_any_covered(G: nx.MultiDiGraph, gold_sources: Set[str], selected_sources: Set[str]) -> bool:
    return any(relaxed_source_covered(G, g, selected_sources) for g in gold_sources)
