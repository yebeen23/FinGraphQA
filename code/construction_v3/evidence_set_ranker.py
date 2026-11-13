# """
# Operation-aware evidence-source ranker for FinGraphQA.

# This module converts graph_builder's question->candidate_operand fact edges into
# source-level evidence rankings, then performs controlled table-text expansion.
# It does not use gold labels, programs, or answers.
# """
# from __future__ import annotations

# from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
# import re

# import networkx as nx

# try:
#     from schema import NODE_FINANCIAL_FACT, REL_CANDIDATE_OPERAND, BRIDGE_RELS
# except Exception:  # pragma: no cover
#     NODE_FINANCIAL_FACT = "financial_fact"
#     REL_CANDIDATE_OPERAND = "candidate_operand"
#     BRIDGE_RELS = {"supported_by_text", "describes_table", "table_text_bridge"}

# try:
#     from preprocess import extract_formula_cues, normalize_text, token_set
# except Exception:  # pragma: no cover
#     def normalize_text(s: str) -> str:
#         return re.sub(r"\s+", " ", str(s).lower()).strip()
#     def token_set(s: str) -> Set[str]:
#         return set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", normalize_text(s)))
#     def extract_formula_cues(question: str) -> List[str]:
#         return []

# GENERIC_TERMS = {
#     "the", "and", "for", "with", "from", "that", "this", "what", "was", "were",
#     "are", "how", "much", "many", "percent", "percentage", "change", "rate",
#     "total", "amount", "value", "fair", "income", "expense", "expenses",
#     "asset", "assets", "cash", "net", "basis", "year", "years", "million",
#     "millions", "thousand", "thousands", "billion", "billions",
# }

# MIXED_EXPANSION_CUES = {
#     "growth_rate", "difference", "part_to_whole_ratio", "inverse_percent_total",
#     "comparison", "average_or_ratio", "sum_or_total", "annual_amortization",
# }


# def _safe_float(x: Any, default: float = 0.0) -> float:
#     try:
#         return float(x)
#     except Exception:
#         return default


# def _source_sort_key(src: str) -> Tuple[int, int, str]:
#     if isinstance(src, str) and "_" in src:
#         prefix, suffix = src.split("_", 1)
#         if suffix.isdigit():
#             order = 0 if prefix == "table" else 1 if prefix == "text" else 2
#             return (order, int(suffix), src)
#     return (9, 10**9, str(src))


# def source_kind(source: Optional[str]) -> str:
#     if not source:
#         return "unknown"
#     if source.startswith("table_"):
#         return "table"
#     if source.startswith("text_"):
#         return "text"
#     return "unknown"


# def get_fact_source(G: nx.MultiDiGraph, fact_id: str) -> Optional[str]:
#     if fact_id not in G.nodes:
#         return None
#     attrs = G.nodes[fact_id]
#     for key in ["evidence_id", "source_evidence", "source"]:
#         value = attrs.get(key)
#         if isinstance(value, str) and (value.startswith("text_") or value.startswith("table_")):
#             return value
#     for _, dst, edge_attrs in G.out_edges(fact_id, data=True):
#         if edge_attrs.get("rel") == "source_evidence":
#             if isinstance(dst, str) and (dst.startswith("text_") or dst.startswith("table_")):
#                 return dst
#     source_node = attrs.get("source_node")
#     if isinstance(source_node, str):
#         if source_node.startswith("text_") or source_node.startswith("table_"):
#             return source_node
#         if source_node.startswith("cell:table_"):
#             parts = source_node.split(":")
#             if len(parts) >= 2 and parts[1].startswith("table_"):
#                 return parts[1]
#     if isinstance(fact_id, str) and fact_id.startswith("fact:"):
#         parts = fact_id.split(":")
#         if len(parts) >= 2 and (parts[1].startswith("text_") or parts[1].startswith("table_")):
#             return parts[1]
#     return None


# def candidate_fact_rows(G: nx.MultiDiGraph, q_id: str) -> List[Dict[str, Any]]:
#     rows: List[Dict[str, Any]] = []
#     if q_id not in G.nodes:
#         return rows
#     for _, fact_id, edge_attrs in G.out_edges(q_id, data=True):
#         if edge_attrs.get("rel") != REL_CANDIDATE_OPERAND:
#             continue
#         if fact_id not in G.nodes:
#             continue
#         fact_attrs = G.nodes[fact_id]
#         if fact_attrs.get("type") != NODE_FINANCIAL_FACT:
#             continue
#         source = get_fact_source(G, fact_id)
#         if source is None:
#             continue
#         rank = edge_attrs.get("rank", 999999)
#         try:
#             rank = int(rank)
#         except Exception:
#             rank = 999999
#         score = _safe_float(edge_attrs.get("score", edge_attrs.get("total_score", 0.0)))
#         rows.append({
#             "rank": rank,
#             "fact_id": fact_id,
#             "source": source,
#             "score": score,
#             "metric": fact_attrs.get("metric"),
#             "period": fact_attrs.get("period"),
#             "period_year": fact_attrs.get("period_year"),
#             "value": fact_attrs.get("raw_value", fact_attrs.get("value")),
#             "normalized_value": fact_attrs.get("normalized_value"),
#             "is_percent": fact_attrs.get("is_percent"),
#             "is_date_number": fact_attrs.get("is_date_number"),
#             "source_kind": fact_attrs.get("source_kind"),
#             "text": fact_attrs.get("text", ""),
#         })
#     rows.sort(key=lambda r: (-r["score"], r["rank"], r["fact_id"]))
#     return rows


# def source_details_from_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
#     details: Dict[str, Dict[str, Any]] = {}
#     for row in rows:
#         source = row["source"]
#         score = _safe_float(row.get("score"))
#         if source not in details:
#             details[source] = {
#                 "source": source,
#                 "kind": source_kind(source),
#                 "best_score": score,
#                 "base_score": score,
#                 "score": score,
#                 "best_fact": row,
#                 "fact_count": 1,
#                 "metrics": set([row.get("metric")]) if row.get("metric") else set(),
#                 "periods": set([row.get("period")]) if row.get("period") else set(),
#                 "expanded": False,
#                 "expanded_from": None,
#                 "bridge_rel": None,
#                 "bridge_score": 0.0,
#             }
#         else:
#             details[source]["fact_count"] += 1
#             if row.get("metric"):
#                 details[source]["metrics"].add(row.get("metric"))
#             if row.get("period"):
#                 details[source]["periods"].add(row.get("period"))
#             if score > details[source]["best_score"]:
#                 details[source]["best_score"] = score
#                 details[source]["base_score"] = score
#                 details[source]["score"] = score
#                 details[source]["best_fact"] = row
#     for d in details.values():
#         d["metrics"] = sorted(str(x) for x in d["metrics"] if x is not None)
#         d["periods"] = sorted(str(x) for x in d["periods"] if x is not None)
#     return details


# def get_bridge_neighbors(G: nx.MultiDiGraph, source: str, target_kind: Optional[str] = None) -> List[Tuple[str, float, str]]:
#     if source not in G.nodes:
#         return []
#     out = []
#     for _, dst, attrs in G.out_edges(source, data=True):
#         rel = attrs.get("rel")
#         if rel not in BRIDGE_RELS:
#             continue
#         if not isinstance(dst, str) or not (dst.startswith("text_") or dst.startswith("table_")):
#             continue
#         if target_kind and source_kind(dst) != target_kind:
#             continue
#         out.append((dst, _safe_float(attrs.get("score", attrs.get("weight", 0.0))), rel))
#     for src, _, attrs in G.in_edges(source, data=True):
#         rel = attrs.get("rel")
#         if rel not in BRIDGE_RELS:
#             continue
#         if not isinstance(src, str) or not (src.startswith("text_") or src.startswith("table_")):
#             continue
#         if target_kind and source_kind(src) != target_kind:
#             continue
#         out.append((src, _safe_float(attrs.get("score", attrs.get("weight", 0.0))), rel))
#     best = {}
#     for n, s, r in out:
#         if n not in best or s > best[n][0]:
#             best[n] = (s, r)
#     return sorted([(n, s, r) for n, (s, r) in best.items()], key=lambda x: (-x[1], _source_sort_key(x[0])))


# def _source_text(G: nx.MultiDiGraph, source: str) -> str:
#     if source in G.nodes:
#         return normalize_text(G.nodes[source].get("text", ""))
#     return ""


# def _question_tokens(question: str) -> Set[str]:
#     return token_set(question) - GENERIC_TERMS


# def _bridge_question_score(G: nx.MultiDiGraph, question: str, bridge_source: str) -> float:
#     qt = _question_tokens(question)
#     st = token_set(_source_text(G, bridge_source)) - GENERIC_TERMS
#     if not qt or not st:
#         return 0.0
#     return len(qt & st) / max(len(qt), 1)


# def _operation_bonus(question: str, detail: Dict[str, Any]) -> float:
#     q = normalize_text(question)
#     cues = set(extract_formula_cues(question)) or {"direct_lookup"}
#     source = detail.get("source", "")
#     kind = detail.get("kind", source_kind(source))
#     metrics = " ".join(detail.get("metrics", []))
#     best = detail.get("best_fact") or {}
#     value = str(best.get("value") or "")
#     bonus = 0.0

#     if ("part_to_whole_ratio" in cues or "inverse_percent_total" in cues):
#         if "total" in metrics:
#             bonus += 0.07
#         if "%" in value or "percent" in metrics:
#             bonus += 0.04
#     if ("growth_rate" in cues or "difference" in cues) and kind == "table":
#         bonus += 0.03
#     if "comparison" in cues and kind == "table":
#         bonus += 0.03
#     if "sum_or_total" in cues and "total" in metrics:
#         bonus += 0.04
#     if "average_or_ratio" in cues and kind == "table":
#         bonus += 0.03
#     if "direct_lookup" in cues and kind == "text" and any(x in q for x in ["assuming", "without", "if "]):
#         bonus += 0.03
#     return bonus


# def _should_expand_mixed(question: str, details: Dict[str, Dict[str, Any]]) -> bool:
#     cues = set(extract_formula_cues(question))
#     if cues & MIXED_EXPANSION_CUES:
#         return True
#     q = normalize_text(question)
#     if any(x in q for x in ["assuming", "without", "if ", "maturity date", "held", "would", "portion of"]):
#         return True
#     # If strong table sources dominate and almost no text is present, mixed expansion is worth trying.
#     ranked = sorted(details.values(), key=lambda d: -d.get("score", d.get("best_score", 0.0)))[:5]
#     if any(d.get("kind") == "table" for d in ranked) and sum(d.get("kind") == "text" for d in ranked) <= 1:
#         return True
#     return False


# def _make_expanded_detail(G: nx.MultiDiGraph, source: str, expanded_from: str, parent_score: float, bridge_score: float, rel: str, question: str) -> Dict[str, Any]:
#     q_bonus = _bridge_question_score(G, question, source)
#     # Make bridge texts competitive enough to enter top-5/top-10, but still below strong primary table facts.
#     score = 0.58 * parent_score + 0.34 * bridge_score + 0.08 * q_bonus
#     node_attrs = G.nodes[source] if source in G.nodes else {}
#     fake_fact = {
#         "fact_id": f"expanded:{expanded_from}->{source}",
#         "source": source,
#         "metric": None,
#         "period": None,
#         "value": None,
#         "score": round(score, 4),
#         "text": node_attrs.get("text", ""),
#     }
#     return {
#         "source": source,
#         "kind": source_kind(source),
#         "best_score": round(score, 4),
#         "base_score": 0.0,
#         "score": round(score, 4),
#         "best_fact": fake_fact,
#         "fact_count": 0,
#         "metrics": [],
#         "periods": [],
#         "expanded": True,
#         "expanded_from": expanded_from,
#         "bridge_rel": rel,
#         "bridge_score": round(bridge_score, 4),
#     }


# def apply_operation_source_bonus(question: str, details: Dict[str, Dict[str, Any]]) -> None:
#     for d in details.values():
#         bonus = _operation_bonus(question, d)
#         d["score"] = round(_safe_float(d.get("best_score")) + bonus, 4)


# def expand_with_bridge_sources(
#     G: nx.MultiDiGraph,
#     question: str,
#     details: Dict[str, Dict[str, Any]],
#     max_bridge_text_per_table: int = 2,
#     max_total_expansions: int = 5,
#     bridge_threshold: float = 0.12,
# ) -> Dict[str, Dict[str, Any]]:
#     if not _should_expand_mixed(question, details):
#         return details

#     expanded = dict(details)
#     table_details = sorted(
#         [d for d in expanded.values() if d.get("kind") == "table"],
#         key=lambda d: (-_safe_float(d.get("score", d.get("best_score"))), _source_sort_key(d.get("source", ""))),
#     )
#     added = 0
#     for td in table_details[:5]:
#         if added >= max_total_expansions:
#             break
#         table_source = td["source"]
#         parent_score = _safe_float(td.get("score", td.get("best_score")))
#         bridges = get_bridge_neighbors(G, table_source, target_kind="text")
#         used_for_table = 0
#         for text_source, bscore, rel in bridges:
#             if added >= max_total_expansions or used_for_table >= max_bridge_text_per_table:
#                 break
#             if bscore < bridge_threshold:
#                 continue
#             if text_source in expanded:
#                 # Promote existing bridge-connected text if it is already present.
#                 existing = expanded[text_source]
#                 bridge_candidate_score = 0.58 * parent_score + 0.34 * bscore + 0.08 * _bridge_question_score(G, question, text_source)
#                 if bridge_candidate_score > _safe_float(existing.get("score", existing.get("best_score"))):
#                     existing["score"] = round(bridge_candidate_score, 4)
#                     existing["expanded"] = True
#                     existing["expanded_from"] = table_source
#                     existing["bridge_rel"] = rel
#                     existing["bridge_score"] = round(bscore, 4)
#                 used_for_table += 1
#                 continue
#             expanded[text_source] = _make_expanded_detail(G, text_source, table_source, parent_score, bscore, rel, question)
#             added += 1
#             used_for_table += 1
#     return expanded


# def rank_evidence_sources(
#     G: nx.MultiDiGraph,
#     q_id: str,
#     question: str,
#     top_k: int = 20,
#     enable_bridge_expansion: bool = True,
# ) -> Tuple[List[str], List[Dict[str, Any]]]:
#     rows = candidate_fact_rows(G, q_id)
#     details = source_details_from_rows(rows)
#     apply_operation_source_bonus(question, details)
#     if enable_bridge_expansion:
#         details = expand_with_bridge_sources(G, question, details)

#     ranked_details = sorted(
#         details.values(),
#         key=lambda d: (
#             -_safe_float(d.get("score", d.get("best_score"))),
#             # Prefer primary over expanded at identical score.
#             1 if d.get("expanded") else 0,
#             _source_sort_key(d.get("source", "")),
#         ),
#     )
#     ranked_details = ranked_details[:top_k]
#     ranked_sources = [d["source"] for d in ranked_details]
#     return ranked_sources, ranked_details


# def bridge_related_sources(G: nx.MultiDiGraph, source: str) -> Set[str]:
#     return {n for n, _, _ in get_bridge_neighbors(G, source)}


# def relaxed_any_covered(G: nx.MultiDiGraph, gold: Set[str], selected: Set[str]) -> bool:
#     if gold & selected:
#         return True
#     for s in selected:
#         if bridge_related_sources(G, s) & gold:
#             return True
#     return False


# def relaxed_all_covered(G: nx.MultiDiGraph, gold: Set[str], selected: Set[str]) -> bool:
#     for g in gold:
#         if g in selected:
#             continue
#         covered = False
#         for s in selected:
#             if g in bridge_related_sources(G, s):
#                 covered = True
#                 break
#         if not covered:
#             return False
#     return True

"""
Mixed-strict-oriented evidence-source ranker for FinGraphQA.

This module keeps the stable candidate-fact source ranking, then applies a
conservative table-anchor-conditioned text selector for mixed table-text cases.

Goal:
  Improve exact mixed source recall without flooding top-k with extra text.
  No gold labels, programs, or answers are used.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import re

import networkx as nx

try:
    from schema import NODE_FINANCIAL_FACT, REL_CANDIDATE_OPERAND, BRIDGE_RELS
except Exception:  # pragma: no cover
    NODE_FINANCIAL_FACT = "financial_fact"
    REL_CANDIDATE_OPERAND = "candidate_operand"
    BRIDGE_RELS = {"supported_by_text", "describes_table", "table_text_bridge"}

try:
    from preprocess import extract_formula_cues, normalize_text, token_set
except Exception:  # pragma: no cover
    def normalize_text(s: str) -> str:
        return re.sub(r"\s+", " ", str(s).lower()).strip()
    def token_set(s: str) -> Set[str]:
        return set(re.findall(r"[a-zA-Z][a-zA-Z\-]+", normalize_text(s)))
    def extract_formula_cues(question: str) -> List[str]:
        return []


GENERIC_TERMS = {
    "the", "and", "for", "with", "from", "that", "this", "what", "was", "were",
    "are", "how", "much", "many", "percent", "percentage", "change", "rate",
    "total", "amount", "value", "fair", "income", "expense", "expenses",
    "asset", "assets", "cash", "net", "basis", "year", "years", "million",
    "millions", "thousand", "thousands", "billion", "billions", "table",
}

MIXED_TEXT_CUES = {
    "percent", "percentage", "portion", "change", "increase", "decrease",
    "difference", "average", "exceed", "greater", "without", "assuming",
    "maturity date", "repurchase", "leased", "facilities", "authorization",
    "withdrawn", "translation gain",
}

MIXED_FORMULA_CUES = {
    "growth_rate", "difference", "part_to_whole_ratio", "inverse_percent_total",
    "comparison", "average_or_ratio",
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _source_sort_key(src: str) -> Tuple[int, int, str]:
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


def _content_tokens(text: str) -> Set[str]:
    return token_set(text) - GENERIC_TERMS


def _source_text(G: nx.MultiDiGraph, source: str) -> str:
    if source in G.nodes:
        return normalize_text(G.nodes[source].get("text", ""))
    return ""


def get_fact_source(G: nx.MultiDiGraph, fact_id: str) -> Optional[str]:
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
        if source_node.startswith("text_") or source_node.startswith("table_"):
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
                "score": score,
                "best_fact": row,
                "fact_count": 1,
                "metrics": set([row.get("metric")]) if row.get("metric") else set(),
                "periods": set([row.get("period")]) if row.get("period") else set(),
                "expanded": False,
                "expanded_from": None,
                "bridge_rel": None,
                "bridge_score": 0.0,
            }
        else:
            d = details[source]
            d["fact_count"] += 1
            if row.get("metric"):
                d["metrics"].add(row.get("metric"))
            if row.get("period"):
                d["periods"].add(row.get("period"))
            if score > d["best_score"]:
                d["best_score"] = score
                d["base_score"] = score
                d["score"] = score
                d["best_fact"] = row

    for d in details.values():
        d["metrics"] = sorted(str(x) for x in d["metrics"] if x is not None)
        d["periods"] = sorted(str(x) for x in d["periods"] if x is not None)

    return details


def get_bridge_neighbors(
    G: nx.MultiDiGraph,
    source: str,
    target_kind: Optional[str] = None,
) -> List[Tuple[str, float, str, Dict[str, Any]]]:
    """Return bridge neighbors with their best edge score and attrs."""
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
        score = _safe_float(attrs.get("bridge_score", attrs.get("score", attrs.get("weight", 0.0))))
        out.append((dst, score, rel, dict(attrs)))

    for src, _, attrs in G.in_edges(source, data=True):
        rel = attrs.get("rel")
        if rel not in BRIDGE_RELS:
            continue
        if not isinstance(src, str) or not (src.startswith("text_") or src.startswith("table_")):
            continue
        if target_kind and source_kind(src) != target_kind:
            continue
        score = _safe_float(attrs.get("bridge_score", attrs.get("score", attrs.get("weight", 0.0))))
        out.append((src, score, rel, dict(attrs)))

    best: Dict[str, Tuple[float, str, Dict[str, Any]]] = {}
    for n, s, r, attrs in out:
        if n not in best or s > best[n][0]:
            best[n] = (s, r, attrs)

    return sorted(
        [(n, s, r, attrs) for n, (s, r, attrs) in best.items()],
        key=lambda x: (-x[1], _source_sort_key(x[0])),
    )


def _operation_bonus(question: str, detail: Dict[str, Any]) -> float:
    """
    Keep operation bonus deliberately small.
    The previous version over-boosted table sources and hurt table-only/mixed strict.
    """
    cues = set(extract_formula_cues(question)) or {"direct_lookup"}
    kind = detail.get("kind", source_kind(detail.get("source", "")))
    metrics = " ".join(detail.get("metrics", []))
    best = detail.get("best_fact") or {}
    value = str(best.get("value") or "")

    bonus = 0.0

    if ("part_to_whole_ratio" in cues or "inverse_percent_total" in cues):
        if "total" in metrics:
            bonus += 0.035
        if "%" in value or "percent" in metrics:
            bonus += 0.025

    if ("growth_rate" in cues or "difference" in cues) and kind == "table":
        bonus += 0.02

    if "comparison" in cues and kind == "table":
        bonus += 0.02

    if "sum_or_total" in cues and "total" in metrics:
        bonus += 0.025

    if "average_or_ratio" in cues and kind == "table":
        bonus += 0.02

    return min(bonus, 0.07)


def apply_operation_source_bonus(question: str, details: Dict[str, Dict[str, Any]]) -> None:
    for d in details.values():
        bonus = _operation_bonus(question, d)
        d["score"] = round(_safe_float(d.get("best_score")) + bonus, 4)


def _is_mixed_risk(question: str) -> bool:
    q = normalize_text(question)
    cues = set(extract_formula_cues(question))
    if cues & MIXED_FORMULA_CUES:
        return True
    if any(cue in q for cue in MIXED_TEXT_CUES):
        return True
    return False


def _question_overlap(question: str, text: str) -> float:
    q = _content_tokens(question)
    t = _content_tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / max(len(q), 1)


def _anchor_overlap(G: nx.MultiDiGraph, table_source: str, text_source: str) -> float:
    tt = _content_tokens(_source_text(G, table_source))
    st = _content_tokens(_source_text(G, text_source))
    if not tt or not st:
        return 0.0
    return len(tt & st) / max(len(tt), 1)


def _bridge_selector_score(
    G: nx.MultiDiGraph,
    question: str,
    table_source: str,
    text_source: str,
    edge_score: float,
    edge_attrs: Dict[str, Any],
) -> float:
    """
    Table-anchor-conditioned text score.

    This is the main mixed-strict mechanism:
      choose exact supporting text for an already strong table anchor.
    """
    q_overlap = _question_overlap(question, _source_text(G, text_source))
    a_overlap = _anchor_overlap(G, table_source, text_source)

    metric_overlap = _safe_float(edge_attrs.get("metric_overlap", 0.0))
    period_overlap = _safe_float(edge_attrs.get("period_overlap", 0.0))
    number_overlap = _safe_float(edge_attrs.get("number_overlap", 0.0))
    retrieval_prior = _safe_float(edge_attrs.get("retrieval_prior", 0.0))
    position_score = _safe_float(edge_attrs.get("position_score", 0.0))
    cue_bonus = _safe_float(edge_attrs.get("cue_bonus", 0.0))

    score = (
        0.24 * edge_score
        + 0.24 * q_overlap
        + 0.22 * a_overlap
        + 0.12 * metric_overlap
        + 0.07 * period_overlap
        + 0.04 * number_overlap
        + 0.04 * retrieval_prior
        + 0.03 * position_score
        + 0.04 * cue_bonus
    )
    return round(score, 4)


def _make_bridge_detail(
    G: nx.MultiDiGraph,
    text_source: str,
    table_source: str,
    selector_score: float,
    edge_score: float,
    rel: str,
) -> Dict[str, Any]:
    node_attrs = G.nodes[text_source] if text_source in G.nodes else {}
    fake_fact = {
        "fact_id": f"bridge:{table_source}->{text_source}",
        "source": text_source,
        "metric": None,
        "period": None,
        "value": None,
        "score": selector_score,
        "text": node_attrs.get("text", ""),
    }
    return {
        "source": text_source,
        "kind": "text",
        "best_score": selector_score,
        "base_score": 0.0,
        "score": selector_score,
        "best_fact": fake_fact,
        "fact_count": 0,
        "metrics": [],
        "periods": [],
        "expanded": True,
        "expanded_from": table_source,
        "bridge_rel": rel,
        "bridge_score": edge_score,
        "selector_score": selector_score,
    }


def _select_bridge_text_for_tables(
    G: nx.MultiDiGraph,
    question: str,
    ordered_details: List[Dict[str, Any]],
    existing_sources: Set[str],
    max_tables: int = 4,
    threshold: float = 0.16,
) -> Optional[Tuple[str, str, Dict[str, Any], float]]:
    """
    Select one bridge text for the strongest table anchors.

    Returns:
      (text_source, table_source, detail, selector_score)
    """
    table_details = [d for d in ordered_details if d.get("kind") == "table"][:max_tables]

    best: Optional[Tuple[str, str, Dict[str, Any], float]] = None

    for td in table_details:
        table_source = td["source"]
        for text_source, edge_score, rel, edge_attrs in get_bridge_neighbors(G, table_source, target_kind="text"):
            # Existing text can still be promoted if it is currently too low.
            selector_score = _bridge_selector_score(
                G=G,
                question=question,
                table_source=table_source,
                text_source=text_source,
                edge_score=edge_score,
                edge_attrs=edge_attrs,
            )

            if selector_score < threshold:
                continue

            detail = _make_bridge_detail(
                G=G,
                text_source=text_source,
                table_source=table_source,
                selector_score=selector_score,
                edge_score=edge_score,
                rel=rel,
            )

            if best is None or selector_score > best[3]:
                best = (text_source, table_source, detail, selector_score)

    return best


def _interleave_bridge_text(
    ordered_details: List[Dict[str, Any]],
    bridge_choice: Tuple[str, str, Dict[str, Any], float],
) -> List[Dict[str, Any]]:
    text_source, table_source, bridge_detail, _ = bridge_choice

    # Remove existing occurrence of the selected text, if any, so we can promote it.
    filtered = [d for d in ordered_details if d.get("source") != text_source]

    out: List[Dict[str, Any]] = []
    inserted = False

    for d in filtered:
        out.append(d)

        if not inserted and d.get("source") == table_source:
            out.append(bridge_detail)
            inserted = True

    if not inserted:
        out.append(bridge_detail)

    # Deduplicate just in case.
    seen = set()
    deduped = []
    for d in out:
        src = d.get("source")
        if src in seen:
            continue
        seen.add(src)
        deduped.append(d)

    return deduped


def expand_with_bridge_sources(
    G: nx.MultiDiGraph,
    question: str,
    details: Dict[str, Dict[str, Any]],
    max_total_expansions: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Backward-compatible API.

    Unlike the previous version, this does not blindly add many bridge texts.
    It adds/promotes at most one table-anchor-conditioned text.
    """
    if not _is_mixed_risk(question):
        return details

    ordered = sorted(
        details.values(),
        key=lambda d: (
            -_safe_float(d.get("score", d.get("best_score"))),
            1 if d.get("expanded") else 0,
            _source_sort_key(d.get("source", "")),
        ),
    )

    # Avoid extra text if top-5 already contains two or more text sources.
    top5 = ordered[:5]
    if sum(1 for d in top5 if d.get("kind") == "text") >= 2:
        return details

    choice = _select_bridge_text_for_tables(
        G=G,
        question=question,
        ordered_details=ordered,
        existing_sources=set(details.keys()),
    )
    if choice is None:
        return details

    ordered2 = _interleave_bridge_text(ordered, choice)

    # Convert back to dict preserving updated selected bridge detail.
    new_details = {d["source"]: d for d in ordered2}
    return new_details


def rank_evidence_sources(
    G: nx.MultiDiGraph,
    q_id: str,
    question: str,
    top_k: int = 20,
    enable_bridge_expansion: bool = True,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    rows = candidate_fact_rows(G, q_id)
    details = source_details_from_rows(rows)

    apply_operation_source_bonus(question, details)

    ranked_details = sorted(
        details.values(),
        key=lambda d: (
            -_safe_float(d.get("score", d.get("best_score"))),
            1 if d.get("expanded") else 0,
            _source_sort_key(d.get("source", "")),
        ),
    )

    if enable_bridge_expansion:
        top5 = ranked_details[:5]
        # Use internal bridge selector only when it can address a mixed case.
        if _is_mixed_risk(question) and any(d.get("kind") == "table" for d in top5):
            choice = _select_bridge_text_for_tables(
                G=G,
                question=question,
                ordered_details=ranked_details,
                existing_sources={d["source"] for d in ranked_details},
            )
            if choice is not None:
                # Insert/promote exactly one selected bridge text after its table anchor.
                ranked_details = _interleave_bridge_text(ranked_details, choice)

    ranked_details = ranked_details[:top_k]
    ranked_sources = [d["source"] for d in ranked_details]
    return ranked_sources, ranked_details


def bridge_related_sources(G: nx.MultiDiGraph, source: str) -> Set[str]:
    return {n for n, _, _, _ in get_bridge_neighbors(G, source)}


def relaxed_any_covered(G: nx.MultiDiGraph, gold: Set[str], selected: Set[str]) -> bool:
    if gold & selected:
        return True
    for s in selected:
        if bridge_related_sources(G, s) & gold:
            return True
    return False


def relaxed_all_covered(G: nx.MultiDiGraph, gold: Set[str], selected: Set[str]) -> bool:
    for g in gold:
        if g in selected:
            continue
        covered = False
        for s in selected:
            if g in bridge_related_sources(G, s):
                covered = True
                break
        if not covered:
            return False
    return True
