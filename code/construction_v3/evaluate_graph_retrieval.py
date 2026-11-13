from __future__ import annotations

from pathlib import Path
import argparse
from collections import defaultdict, Counter
import json
import re
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

sys.path.append(str(CURRENT_DIR))

from graph_builder import build_inference_graph
from load_data import get_gold_evidence, load_split
from preprocess import extract_formula_cues
from evidence_set_ranker import rank_evidence_sources, relaxed_all_covered, relaxed_any_covered


# -----------------------------------------------------------------------------
# Conservative mixed-evidence expansion
# -----------------------------------------------------------------------------
# This is the current stable version:
#   - internal evidence_set_ranker bridge expansion is forced OFF in evaluate()
#   - only one external table -> text fallback can be inserted
#   - fallback is triggered only by explicit question text cues
#   - if top-5 already contains text, no extra text is inserted

MIXED_RISK_CUES = {
    "percent",
    "percentage",
    "portion",
    "as part of",
    "comprised",
    "change",
    "increase",
    "decrease",
    "difference",
    "average",
    "exceed",
    "greater",
    "without",
    "assuming",
}

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "was", "were",
    "is", "are", "what", "how", "much", "many", "from", "between", "during",
    "with", "as", "at", "on", "by", "that", "this", "these", "those", "be",
    "been", "would", "could", "should", "did", "does", "do", "it", "its", "into",
    "than", "then", "there", "their", "our", "we", "you", "your", "which", "who",
}

FINANCE_KEYWORDS = {
    "revenue", "sales", "income", "expense", "expenses", "asset", "assets",
    "cash", "debt", "tax", "taxes", "shares", "share", "dividend", "dividends",
    "goodwill", "facilities", "facility", "leased", "lease", "purchase", "price",
    "payments", "payment", "securities", "security", "bonds", "bond", "benefits",
    "repurchase", "repurchases", "amortization", "depreciation", "interest",
    "operating", "financing", "activities", "maturity", "translation", "gain",
    "loss", "losses", "contingent", "acquisition", "dilutive", "securitization",
}


def _is_mixed_risk_question(question: str) -> bool:
    q = (question or "").lower()
    return any(cue in q for cue in MIXED_RISK_CUES)


def _simple_tokens(text: str) -> Set[str]:
    text = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+|\d{4}", text)
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def _get_retrieved_text_ids(example: Dict[str, Any]) -> Set[str]:
    """
    Optional weak prior from FinQA retrieval fields if they exist.
    This does not use gold evidence.
    """
    out: Set[str] = set()

    for key in ["text_retrieved_all", "text_retrieved"]:
        items = example.get(key, []) or []
        for item in items:
            text_id = None
            if isinstance(item, dict):
                text_id = item.get("ind") or item.get("id") or item.get("text_id")
            elif isinstance(item, int):
                text_id = f"text_{item}"
            elif isinstance(item, str) and item.startswith("text_"):
                text_id = item

            if isinstance(text_id, str) and text_id.startswith("text_"):
                out.add(text_id)

    return out


def _collect_text_candidates(example: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build text_id -> text candidates from pre_text and post_text.

    FinQA convention:
      pre_text:  text_0 ... text_{len(pre_text)-1}
      post_text: continues after pre_text
    """
    pre = example.get("pre_text", []) or []
    post = example.get("post_text", []) or []

    candidates: List[Dict[str, Any]] = []

    for i, sent in enumerate(pre):
        candidates.append({
            "text_id": f"text_{i}",
            "text": str(sent),
            "section": "pre",
            "position": i,
        })

    offset = len(pre)
    for j, sent in enumerate(post):
        candidates.append({
            "text_id": f"text_{offset + j}",
            "text": str(sent),
            "section": "post",
            "position": j,
        })

    return candidates


def _nearby_context_ids(example: Dict[str, Any], max_pre: int = 5, max_post: int = 5) -> Set[str]:
    pre = example.get("pre_text", []) or []
    post = example.get("post_text", []) or []

    ids: Set[str] = set()

    pre_start = max(0, len(pre) - max_pre)
    for i in range(pre_start, len(pre)):
        ids.add(f"text_{i}")

    offset = len(pre)
    for j in range(min(max_post, len(post))):
        ids.add(f"text_{offset + j}")

    return ids


def _anchor_detail_for_source(
    ranked_sources: List[str],
    ranked_details: List[Dict[str, Any]],
    anchor_source: str,
) -> Dict[str, Any]:
    for src, detail in zip(ranked_sources, ranked_details):
        if src == anchor_source:
            return detail or {}
    return {}


def _anchor_text_from_detail(detail: Dict[str, Any]) -> str:
    best = detail.get("best_fact") or {}
    parts: List[str] = []
    for key in ["metric", "period", "value"]:
        value = best.get(key)
        if value is not None:
            parts.append(str(value))
    return " ".join(parts)


def _score_text_candidate(
    question: str,
    text: str,
    text_id: str,
    nearby_ids: Set[str],
    retrieved_ids: Set[str],
    anchor_text: str = "",
) -> float:
    q_tokens = _simple_tokens(question)
    t_tokens = _simple_tokens(text)
    a_tokens = _simple_tokens(anchor_text)

    if not t_tokens:
        return -1.0

    q_overlap = q_tokens & t_tokens
    a_overlap = a_tokens & t_tokens
    finance_overlap = q_overlap & FINANCE_KEYWORDS

    score = 0.0

    # Main signal: question-text lexical overlap.
    score += 2.0 * (len(q_overlap) / max(len(q_tokens), 1))
    score += 0.15 * len(q_overlap)

    # Anchor table metric/value overlap.
    score += 1.4 * (len(a_overlap) / max(len(a_tokens), 1))
    score += 0.20 * len(a_overlap)

    # Finance-specific words matter more.
    score += 0.35 * len(finance_overlap)

    # Year matching is useful in FinQA.
    q_years = set(re.findall(r"(?:19|20)\d{2}", question or ""))
    t_years = set(re.findall(r"(?:19|20)\d{2}", text or ""))
    a_years = set(re.findall(r"(?:19|20)\d{2}", anchor_text or ""))
    score += 0.30 * len(q_years & t_years)
    score += 0.20 * len(a_years & t_years)

    # Weak priors.
    if text_id in nearby_ids:
        score += 0.15
    if text_id in retrieved_ids:
        score += 0.25

    # Avoid useless short fragments.
    if len(str(text).split()) < 5:
        score -= 0.20

    return score


def _rank_context_text_ids(
    example: Dict[str, Any],
    question: str,
    existing_sources: Set[str],
    anchor_text: str = "",
    max_candidates: int = 5,
) -> List[Tuple[str, float]]:
    nearby_ids = _nearby_context_ids(example, max_pre=5, max_post=5)
    retrieved_ids = _get_retrieved_text_ids(example)
    candidates = _collect_text_candidates(example)

    scored: List[Tuple[str, float]] = []

    for item in candidates:
        text_id = item["text_id"]
        if text_id in existing_sources:
            continue

        score = _score_text_candidate(
            question=question,
            text=item["text"],
            text_id=text_id,
            nearby_ids=nearby_ids,
            retrieved_ids=retrieved_ids,
            anchor_text=anchor_text,
        )
        scored.append((text_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_candidates]


def _make_expanded_detail(
    text_id: str,
    from_source: Optional[str] = None,
    bridge_score: float = 0.0,
) -> Dict[str, Any]:
    return {
        "source": text_id,
        "kind": "text",
        "score": bridge_score,
        "best_score": bridge_score,
        "base_score": 0.0,
        "fact_count": 0,
        "count": 0,
        "expanded": True,
        "expanded_from": from_source,
        "from": from_source,
        "bridge_rel": "question_overlap_context_fallback",
        "bridge_score": bridge_score,
        "best_fact": {
            "fact_id": f"expanded:{text_id}",
            "metric": None,
            "period": None,
            "value": None,
            "raw_value": None,
            "score": bridge_score,
            "is_date_number": False,
        },
    }


def force_simple_context_expansion(
    example: Dict[str, Any],
    ranked_sources: List[str],
    ranked_details: List[Dict[str, Any]],
    question: str,
    max_extra_texts: int = 1,
) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Conservative mixed-evidence fallback.

    Rule:
      - expand only if question text has explicit mixed-risk cues
      - use the first table source as anchor
      - choose one text by question overlap + anchor metric overlap
      - do not expand if top-5 already has a text source
    """
    if not ranked_sources:
        return ranked_sources, ranked_details

    if not _is_mixed_risk_question(question):
        return ranked_sources, ranked_details

    top5 = ranked_sources[:5]
    top5_text_count = sum(1 for s in top5 if str(s).startswith("text_"))
    top5_table_count = sum(1 for s in top5 if str(s).startswith("table_"))

    if top5_table_count == 0:
        return ranked_sources, ranked_details

    # Important: if top-5 already contains text, adding another text usually
    # hurts strict source ranking.
    if top5_text_count >= 1:
        return ranked_sources, ranked_details

    first_table = next((s for s in ranked_sources if str(s).startswith("table_")), None)
    if first_table is None:
        return ranked_sources, ranked_details

    existing = set(ranked_sources)

    anchor_detail = _anchor_detail_for_source(
        ranked_sources=ranked_sources,
        ranked_details=ranked_details,
        anchor_source=first_table,
    )
    anchor_text = _anchor_text_from_detail(anchor_detail)

    ranked_texts = _rank_context_text_ids(
        example=example,
        question=question,
        existing_sources=existing,
        anchor_text=anchor_text,
        max_candidates=5,
    )

    if not ranked_texts:
        return ranked_sources, ranked_details

    insert_texts = ranked_texts[:max_extra_texts]

    source_to_detail: Dict[str, Dict[str, Any]] = {}
    for i, src in enumerate(ranked_sources):
        if i < len(ranked_details):
            source_to_detail[src] = ranked_details[i]
        else:
            source_to_detail[src] = {
                "source": src,
                "kind": "unknown",
                "score": 0.0,
                "best_score": 0.0,
                "base_score": 0.0,
                "fact_count": 0,
                "expanded": False,
                "expanded_from": None,
                "bridge_score": 0.0,
                "best_fact": {},
            }

    new_sources: List[str] = []
    new_details: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    inserted = False

    for src in ranked_sources:
        if src in seen:
            continue

        new_sources.append(src)
        new_details.append(source_to_detail[src])
        seen.add(src)

        if src == first_table and not inserted:
            for text_id, score in insert_texts:
                if text_id in seen:
                    continue
                new_sources.append(text_id)
                new_details.append(
                    _make_expanded_detail(
                        text_id=text_id,
                        from_source=first_table,
                        bridge_score=score,
                    )
                )
                seen.add(text_id)
            inserted = True

    return new_sources, new_details


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------

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


def _update_group_stats(group_stats, group_name: str, gold: Set[str], ranked_sources: List[str], G: nx.MultiDiGraph, ks: List[int]) -> None:
    group_stats[group_name]["n"] += 1
    for k in ks:
        selected = set(ranked_sources[:k])
        group_stats[group_name][f"strict_any@{k}"] += float(bool(gold & selected))
        group_stats[group_name][f"strict_all@{k}"] += float(gold <= selected)
        group_stats[group_name][f"relaxed_any@{k}"] += float(relaxed_any_covered(G, gold, selected))
        group_stats[group_name][f"relaxed_all@{k}"] += float(relaxed_all_covered(G, gold, selected))


def _print_group_stats(title: str, group_stats, ks: List[int]) -> None:
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
        "bridge_score": detail.get("bridge_score"),
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


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _print_missed_sample(title: str, missed: List[Dict[str, Any]], verbose: bool, n: int = 10) -> None:
    print(f"\n[{title}]")
    if not missed:
        print("  EMPTY")
        return
    for m in missed[:n]:
        print(f"  idx={m['idx']} id={m['id']} modality={m['modality']}")
        print(f"    q={m['question']}")
        print(f"    gold={m['gold']}")
        print(f"    cues={m['cues']}")
        print(f"    top_sources={m['top_sources']}")
        if verbose:
            print("    top source details:")
            for d in m.get("top_details", []):
                bf = d.get("best_fact") or {}
                print(
                    "      "
                    f"source={d.get('source')} "
                    f"score={_safe_float(d.get('score', d.get('best_score', 0.0))):.4f} "
                    f"base={_safe_float(d.get('base_score', 0.0)):.4f} "
                    f"count={d.get('fact_count')} "
                    f"expanded={d.get('expanded')} "
                    f"from={d.get('expanded_from')} "
                    f"bridge={d.get('bridge_rel') or d.get('bridge_score')} "
                    f"fact={bf.get('fact_id')} "
                    f"metric={bf.get('metric')} "
                    f"period={bf.get('period')} "
                    f"value={bf.get('value')}"
                )


def evaluate(
    split: str,
    limit: Optional[int],
    ks: List[int],
    verbose_missed: bool,
    save_missed: bool,
    enable_bridge_expansion: bool,
) -> None:
    data = load_split(DATASET_DIR, split)
    if limit is not None:
        data = data[:limit]

    ks = sorted(set(ks))
    max_k = max(ks)
    totals = defaultdict(int)
    sums = defaultdict(float)
    modality_stats = defaultdict(lambda: defaultdict(float))
    cue_stats = defaultdict(lambda: defaultdict(float))
    missed_strict: List[Dict[str, Any]] = []
    missed_relaxed: List[Dict[str, Any]] = []
    expansion_counter = Counter()

    for idx, example in enumerate(data):
        gold = get_gold_evidence(example)
        if not gold:
            continue
        question = example.get("qa", {}).get("question", "")
        cues = extract_formula_cues(question) or ["direct_lookup"]
        modality = evidence_modality(gold)

        G, q_id = build_inference_graph(example)

        # Base ranking only. Ask for extra candidates because we hard-filter any
        # internally expanded source returned by evidence_set_ranker.
        raw_sources, raw_details = rank_evidence_sources(
            G,
            q_id,
            question=question,
            top_k=max_k * 3,
            enable_bridge_expansion=False,
        )

        ranked_sources: List[str] = []
        ranked_details: List[Dict[str, Any]] = []
        seen_sources: Set[str] = set()

        for src, detail in zip(raw_sources, raw_details):
            if detail.get("expanded"):
                expansion_counter["ranker_expanded_dropped"] += 1
                continue
            if src in seen_sources:
                continue
            ranked_sources.append(src)
            ranked_details.append(detail)
            seen_sources.add(src)
            if len(ranked_sources) >= max_k:
                break

        # Conservative external fallback only.
        if enable_bridge_expansion:
            ranked_sources, ranked_details = force_simple_context_expansion(
                example=example,
                ranked_sources=ranked_sources,
                ranked_details=ranked_details,
                question=question,
                max_extra_texts=1,
            )

        ranked_sources = ranked_sources[:max_k]
        ranked_details = ranked_details[:max_k]

        expansion_counter["examples"] += 1
        expansion_counter["expanded_sources"] += sum(1 for d in ranked_details if d.get("expanded"))
        if any(d.get("expanded") for d in ranked_details):
            expansion_counter["examples_with_expansion"] += 1

        totals["examples"] += 1
        for k in ks:
            selected = set(ranked_sources[:k])
            sums[f"strict_any@{k}"] += float(bool(gold & selected))
            sums[f"strict_all@{k}"] += float(gold <= selected)
            sums[f"relaxed_any@{k}"] += float(relaxed_any_covered(G, gold, selected))
            sums[f"relaxed_all@{k}"] += float(relaxed_all_covered(G, gold, selected))
        sums["gold_reachability"] += compute_reachability(G, q_id, gold)

        group_ks = [k for k in ks if k in {5, 10}]
        _update_group_stats(modality_stats, modality, gold, ranked_sources, G, group_ks)
        for cue in cues:
            _update_group_stats(cue_stats, cue, gold, ranked_sources, G, group_ks)

        selected_max = set(ranked_sources[:max_k])
        strict_ok = gold <= selected_max
        relaxed_ok = relaxed_all_covered(G, gold, selected_max)
        if not strict_ok:
            missed_strict.append({
                "idx": idx,
                "id": example.get("id"),
                "question": question,
                "gold": sorted(gold),
                "modality": modality,
                "cues": cues,
                "top_sources": ranked_sources[:max_k],
                "top_details": ranked_details[:max_k],
            })
        if not relaxed_ok:
            missed_relaxed.append({
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
    print("\n[Expansion diagnostics]")
    print(f"  examples with expansion: {expansion_counter['examples_with_expansion']}/{max(expansion_counter['examples'], 1)}")
    print(f"  expanded sources in top@{max_k}: {expansion_counter['expanded_sources']}")
    print(f"  internally expanded sources dropped: {expansion_counter['ranker_expanded_dropped']}")

    _print_group_stats("Recall by evidence modality", modality_stats, ks=[k for k in [5, 10] if k in ks])
    _print_group_stats("Recall by formula cue", cue_stats, ks=[k for k in [5, 10] if k in ks])

    miss_by_mod = Counter(m["modality"] for m in missed_strict)
    print("\n[Missed count by modality, strict@max_k]")
    for key, val in sorted(miss_by_mod.items()):
        print(f"  {key}: {val}")

    mixed_only = [m for m in missed_strict if m["modality"] == "mixed"]
    _print_missed_sample("Mixed-only missed examples sample", mixed_only, verbose=False, n=10)
    _print_missed_sample("Strict missed examples sample", missed_strict, verbose=verbose_missed, n=10)

    if save_missed:
        suffix = "all" if limit is None else str(limit)
        strict_path = OUTPUT_DIR / f"graph_retrieval_missed_{split}_{suffix}.jsonl"
        mixed_path = OUTPUT_DIR / f"graph_retrieval_missed_{split}_{suffix}_mixed_only.jsonl"
        relaxed_path = OUTPUT_DIR / f"graph_retrieval_missed_{split}_{suffix}_relaxed.jsonl"
        save_missed_cases(strict_path, missed_strict)
        save_missed_cases(mixed_path, mixed_only)
        save_missed_cases(relaxed_path, missed_relaxed)
        print(f"\nSaved strict missed cases to: {strict_path}")
        print(f"Saved mixed-only missed cases to: {mixed_path}")
        print(f"Saved relaxed missed cases to: {relaxed_path}")


def parse_ks(raw: str) -> List[int]:
    ks = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"k must be positive, got {value}")
        ks.append(value)
    if not ks:
        raise ValueError("At least one k value is required")
    return ks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "dev", "test", "private_test"])
    parser.add_argument("--limit", type=int, default=200, help="Number of examples. Use -1 for all.")
    parser.add_argument("--ks", default="1,3,5,10,20")
    parser.add_argument("--verbose_missed", "--verbose-missed", action="store_true")
    parser.add_argument("--save_missed", "--save-missed", action="store_true")
    parser.add_argument("--disable_bridge_expansion", "--disable-bridge-expansion", action="store_true")
    args = parser.parse_args()
    limit = None if args.limit is not None and args.limit < 0 else args.limit
    evaluate(
        split=args.split,
        limit=limit,
        ks=parse_ks(args.ks),
        verbose_missed=args.verbose_missed,
        save_missed=args.save_missed,
        enable_bridge_expansion=not args.disable_bridge_expansion,
    )


if __name__ == "__main__":
    main()
