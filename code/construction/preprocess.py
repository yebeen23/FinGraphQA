import re
from typing import Any, Dict, List, Tuple


YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
NUMBER_RE = re.compile(
    r"""
    (?:
        \$\s*)?
    -?
    \d+(?:,\d{3})*
    (?:\.\d+)?
    %?
    """,
    re.VERBOSE,
)


def get_all_text(example: Dict[str, Any]) -> List[str]:
    return example.get("pre_text", []) + example.get("post_text", [])


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def extract_years(text: str) -> List[str]:
    return sorted(set(match.group(0) for match in YEAR_RE.finditer(text)))


def extract_numbers(text: str) -> List[str]:
    nums = []
    for m in NUMBER_RE.finditer(text):
        raw = m.group(0).strip()
        if raw:
            nums.append(raw)
    return nums


def parse_number(raw: str) -> float | None:
    """
    Converts raw numeric string into float.
    Keeps percent as numeric 23.6, not 0.236 here.
    Percent handling is done later depending on formula.
    """
    if raw is None:
        return None

    s = raw.lower().strip()
    s = s.replace("$", "").replace(",", "").strip()

    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1].strip()

    # Handle forms like "-32 ( 32 )"
    # Prefer explicit leading negative.
    try:
        return float(s)
    except ValueError:
        # fallback: extract first numeric token
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        if m:
            return float(m.group(0))
    return None


def detect_units(text: str) -> List[str]:
    text_l = text.lower()
    units = []

    unit_terms = {
        "thousand": ["thousand", "thousands", "in thousands"],
        "million": ["million", "millions", "in millions"],
        "billion": ["billion", "billions", "in billions"],
        "percent": ["%", "percent", "percentage"],
        "basis_points": ["basis points", "bps"],
        "dollar": ["$", "dollar", "dollars"],
        "shares": ["shares", "share"],
        "per_share": ["per share"],
        "years": ["year", "years"],
    }

    for unit, patterns in unit_terms.items():
        if any(p in text_l for p in patterns):
            units.append(unit)

    return sorted(set(units))


def linearize_table_row(table: List[List[str]], row_idx: int) -> str:
    """
    Converts table row into FinQA-like textual evidence.
    """
    if not table:
        return ""

    headers = table[0]
    row = table[row_idx]

    if row_idx == 0:
        return " ; ".join(headers)

    row_name = row[0] if row else ""

    pieces = []
    for col_idx in range(1, min(len(headers), len(row))):
        header = headers[col_idx]
        value = row[col_idx]
        if value:
            pieces.append(f"the {row_name} of {header} is {value}")

    return " ; ".join(pieces) + " ;"


def build_evidence_candidates(example: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns:
      {
        "text_0": "...",
        "text_1": "...",
        "table_0": "...",
        "table_1": "..."
      }
    """
    candidates = {}

    all_text = get_all_text(example)
    for i, sent in enumerate(all_text):
        candidates[f"text_{i}"] = normalize_text(sent)

    table = example.get("table", [])
    for i in range(len(table)):
        candidates[f"table_{i}"] = normalize_text(linearize_table_row(table, i))

    return candidates


def get_gold_evidence_ids(example: Dict[str, Any]) -> set[str]:
    qa = example.get("qa", {})
    return set(qa.get("gold_inds", {}).keys())


METRIC_TERMS = [
    "interest expense",
    "net revenue",
    "revenue",
    "growth rate",
    "total operating expenses",
    "operating expenses",
    "aircraft fuel expense",
    "fuel expense",
    "available-for-sale investments",
    "total cash and investments",
    "cash and investments",
    "fair value",
    "aggregate cost",
    "level 2",
    "level 3",
    "shares",
    "number of shares",
    "weighted average grant date fair value",
    "grant date fair value",
    "stock-based compensation expense",
    "compensation expense",
    "intangible assets",
    "customer-related intangibles",
    "network location intangibles",
    "amortization expense",
    "loans held-for-sale",
]


def extract_metrics(text: str) -> List[str]:
    text_l = normalize_text(text)
    found = []

    for term in METRIC_TERMS:
        if term in text_l:
            found.append(term)

    return sorted(set(found))


def extract_formula_cues(question: str) -> List[str]:
    q = normalize_text(question)
    cues = []

    if any(x in q for x in ["growth rate", "percentage change", "percent change"]):
        cues.append("growth_rate")

    if any(x in q for x in ["what percentage", "what percent", "comprised of"]):
        cues.append("part_to_whole_ratio")

    if any(x in q for x in ["average", "per "]):
        cues.append("average_or_ratio")

    if any(x in q for x in ["change", "increase", "decrease", "difference"]):
        cues.append("difference")

    if any(x in q for x in ["exceed", "greater", "larger", "did "]):
        cues.append("comparison")

    if any(x in q for x in ["total", "sum"]):
        cues.append("sum_or_total")

    if any(x in q for x in ["annual amortization", "amortization expenses"]):
        cues.append("annual_amortization")

    return sorted(set(cues))