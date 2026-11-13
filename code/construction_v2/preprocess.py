import re
from typing import Any, Dict, List, Optional, Sequence

from schema import SCALE_MULTIPLIER

MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04",
    "may": "05", "june": "06", "july": "07", "august": "08",
    "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "sept": "09", "oct": "10",
    "nov": "11", "dec": "12",
}

YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
DATE_RE = re.compile(
    r"\b("
    + "|".join(sorted(MONTHS.keys(), key=len, reverse=True))
    + r")\.?\s+(\d{1,2})\s*,?\s*((?:19|20)\d{2})\b",
    re.IGNORECASE,
)
QUARTER_RE = re.compile(r"\b(?:q([1-4])|([1-4])(?:st|nd|rd|th) quarter)\s*(?:of\s*)?((?:19|20)\d{2})?\b", re.IGNORECASE)

# Matches common FinQA numeric forms, including $6,427, 23.6%, (23,158), $(23,158), -17.1.
FIN_NUM_RE = re.compile(
    r"""
    (?P<currency>\$\s*)?
    (?P<paren_open>\(\s*)?
    (?P<sign>-)?
    (?P<num>\d+(?:,\d{3})*(?:\.\d+)?)
    \s*(?P<percent>%)?
    (?P<paren_close>\s*\))?
    """,
    re.VERBOSE,
)

STOP_METRIC_WORDS = {
    "the", "of", "is", "are", "was", "were", "in", "as", "at", "on", "for",
    "and", "or", "to", "from", "by", "with", "without", "amount", "amounts",
    "number", "numbers", "table", "year", "years", "date", "period", "ended",
    "dec", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov",
    "december", "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november",
}

COMPOUND_FIXES = {
    "priceper": "price per",
    "fuelexpense": "fuel expense",
    "totaloperating": "total operating",
    "operatingexpenses": "operating expenses",
    "grantdate": "grant date",
    "cashand": "cash and",
    "availableforsale": "available for sale",
    "noncurrent": "non current",
    "longterm": "long term",
}

METRIC_TERMS = [
    "interest expense", "annual interest expense", "net revenue", "revenue",
    "growth rate", "total operating expenses", "operating expenses",
    "aircraft fuel expense", "fuel expense", "average price per gallon",
    "gallons", "available for sale investments", "available-for-sale investments",
    "total cash and investments", "cash and investments", "cash", "fair value",
    "aggregate cost", "level 2", "level 3", "shares", "number of shares",
    "weighted average grant date fair value", "grant date fair value",
    "stock based compensation expense", "stock-based compensation expense",
    "compensation expense", "intangible assets", "customer related intangibles",
    "customer-related intangibles", "network location intangibles",
    "amortization expense", "amortization expenses", "loans held for sale",
    "loans held-for-sale", "net assets", "goodwill", "current assets",
    "non current assets", "property and equipment", "liabilities", "debt",
    "investment", "investments", "cost", "expense", "expenses", "income", "loss",
    "assets", "sales", "margin", "price", "tax", "taxes", "dividend",
]


def get_all_text(example: Dict[str, Any]) -> List[str]:
    return list(example.get("pre_text", [])) + list(example.get("post_text", []))


def normalize_text(s: Any) -> str:
    if s is None:
        return ""
    out = str(s).lower()
    for bad, good in COMPOUND_FIXES.items():
        out = out.replace(bad, good)
    out = out.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    out = re.sub(r"\s+", " ", out)
    return out.strip()


def normalize_metric_phrase(text: Any) -> str:
    text_l = normalize_text(text)
    text_l = re.sub(r"\$|%|\([^)]*\)", " ", text_l)
    text_l = YEAR_RE.sub(" ", text_l)
    text_l = re.sub(r"\b\d+(?:,\d{3})*(?:\.\d+)?\b", " ", text_l)
    text_l = re.sub(r"[^a-z0-9%\- ]+", " ", text_l)
    toks = [t for t in text_l.split() if t not in STOP_METRIC_WORDS]
    return " ".join(toks).strip()


def extract_years(text: Any) -> List[str]:
    return sorted(set(m.group(0) for m in YEAR_RE.finditer(normalize_text(text))))


def extract_periods(text: Any) -> List[Dict[str, Any]]:
    text_l = normalize_text(text)
    periods: List[Dict[str, Any]] = []
    seen = set()

    for m in DATE_RE.finditer(text_l):
        mon_raw, day_raw, year_raw = m.group(1).lower().rstrip("."), m.group(2), m.group(3)
        month = MONTHS.get(mon_raw)
        if month:
            value = f"{year_raw}-{month}-{int(day_raw):02d}"
            key = ("date", value)
            if key not in seen:
                periods.append({"type": "date", "value": value, "year": int(year_raw), "raw": m.group(0)})
                seen.add(key)

    for m in QUARTER_RE.finditer(text_l):
        q = m.group(1) or m.group(2)
        year = m.group(3)
        if q:
            value = f"Q{q}" + (f"-{year}" if year else "")
            key = ("quarter", value)
            if key not in seen:
                periods.append({"type": "quarter", "value": value, "year": int(year) if year else None, "quarter": int(q), "raw": m.group(0)})
                seen.add(key)

    for y in extract_years(text_l):
        key = ("year", y)
        if key not in seen:
            periods.append({"type": "year", "value": int(y), "year": int(y), "raw": y})
            seen.add(key)

    return periods


def detect_scale(text: Any) -> Optional[str]:
    t = normalize_text(text)
    if any(x in t for x in ["in billions", "billions", "billion"]):
        return "billion"
    if any(x in t for x in ["in millions", "millions", "million"]):
        return "million"
    if any(x in t for x in ["in thousands", "thousands", "thousand"]):
        return "thousand"
    return None


def detect_currency(text: Any) -> Optional[str]:
    t = normalize_text(text)
    if "$" in str(text) or "dollar" in t or "usd" in t or "u.s. dollar" in t:
        return "USD"
    return None


def detect_units(text: Any) -> List[str]:
    text_l = normalize_text(text)
    units = []
    unit_terms = {
        "thousand": ["thousand", "thousands", "in thousands"],
        "million": ["million", "millions", "in millions"],
        "billion": ["billion", "billions", "in billions"],
        "percent": ["%", "percent", "percentage"],
        "basis_points": ["basis points", "bps"],
        "dollar": ["$", "dollar", "dollars", "usd"],
        "shares": ["shares", "share"],
        "per_share": ["per share"],
        "years": ["year", "years"],
        "gallon": ["gallon", "gallons"],
    }
    for unit, patterns in unit_terms.items():
        if any(p in text_l for p in patterns):
            units.append(unit)
    return sorted(set(units))


def extract_numbers(text: Any) -> List[str]:
    return [m.group(0).strip() for m in FIN_NUM_RE.finditer(str(text)) if m.group(0).strip()]


def parse_financial_number(raw: Any, context: Any = "", inherited_scale: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None

    raw_s = str(raw).strip()
    m = FIN_NUM_RE.search(raw_s)
    if not m:
        return None

    num_s = m.group("num").replace(",", "")
    try:
        value = float(num_s)
    except ValueError:
        return None

    has_currency = bool(m.group("currency")) or "$" in raw_s or detect_currency(context) is not None
    has_percent = bool(m.group("percent")) or raw_s.strip().endswith("%")
    has_parentheses = bool(m.group("paren_open") and m.group("paren_close")) or ("(" in raw_s and ")" in raw_s)
    has_negative = bool(m.group("sign")) or (has_parentheses and not has_percent)

    signed_value = -value if has_negative else value
    scale = inherited_scale or detect_scale(context)
    currency = "USD" if has_currency else detect_currency(context)

    if has_percent:
        normalized_value = signed_value / 100.0
        unit = "percent"
    else:
        normalized_value = signed_value * SCALE_MULTIPLIER.get(scale, 1.0)
        unit = "dollar" if currency else None

    return {
        "raw": raw_s,
        "value": signed_value,
        "display_value": value,
        "normalized_value": normalized_value,
        "is_percent": has_percent,
        "currency": currency,
        "scale": scale,
        "unit": unit,
        "sign": "negative" if has_negative else "positive",
    }


def get_financial_numbers(text: Any, context: Any = "", inherited_scale: Optional[str] = None) -> List[Dict[str, Any]]:
    nums = []
    for raw in extract_numbers(text):
        parsed = parse_financial_number(raw, context=context, inherited_scale=inherited_scale)
        if parsed is not None:
            nums.append(parsed)
    return nums


def parse_number(raw: str) -> float | None:
    parsed = parse_financial_number(raw)
    return None if parsed is None else parsed["value"]


def infer_table_scale(example: Dict[str, Any]) -> Optional[str]:
    table = example.get("table", []) or []
    table_ori = example.get("table_ori", []) or []
    contexts: List[str] = []
    if table:
        contexts.append(" ".join(str(x) for x in table[0]))
    if table_ori:
        contexts.append(" ".join(str(x) for x in table_ori[0]))
    pre = example.get("pre_text", []) or []
    post = example.get("post_text", []) or []
    contexts.extend(pre[-3:])
    contexts.extend(post[:2])
    for c in contexts:
        scale = detect_scale(c)
        if scale:
            return scale
    return None


def is_period_like(text: Any) -> bool:
    t = normalize_text(text)
    if not t:
        return False
    if extract_periods(t):
        stripped = re.sub(r"[^a-z0-9 ]", " ", t).strip()
        return len(stripped.split()) <= 5
    return False


def extract_metrics(text: Any) -> List[str]:
    text_l = normalize_text(text)
    found = set()
    for term in METRIC_TERMS:
        term_n = normalize_text(term).replace("-", " ")
        if term_n in text_l.replace("-", " "):
            found.add(term_n)
    candidate = normalize_metric_phrase(text_l)
    if candidate and len(candidate.split()) >= 2:
        # Keep a compact candidate; long sentences become noisy otherwise.
        toks = candidate.split()
        if len(toks) <= 7:
            found.add(candidate)
    return sorted(found)


def extract_metric_candidates_from_header(row_header: Any, col_header: Any) -> List[str]:
    row_h = normalize_text(row_header)
    col_h = normalize_text(col_header)
    candidates = set()

    row_metric = normalize_metric_phrase(row_h)
    col_metric = normalize_metric_phrase(col_h)

    # Common FinQA patterns:
    #   row = 2018, col = aircraft fuel expense -> metric from column
    #   row = available-for-sale investments, col = Dec 2012 -> metric from row
    #   row = granted, col = weighted average fair value -> combined row+col can be useful
    if col_metric and not is_period_like(col_h):
        candidates.add(col_metric)
    if row_metric and not is_period_like(row_h):
        candidates.add(row_metric)
    if row_metric and col_metric and row_metric != col_metric:
        if not is_period_like(row_h) and not is_period_like(col_h):
            candidates.add(f"{row_metric} {col_metric}".strip())

    # Add known metric matches from raw headers.
    for m in extract_metrics(f"{row_h} {col_h}"):
        candidates.add(m)

    return sorted(c for c in candidates if c)


def extract_metric_candidates_from_question(question: Any) -> List[str]:
    q = normalize_text(question)
    candidates = set(extract_metrics(q))
    normalized = normalize_metric_phrase(q)
    if normalized:
        # Remove formula words that should become operations, not metrics.
        toks = [
            t for t in normalized.split()
            if t not in {"what", "percentage", "percent", "growth", "rate", "change", "difference", "increase", "decrease", "larger", "greater", "exceed", "total"}
        ]
        if 1 <= len(toks) <= 8:
            candidates.add(" ".join(toks))
    return sorted(candidates)


def linearize_table_row(table: List[List[str]], row_idx: int) -> str:
    if not table or row_idx >= len(table):
        return ""
    headers = table[0]
    row = table[row_idx]
    if row_idx == 0:
        return " ; ".join(str(h) for h in headers)
    row_name = row[0] if row else ""
    pieces = []
    for col_idx in range(1, min(len(headers), len(row))):
        header = headers[col_idx]
        value = row[col_idx]
        if str(value).strip():
            pieces.append(f"the {row_name} of {header} is {value}")
    return " ; ".join(pieces) + (" ;" if pieces else "")


def build_evidence_candidates(example: Dict[str, Any]) -> Dict[str, str]:
    candidates: Dict[str, str] = {}
    for i, sent in enumerate(get_all_text(example)):
        candidates[f"text_{i}"] = normalize_text(sent)
    table = example.get("table", []) or []
    for i in range(len(table)):
        candidates[f"table_{i}"] = normalize_text(linearize_table_row(table, i))
    return candidates


def get_gold_evidence_ids(example: Dict[str, Any]) -> set[str]:
    return set(example.get("qa", {}).get("gold_inds", {}).keys())


def extract_formula_cues(question: Any) -> List[str]:
    q = normalize_text(question)
    cues = []
    if any(x in q for x in ["growth rate", "percentage change", "percent change"]):
        cues.append("growth_rate")
    if any(x in q for x in ["what percentage", "what percent", "comprised of", "as a percentage of"]):
        cues.append("part_to_whole_ratio")
    if any(x in q for x in ["total operating expenses", "total expenses"]) and any(x in q for x in ["percent", "percentage"]):
        cues.append("inverse_percent_total")
    if any(x in q for x in ["average", "per "]):
        cues.append("average_or_ratio")
    if any(x in q for x in ["change", "increase", "decrease", "difference"]):
        cues.append("difference")
    if any(x in q for x in ["exceed", "greater", "larger", "less", "smaller", "did "]):
        cues.append("comparison")
    if any(x in q for x in ["total", "sum"]):
        cues.append("sum_or_total")
    if any(x in q for x in ["annual amortization", "amortization expenses", "amortization expense"]):
        cues.append("annual_amortization")
    return sorted(set(cues))


def extract_program_ops(program: Any) -> List[str]:
    p = str(program or "")
    return re.findall(r"([a-zA-Z_]+)\s*\(", p)


def token_set(text: Any) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", normalize_text(text)))


def jaccard(a: Sequence[str] | set[str], b: Sequence[str] | set[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

import re

DATE_MONTHS = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}


def is_probably_date_number(raw_num: str, context: str, window: int = 24) -> bool:
    """
    Returns True if raw_num is likely part of a date expression, not a financial value.

    Examples:
      "march 31, 2012" -> 31 and 2012 are date numbers
      "december 31 2017" -> 31 and 2017 are date numbers
    """
    if raw_num is None or context is None:
        return False

    raw = str(raw_num).strip().replace("$", "").replace(",", "")
    ctx = str(context).lower()

    # Percentages and currency values are usually not date numbers.
    if "%" in str(raw_num) or "$" in str(raw_num):
        return False

    # Year-like number alone can be period, but not an operand.
    if re.fullmatch(r"(19|20)\d{2}", raw):
        return True

    # Day number near month name.
    try:
        value = int(float(raw))
    except Exception:
        return False

    if not (1 <= value <= 31):
        return False

    idx = ctx.find(str(int(value)))
    if idx < 0:
        return False

    left = max(0, idx - window)
    right = min(len(ctx), idx + window)
    local = ctx[left:right]

    return any(month in local for month in DATE_MONTHS)
