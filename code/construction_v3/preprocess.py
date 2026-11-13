from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
FULL_DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan\.?|feb\.?|mar\.?|apr\.?|jun\.?|jul\.?|aug\.?|sep\.?|sept\.?|oct\.?|nov\.?|dec\.?)\s+"
    r"(\d{1,2})\s*,?\s*((?:19|20)\d{2})\b",
    flags=re.I,
)
NUMBER_RE = re.compile(
    r"(?P<negparen>\()?\s*(?P<currency>\$)?\s*(?P<sign>-)?\s*\d+(?:,\d{3})*(?:\.\d+)?\s*(?P<pct>%)?\s*(?(negparen)\))"
)

MONTH_TO_NUM = {
    "january": 1, "jan": 1, "jan.": 1,
    "february": 2, "feb": 2, "feb.": 2,
    "march": 3, "mar": 3, "mar.": 3,
    "april": 4, "apr": 4, "apr.": 4,
    "may": 5,
    "june": 6, "jun": 6, "jun.": 6,
    "july": 7, "jul": 7, "jul.": 7,
    "august": 8, "aug": 8, "aug.": 8,
    "september": 9, "sep": 9, "sep.": 9, "sept": 9, "sept.": 9,
    "october": 10, "oct": 10, "oct.": 10,
    "november": 11, "nov": 11, "nov.": 11,
    "december": 12, "dec": 12, "dec.": 12,
}

SCALE_MULTIPLIER = {
    "thousand": 1_000.0,
    "million": 1_000_000.0,
    "billion": 1_000_000_000.0,
}

GENERIC_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "for", "and", "or", "as", "at", "by", "from", "with", "on",
    "what", "was", "were", "is", "are", "be", "been", "being", "did", "does", "do", "how", "much", "many",
}

GENERIC_HEADER_TERMS = {
    "amount", "in", "millions", "million", "thousands", "thousand", "billions", "billion", "dollars", "dollar",
    "year", "years", "date", "dec", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov",
}

COMMON_METRIC_TERMS = [
    "interest expense", "annual interest expense", "net revenue", "revenue", "growth rate",
    "total operating expenses", "operating expenses", "aircraft fuel expense", "fuel expense",
    "available-for-sale investments", "total cash and investments", "cash and investments", "cash",
    "fair value", "aggregate cost", "level 2", "level 3", "shares", "number of shares",
    "weighted average grant date fair value", "grant date fair value", "stock-based compensation expense",
    "stock based compensation expense", "compensation expense", "intangible assets", "customer-related intangibles",
    "network location intangibles", "amortization expense", "loans held-for-sale", "deferred acquisition payments",
    "contingent acquisition payments", "net interest income", "managed basis", "dividend", "dividends",
    "bond authorized", "bond authorization", "withdrawn", "facilities", "leased", "owned",
]


def normalize_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    # Repair common FinQA table OCR joins.
    repairs = {
        "priceper": "price per",
        "fuelexpense": "fuel expense",
        "totaloperating": "total operating",
        "averageprice": "average price",
        "weighted-average": "weighted average",
    }
    s = s.lower()
    for a, b in repairs.items():
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s).strip()


def token_set(text: Any) -> Set[str]:
    text = normalize_text(text)
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)
        if len(t) > 1 and t not in GENERIC_STOPWORDS
    }


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def get_all_text(example: Dict[str, Any]) -> List[str]:
    return list(example.get("pre_text", []) or []) + list(example.get("post_text", []) or [])


def get_pre_text_count(example: Dict[str, Any]) -> int:
    return len(example.get("pre_text", []) or [])


def detect_scale(text: Any) -> Optional[str]:
    t = normalize_text(text)
    if re.search(r"\b(in\s+)?billions?\b", t):
        return "billion"
    if re.search(r"\b(in\s+)?millions?\b", t):
        return "million"
    if re.search(r"\b(in\s+)?thousands?\b", t):
        return "thousand"
    return None


def detect_currency(text: Any) -> Optional[str]:
    t = normalize_text(text)
    if "$" in str(text) or "dollar" in t or "dollars" in t:
        return "USD"
    return None


def detect_units(text: Any) -> List[str]:
    t = normalize_text(text)
    units = []
    for unit, patterns in {
        "thousand": ["thousand", "thousands", "in thousands"],
        "million": ["million", "millions", "in millions"],
        "billion": ["billion", "billions", "in billions"],
        "percent": ["%", "percent", "percentage"],
        "basis_points": ["basis points", "bps"],
        "dollar": ["$", "dollar", "dollars"],
        "shares": ["shares", "share"],
        "per_share": ["per share"],
        "years": ["year", "years"],
    }.items():
        if any(p in t or p in str(text) for p in patterns):
            units.append(unit)
    return sorted(set(units))


def infer_table_scale(example: Dict[str, Any]) -> Optional[str]:
    candidates = []
    table = example.get("table", []) or []
    table_ori = example.get("table_ori", []) or []
    if table:
        candidates.append(" ".join(str(x) for x in table[0]))
    if table_ori:
        candidates.append(" ".join(str(x) for x in table_ori[0]))
    pre = example.get("pre_text", []) or []
    post = example.get("post_text", []) or []
    candidates.extend(pre[-4:])
    candidates.extend(post[:3])
    for text in candidates:
        scale = detect_scale(text)
        if scale:
            return scale
    return None


def extract_periods(text: Any) -> List[Dict[str, Any]]:
    t = normalize_text(text)
    periods: List[Dict[str, Any]] = []
    seen = set()

    for m in FULL_DATE_RE.finditer(t):
        mon_raw, day, year = m.group(1).lower().rstrip("."), int(m.group(2)), int(m.group(3))
        mon = MONTH_TO_NUM.get(mon_raw, None)
        if mon:
            val = f"{year:04d}-{mon:02d}-{day:02d}"
            key = ("date", val)
            if key not in seen:
                periods.append({"type": "date", "value": val, "year": year, "raw": m.group(0)})
                seen.add(key)

    for m in YEAR_RE.finditer(t):
        year = int(m.group(0))
        # If already represented inside a full date, still keep year only when text is table header/row short.
        key = ("year", str(year))
        if key not in seen:
            periods.append({"type": "year", "value": year, "year": year, "raw": str(year)})
            seen.add(key)

    return periods


def is_period_like(text: Any) -> bool:
    t = normalize_text(text)
    if re.fullmatch(r"(19|20)\d{2}", t):
        return True
    if FULL_DATE_RE.search(t):
        return True
    if YEAR_RE.search(t) and len(t.split()) <= 4:
        return True
    return False


def parse_financial_number(raw: str, context: str = "", inherited_scale: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    raw_s = str(raw).strip()
    if not raw_s:
        return None

    is_percent = "%" in raw_s
    currency = "USD" if "$" in raw_s or detect_currency(context) else None
    scale = detect_scale(context) or inherited_scale

    neg_paren = raw_s.strip().startswith("(") and raw_s.strip().endswith(")")
    sign = "negative_parentheses" if neg_paren else "negative" if "-" in raw_s else "positive"

    s = raw_s.replace("$", "").replace(",", "").replace("%", "")
    s = s.replace("(", "").replace(")", "").strip()
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        value = float(m.group(0))
    except Exception:
        return None
    if neg_paren and value > 0:
        value = -value

    normalized = value / 100.0 if is_percent else value
    display_value = abs(value) if neg_paren else value
    unit = "percent" if is_percent else "dollar" if currency else None

    return {
        "raw": raw_s,
        "value": value,
        "display_value": display_value,
        "normalized_value": normalized,
        "is_percent": is_percent,
        "currency": currency,
        "scale": scale,
        "unit": unit,
        "sign": sign,
    }


def get_financial_numbers(text: Any, context: Optional[str] = None, inherited_scale: Optional[str] = None) -> List[Dict[str, Any]]:
    context = normalize_text(context if context is not None else text)
    out = []
    for m in NUMBER_RE.finditer(str(text)):
        raw = m.group(0).strip()
        parsed = parse_financial_number(raw, context=context, inherited_scale=inherited_scale)
        if parsed:
            out.append(parsed)
    return out


def normalize_metric_phrase(text: Any) -> str:
    t = normalize_text(text)
    # remove numbers/dates/currency/units, keep financial modifiers.
    t = FULL_DATE_RE.sub(" ", t)
    t = YEAR_RE.sub(" ", t)
    t = NUMBER_RE.sub(" ", t)
    t = t.replace("$", " ").replace("%", " percent ")
    t = re.sub(r"[^a-zA-Z\- ]+", " ", t)
    toks = [x for x in t.split() if x not in GENERIC_STOPWORDS and x not in GENERIC_HEADER_TERMS]
    return " ".join(toks).strip()


def extract_metrics(text: Any) -> List[str]:
    t = normalize_text(text)
    found = []
    for term in COMMON_METRIC_TERMS:
        if term in t:
            found.append(normalize_metric_phrase(term))
    # Conservative fallback: only short-ish phrases with financial cue words.
    cue_words = {"revenue", "income", "expense", "expenses", "assets", "liabilities", "cash", "debt", "shares", "dividend", "sales", "cost", "value", "fair", "fuel", "interest", "lease", "leased", "facilities", "bond"}
    phrase = normalize_metric_phrase(t)
    if phrase:
        toks = set(phrase.split())
        if toks & cue_words and len(phrase.split()) <= 10:
            found.append(phrase)
    return sorted(set(x for x in found if x))


def extract_metric_candidates_from_header(row_header: Any, col_header: Any) -> List[str]:
    row = normalize_metric_phrase(row_header)
    col = normalize_metric_phrase(col_header)
    candidates = []
    if row:
        candidates.append(row)
    if col:
        candidates.append(col)
    if row and col:
        candidates.append(normalize_metric_phrase(f"{row} {col}"))
    out = []
    for c in candidates:
        if c and c not in out:
            out.append(c)
    return out


def extract_metric_candidates_from_question(question: Any) -> List[str]:
    q = normalize_text(question)
    candidates = set(extract_metrics(q))
    # remove formula cue words but keep metric nouns
    cleaned = q
    for pat in [
        "what percentage", "what percent", "percentage decrease", "percentage increase", "percent change",
        "growth rate", "difference", "increase", "decrease", "average", "total", "sum", "what is", "what was", "how much", "how many",
        "from", "between", "during", "in millions", "in thousands", "in billions",
    ]:
        cleaned = cleaned.replace(pat, " ")
    phrase = normalize_metric_phrase(cleaned)
    if phrase:
        # Split around common prepositions left in phrase by normalization.
        candidates.add(phrase)
    return sorted(candidates, key=lambda x: (-len(x.split()), x))


def extract_formula_cues(question: Any) -> List[str]:
    q = normalize_text(question)
    cues = []
    if any(x in q for x in ["growth rate", "percentage change", "percent change"]):
        cues.append("growth_rate")
    if any(x in q for x in ["what percentage", "what percent", "comprised of", "portion of", "as part of"]):
        cues.append("part_to_whole_ratio")
    if any(x in q for x in ["average", "per "]):
        cues.append("average_or_ratio")
    if any(x in q for x in ["change", "increase", "decrease", "difference", "lost from"]):
        cues.append("difference")
    if any(x in q for x in ["exceed", "greater", "larger", "less than", "did "]):
        cues.append("comparison")
    if any(x in q for x in ["total", "sum"]):
        cues.append("sum_or_total")
    if any(x in q for x in ["annual amortization", "amortization expenses"]):
        cues.append("annual_amortization")
    if any(x in q for x in ["percent of total", "as part of"]):
        cues.append("inverse_percent_total")
    return sorted(set(cues))


def linearize_table_row(table: List[List[str]], row_idx: int) -> str:
    if not table or row_idx >= len(table):
        return ""
    headers = table[0]
    row = table[row_idx]
    if row_idx == 0:
        return " ; ".join(str(x) for x in headers)
    row_name = row[0] if row else ""
    pieces = []
    for col_idx in range(1, min(len(headers), len(row))):
        header = headers[col_idx]
        value = row[col_idx]
        if str(value).strip():
            pieces.append(f"the {row_name} of {header} is {value}")
    return " ; ".join(pieces) + " ;"


def build_evidence_candidates(example: Dict[str, Any]) -> Dict[str, str]:
    cands = {}
    for i, sent in enumerate(get_all_text(example)):
        cands[f"text_{i}"] = normalize_text(sent)
    table = example.get("table", []) or []
    for i in range(len(table)):
        cands[f"table_{i}"] = normalize_text(linearize_table_row(table, i))
    return cands


def get_gold_evidence_ids(example: Dict[str, Any]) -> set[str]:
    return set((example.get("qa", {}) or {}).get("gold_inds", {}).keys())
