from __future__ import annotations

import re

from .calculator import format_number


FINAL_RE = re.compile(r"FINAL_ANSWER\s*[:：]\s*(.+)", re.IGNORECASE)
TOOL_RE = re.compile(r"TOOL_CALC\s*[:：]\s*(.+)", re.IGNORECASE)


def extract_tool_expression(text: str) -> str | None:
    match = TOOL_RE.search(text)
    if not match:
        return None
    expr = match.group(1).strip()
    expr = expr.splitlines()[0].strip()
    return expr or None


def extract_final_answer(text: str) -> str | None:
    matches = FINAL_RE.findall(text)
    if not matches:
        return None
    return normalize_answer(matches[-1])


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    answer = answer.replace("\u3000", " ").replace("\xa0", " ")
    answer = answer.strip("`*_ \t\r\n")
    if answer.startswith("$") and answer.endswith("$") and len(answer) >= 2:
        answer = answer[1:-1].strip()
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.rstrip(".。;,，")
    numeric_percent = _percent_to_decimal(answer)
    if numeric_percent is not None:
        return numeric_percent
    numeric_fraction = _fraction_to_decimal(answer)
    if numeric_fraction is not None:
        return numeric_fraction
    return answer.strip()


def _percent_to_decimal(answer: str) -> str | None:
    text = answer.strip().replace(r"\%", "%")
    match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*%", text)
    if not match:
        return None
    return format_number(float(match.group(1)) / 100)


def _fraction_to_decimal(answer: str) -> str | None:
    text = answer.strip()
    latex = re.fullmatch(r"([-+]?)\s*\\frac\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}\s*\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", text)
    simple = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)", text)
    if latex:
        sign = -1.0 if latex.group(1) == "-" else 1.0
        numerator = sign * float(latex.group(2))
        denominator = float(latex.group(3))
    elif simple:
        numerator = float(simple.group(1))
        denominator = float(simple.group(2))
    else:
        return None
    if denominator == 0:
        return None
    return format_number(numerator / denominator)


def equivalent_answer_text(left: str | None, right: str | None) -> bool:
    if not left or not right:
        return False
    return _compact(left) == _compact(right)


def _compact(value: str) -> str:
    value = normalize_answer(value).lower()
    value = value.replace("\\mathrm", "")
    return re.sub(r"[\s{}$]", "", value)


def looks_invalid_answer(answer: str) -> str | None:
    if not answer:
        return "empty answer"
    lowered = answer.lower()
    bad_phrases = [
        "the answer",
        "答案",
        "无法",
        "cannot",
        "unknown",
        "not enough",
        "缺少",
        "therefore",
        "because",
    ]
    for phrase in bad_phrases:
        if phrase in lowered:
            return f"contains explanatory phrase: {phrase}"
    if "=" in answer and not answer.strip().startswith("\\"):
        return "contains assignment/equation"
    unit_patterns = [
        r"\b(m|s|kg|j|n|pa|hz|ev|mol|k|v|a|w|c|nm|pm|cm|mm|rad|deg|l|atm)\b",
        r"\\mathrm",
        r"\\text",
        r"\^\s*\\circ",
    ]
    for pattern in unit_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            return "appears to contain a unit"
    if _contains_unresolved_variable(answer):
        return "contains unresolved variable"
    return None


def _contains_unresolved_variable(answer: str) -> bool:
    text = normalize_answer(answer)
    try:
        float(text)
        return False
    except ValueError:
        pass
    text = re.sub(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", " ", text)
    text = re.sub(r"\\[A-Za-z]+", " ", text)
    allowed_names = [
        "kB_eV",
        "e_charge",
        "epsilon0",
        "hbar",
        "k_B",
        "N_A",
        "m_e",
        "m_p",
        "m_n",
        "amu",
        "kB",
        "pi",
        "mu0",
        "NA",
        "c",
        "e",
        "h",
        "R",
        "F",
        "G",
        "g",
        "u",
    ]
    for name in allowed_names:
        text = re.sub(rf"\b{re.escape(name)}\b", " ", text)
    return bool(re.search(r"[A-Za-z]", text))
