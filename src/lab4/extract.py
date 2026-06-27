from __future__ import annotations

import re


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
    return answer.strip()


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
    return None

