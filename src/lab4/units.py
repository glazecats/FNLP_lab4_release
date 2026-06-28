from __future__ import annotations

import math
import re


NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?$", re.IGNORECASE)

TARGET_UNIT_PATTERNS = [
    re.compile(r"（单位[:：]\s*([^）]+)）"),
    re.compile(r"\(单位[:：]\s*([^)]+)\)"),
    re.compile(r"单位为\s*([^。；，,\n]+)"),
    re.compile(r"以\s*([^，。；\n]+?)\s*为单位"),
    re.compile(r"为\s*X\s*[*x]\s*(10\s*\^\s*[-+]?\d+\s*[^，。；\n]*)", re.IGNORECASE),
]


def infer_target_unit(question_text: str, explicit_unit: str | None = None) -> str | None:
    """Return an explicitly stated target unit/form from metadata or the question text."""

    if explicit_unit:
        return explicit_unit.strip()
    if re.search(r"百分之多少|百分比|percentage|what\s+percent", question_text, flags=re.IGNORECASE):
        return "%"
    for pattern in TARGET_UNIT_PATTERNS:
        match = pattern.search(question_text)
        if match:
            unit = match.group(1).strip()
            tail = question_text[match.end() :]
            if "kg" in unit and re.search(r"次数|发生次数|fissions", tail, flags=re.IGNORECASE):
                continue
            return unit
    return None


def _unit_power_of_ten(unit: str | None) -> int | None:
    if not unit:
        return None
    patterns = [
        r"10\s*\^\s*\{\s*([+-]?\d+)\s*\}",
        r"10\s*\^\s*([+-]?\d+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, unit)
        if match:
            return int(match.group(1))
    return None


def _format_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value - round(value)) < 1e-10 and abs(value) < 1e12:
        return str(int(round(value)))
    return f"{value:.10g}"


def _parse_numeric_answer(answer: str) -> tuple[float, bool] | tuple[None, bool]:
    value = answer.strip()
    scientific_like = bool(re.search(r"e[+-]?\d+|\\times\s*10|10\s*\^", value, flags=re.I))
    value = value.replace("−", "-")
    value = value.replace(" ", "")
    value = re.sub(
        r"^([-+]?\d+(?:\.\d+)?)\\times10\^\{?([+-]?\d+)\}?$",
        r"\1e\2",
        value,
    )
    value = re.sub(r"^10\^\{?([+-]?\d+)\}?$", r"1e\1", value)
    if NUMERIC_RE.fullmatch(value):
        return float(value), scientific_like
    return None, scientific_like


def _prefix_scale(unit: str | None) -> float | None:
    if not unit:
        return None
    normalized = unit.replace(" ", "")
    if r"\mu" in normalized or "μ" in normalized:
        return 1e-6
    if "pm" in normalized:
        return 1e-12
    if "nm" in normalized:
        return 1e-9
    if "mJ" in normalized:
        return 1e-3
    return None


def normalize_for_unit(answer: str, unit: str | None, question_text: str | None = None) -> str:
    """Convert obvious absolute numeric answers to coefficients for 10^n units.

    Example: answer=4.74e14, unit=10^14 Hz -> 4.74.
    If the answer already looks like a coefficient, leave it unchanged.
    """

    value = answer.strip()
    number, scientific_like = _parse_numeric_answer(value)
    exponent = _unit_power_of_ten(unit)
    if number is None or number == 0:
        return answer

    if exponent is not None:
        scale = 10.0**exponent
        coefficient = number / scale
        looks_absolute = (
            (exponent >= 0 and abs(number) >= abs(scale) * 1e-2)
            or (exponent < 0 and abs(number) <= abs(scale) * 1e4)
            or scientific_like
        )
        if looks_absolute and 1e-3 <= abs(coefficient) <= 1e3:
            return _format_number(coefficient)

    prefix_scale = _prefix_scale(unit)
    prefix_looks_absolute = scientific_like or (prefix_scale == 1e-3 and "mJ" in (unit or "") and abs(number) < 1)
    if prefix_scale is not None and prefix_looks_absolute:
        coefficient = number / prefix_scale
        if 1e-6 <= abs(coefficient) <= 1e6:
            return _format_number(coefficient)

    if unit == "%" and 0 < abs(number) <= 1:
        return _format_number(number * 100)

    if question_text and number < 0:
        if re.search(r"高度是多少|高多少|height", question_text, flags=re.IGNORECASE):
            return _format_number(abs(number))

    if question_text and unit and "J" in unit and "mol" not in unit and 0 < abs(number) < 1e-15:
        if re.search(r"一摩尔|1\.?0*\s*mol|one\s+mole", question_text, flags=re.IGNORECASE):
            return _format_number(number * 6.02214076e23)

    return answer
