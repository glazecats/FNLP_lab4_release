from __future__ import annotations

import re


FINAL_PATTERNS = [
    re.compile(r"FINAL_ANSWER\s*[:：]\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"最终答案\s*[:：]\s*([^\n\r]+)"),
    re.compile(r"答案\s*[:：]\s*([^\n\r]+)"),
]

SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")

NUMBER_RE = r"[-+]?\d+(?:\.\d+)?(?:\s*(?:e[+-]?\d+|\\times\s*10\s*\^\s*\{?\s*[+-]?\d+\s*\}?))?"


def normalize_unicode_math(text: str) -> str:
    value = text.strip().translate(SUPERSCRIPT_DIGITS)
    value = value.replace("×", r"\times")
    value = value.replace("−", "-")
    value = re.sub(
        r"([-+]?\d+(?:\.\d+)?)\s*\\times\s*10\s*(?:\^\s*\{?\s*([-+]?\d+)\s*\}?|([+-]?\d+))",
        lambda match: f"{match.group(1)}e{match.group(2) or match.group(3)}",
        value,
    )
    return value


def _extract_balanced_braces(text: str, start: int) -> str | None:
    depth = 0
    begin = None
    for i in range(start, len(text)):
        if text[i] == "{":
            if depth == 0:
                begin = i + 1
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0 and begin is not None:
                return text[begin:i]
    return None


def _boxed_value(text: str) -> str | None:
    for marker in ("\\boxed", "\\fbox"):
        idx = text.rfind(marker)
        if idx >= 0:
            brace_idx = text.find("{", idx)
            if brace_idx >= 0:
                return _extract_balanced_braces(text, brace_idx)
    return None


def _has_single_wrapping_pair(value: str, open_ch: str, close_ch: str) -> bool:
    if not (value.startswith(open_ch) and value.endswith(close_ch)):
        return False
    depth = 0
    for i, ch in enumerate(value):
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0 and i != len(value) - 1:
                return False
    return depth == 0


def _format_decimal(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value - round(value)) < 1e-12 and abs(value) < 1e12:
        return str(int(round(value)))
    return f"{value:.12g}"


def _decimalize_simple_fraction(value: str) -> str:
    latex_match = re.fullmatch(
        r"([+-]?)\\frac\{\s*([+-]?\d+(?:\.\d+)?)\s*\}\{\s*([+-]?\d+(?:\.\d+)?)\s*\}",
        value,
    )
    if latex_match:
        sign, numerator, denominator = latex_match.groups()
        denom = float(denominator)
        if denom != 0:
            number = float(numerator) / denom
            if sign == "-":
                number = -number
            return _format_decimal(number)

    slash_match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*/\s*([+-]?\d+(?:\.\d+)?)", value)
    if slash_match:
        numerator, denominator = slash_match.groups()
        denom = float(denominator)
        if denom != 0:
            return _format_decimal(float(numerator) / denom)

    return value


def _target_unit_aliases(target_unit: str | None) -> list[str]:
    if not target_unit or "10" in target_unit:
        return []
    unit = target_unit.replace("~", " ")
    unit = re.sub(r"\\(?:mathrm|text)\{([^{}]+)\}", r"\1", unit)
    unit = unit.replace("{", " ").replace("}", " ").replace("$", " ")
    unit = re.sub(r"\s+", " ", unit).strip()
    compound_aliases: list[str] = []
    if re.search(r"kg\s*m\s*/\s*s|kg\s*m\s*s\^-?1", unit, flags=re.I):
        compound_aliases.append(r"kg\s*m\s*/\s*s")
    if re.search(r"\bm\s*/\s*s\b|\bm\s*s\^-?1\b", unit, flags=re.I):
        compound_aliases.append(r"m\s*/\s*s")
    if compound_aliases:
        return compound_aliases

    aliases: list[str] = []
    known_units = [
        "kJ",
        "MeV",
        "eV",
        "mm",
        "cm",
        "pm",
        "nm",
        "kg",
        "mol",
        "atm",
        "Pa",
        "Hz",
        "J",
        "K",
        "V",
        "A",
        "C",
        "N",
        "m",
        "%",
    ]
    for alias in known_units:
        if re.search(rf"(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])", unit, flags=re.I):
            aliases.append(re.escape(alias))
    return aliases


def _is_plain_numeric(value: str) -> bool:
    cleaned = clean_answer(value)
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", cleaned, flags=re.I))


def _extract_by_target_unit(response: str, target_unit: str | None) -> str | None:
    aliases = _target_unit_aliases(target_unit)
    if not aliases:
        return None
    matches: list[str] = []
    for alias in aliases:
        pattern = re.compile(
            rf"({NUMBER_RE})\s*(?:\(?\s*)?(?<![A-Za-z\\])(?:{alias})(?![A-Za-z])",
            flags=re.I,
        )
        matches.extend(match.group(1) for match in pattern.finditer(response))
    if not matches:
        return None
    return clean_answer(matches[-1])


def _has_multiple_numeric_tokens(value: str) -> bool:
    normalized = normalize_unicode_math(value).splitlines()[0]
    tokens = normalized.split()
    return (
        len(tokens) > 1
        and "\\" not in normalized
        and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", token, flags=re.I) for token in tokens)
    )


def clean_answer(answer: str) -> str:
    value = normalize_unicode_math(answer)
    value = value.splitlines()[0].strip()
    value = re.sub(r"^\s*\\text\{\s*", "", value).strip()
    value = re.sub(r"^\s*\}+\s*", "", value).strip()
    value = value.strip("*_ ")
    value = re.sub(r"^\$+|\$+$", "", value).strip()
    value = re.sub(r"^<[^>\s]+>\s*|\s*</[^>\s]+>$", "", value).strip()
    if _has_single_wrapping_pair(value, "(", ")") or _has_single_wrapping_pair(value, "[", "]"):
        value = value[1:-1].strip()
    value = re.sub(
        r"^(?:the\s+answer\s+is|answer\s+is|答案是|最终答案为|最终答案)\s*",
        "",
        value,
        flags=re.I,
    ).strip()
    value = value.strip("*_ ")
    value = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*(?:\([^)]*\))?\s*(?:=|\\approx|≈)\s*", "", value).strip()
    value = re.sub(r"^n\s*=\s*", "", value, flags=re.I).strip()
    value = re.sub(r"sqrt\(([^()]+)\)", r"\\sqrt{\1}", value)
    value = re.sub(r"([()0-9.]+)\s*\^\s*\(([^()]+)\)", r"\1^{\2}", value)
    value = re.sub(r"\(([^()]+)\)\^\{([^{}]+)\}", r"\1^{\2}", value)
    value = re.sub(r"\be\^([+-]?\d+(?:\.\d+)?)\b", r"e^{\1}", value)
    value = re.sub(r"^([+-]?)e([+-]?\d+(?:\.\d+)?)$", r"\1e^{\2}", value)
    value = re.split(r"[≈≃]", value)[-1].strip()
    value = re.sub(r"\\(?:mathrm|text)\{[^{}]*\}\s*$", "", value).strip()
    value = re.sub(
        r"\s*(?:m/s|kg m/s|kg|m|s|nm|eV|MeV|J|K|mol|L|M|atm|Pa|Hz|W|V|A|C|N|rad|degree|degrees|°)\s*$",
        "",
        value,
    ).strip()
    numeric_tokens = value.split()
    if (
        len(numeric_tokens) > 1
        and "\\" not in value
        and all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", token, flags=re.I) for token in numeric_tokens)
    ):
        value = numeric_tokens[-1]
    value = value.strip("。.;,，：:")
    value = _decimalize_simple_fraction(value)
    return value or "0"


def extract_answer(response: str, target_unit: str | None = None) -> str:
    for pattern in FINAL_PATTERNS:
        matches = list(pattern.finditer(response))
        if matches:
            raw_answer = matches[-1].group(1)
            if _has_multiple_numeric_tokens(raw_answer):
                unit_answer = _extract_by_target_unit(response, target_unit)
                if unit_answer:
                    return unit_answer
            if target_unit and not _is_plain_numeric(raw_answer):
                unit_answer = _extract_by_target_unit(response, target_unit)
                if unit_answer:
                    return unit_answer
            return clean_answer(raw_answer)

    if target_unit:
        unit_answer = _extract_by_target_unit(response, target_unit)
        if unit_answer:
            return unit_answer

    boxed = _boxed_value(response)
    if boxed:
        return clean_answer(boxed)

    nonempty = [line.strip() for line in response.splitlines() if line.strip()]
    if not nonempty:
        return "0"
    return clean_answer(nonempty[-1])
