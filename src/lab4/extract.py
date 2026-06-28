from __future__ import annotations

import re


FINAL_PATTERNS = [
    re.compile(r"FINAL_ANSWER\s*[:：]\s*([^\n\r]+)", re.IGNORECASE),
    re.compile(r"最终答案\s*[:：]\s*([^\n\r]+)"),
    re.compile(r"答案\s*[:：]\s*([^\n\r]+)"),
]

SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")


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


def clean_answer(answer: str) -> str:
    value = normalize_unicode_math(answer)
    value = value.splitlines()[0].strip()
    value = re.sub(r"^\s*\\text\{\s*", "", value).strip()
    value = re.sub(r"^\s*\}+\s*", "", value).strip()
    value = value.strip("*_ ")
    value = re.sub(r"^\$+|\$+$", "", value).strip()
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


def extract_answer(response: str) -> str:
    for pattern in FINAL_PATTERNS:
        matches = list(pattern.finditer(response))
        if matches:
            return clean_answer(matches[-1].group(1))

    boxed = _boxed_value(response)
    if boxed:
        return clean_answer(boxed)

    nonempty = [line.strip() for line in response.splitlines() if line.strip()]
    if not nonempty:
        return "0"
    return clean_answer(nonempty[-1])
