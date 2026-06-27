from __future__ import annotations

import ast
import math
import re


CALC_EXPR_RE = re.compile(r"CALC_EXPR\s*[:：]\s*([^\n\r]+)", re.IGNORECASE)
DEGREE_RE = re.compile(r"(?<![A-Za-z_])([-+]?\d+(?:\.\d+)?)\s*°")

ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "asin_deg": lambda x: math.degrees(math.asin(x)),
    "acos_deg": lambda x: math.degrees(math.acos(x)),
    "atan_deg": lambda x: math.degrees(math.atan(x)),
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "radians": math.radians,
    "degrees": math.degrees,
    "fabs": math.fabs,
    "floor": math.floor,
    "ceil": math.ceil,
    "pow": pow,
    "abs": abs,
}


class CalculatorError(ValueError):
    pass


def extract_calc_expr(response: str) -> str | None:
    matches = list(CALC_EXPR_RE.finditer(response))
    if not matches:
        return None
    expr = matches[-1].group(1).strip()
    return expr or None


def looks_calculable(answer: str) -> bool:
    value = answer.strip()
    if not value or "\\" in value:
        return False
    return bool(
        re.search(r"\b(?:sin|cos|tan|asin|acos|atan|sqrt|log|ln|exp)\b", value, flags=re.I)
        or any(op in value for op in ("*", "/", "^", "°", "(", ")"))
    )


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        if isinstance(node.op, ast.Mod):
            return left % right
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
    if isinstance(node, ast.Name):
        if node.id in ALLOWED_NAMES and isinstance(ALLOWED_NAMES[node.id], (int, float)):
            return float(ALLOWED_NAMES[node.id])
        raise CalculatorError(f"Name is not a numeric constant: {node.id}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = ALLOWED_NAMES.get(node.func.id)
        if not callable(fn):
            raise CalculatorError(f"Function is not allowed: {node.func.id}")
        if node.keywords:
            raise CalculatorError("Keyword arguments are not allowed")
        args = [_eval_node(arg) for arg in node.args]
        return float(fn(*args))
    raise CalculatorError(f"Unsupported expression: {ast.dump(node, include_attributes=False)}")


def evaluate_expression(expr: str) -> float:
    normalized = normalize_expression(expr)
    tree = ast.parse(normalized, mode="eval")
    return _eval_node(tree)


def normalize_expression(expr: str) -> str:
    normalized = expr.strip()
    normalized = normalized.strip("$ ")
    normalized = normalized.replace("−", "-")
    normalized = normalized.replace("^", "**")
    normalized = normalized.replace("×", "*")
    normalized = re.sub(r"\\?ln\s*\(", "log(", normalized, flags=re.I)
    normalized = re.sub(r"\\?sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = re.sub(r"\\?exp\s*\{([^{}]+)\}", r"exp(\1)", normalized)
    normalized = re.sub(r"^e\s*\*\*\s*\{([^{}]+)\}$", r"exp(\1)", normalized)
    normalized = re.sub(r"^e\s*\*\*\s*\(([^()]+)\)$", r"exp(\1)", normalized)
    normalized = re.sub(r"^e\s*\*\*\s*([-+]?\d+(?:\.\d+)?)$", r"exp(\1)", normalized)

    # Common model notation for inverse trig in degree-valued school physics answers.
    normalized = re.sub(r"\bsin\s*-\s*1\s*\(", "asin_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\bcos\s*-\s*1\s*\(", "acos_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\btan\s*-\s*1\s*\(", "atan_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\bsin\s*\*\*\s*-1\s*\(", "asin_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\bcos\s*\*\*\s*-1\s*\(", "acos_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\btan\s*\*\*\s*-1\s*\(", "atan_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\barcsin\s*\(", "asin_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\barccos\s*\(", "acos_deg(", normalized, flags=re.I)
    normalized = re.sub(r"\barctan\s*\(", "atan_deg(", normalized, flags=re.I)

    normalized = DEGREE_RE.sub(r"radians(\1)", normalized)
    return normalized


def format_calculated_value(value: float) -> str:
    if not math.isfinite(value):
        raise CalculatorError("Calculated value is not finite")
    if value == 0:
        return "0"
    if abs(value) >= 1e-9 and abs(value - round(value)) < 1e-12 and abs(value) < 1e12:
        return str(int(round(value)))
    return f"{value:.12g}"
