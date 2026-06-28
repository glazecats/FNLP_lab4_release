from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from typing import Any


SAFE_NAMES = {
    "pi": math.pi,
    "e": math.e,
}

SAFE_FUNCTIONS = {
    "abs": abs,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "radians": math.radians,
    "degrees": math.degrees,
}

ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


@dataclass
class CalculationRequest:
    label: str
    expression: str
    source: str = ""
    purpose: str = ""


@dataclass
class CalculationResult:
    label: str
    expression: str
    value: str | None
    source: str = ""
    purpose: str = ""
    error: str | None = None


def format_number(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("non-finite result")
    if value == 0:
        return "0"
    if abs(value - round(value)) < 1e-12 and abs(value) < 1e12:
        return str(int(round(value)))
    return f"{value:.12g}"


def normalize_expression(expression: str) -> str:
    value = expression.strip()
    value = value.replace("^", "**")
    value = value.replace("×", "*").replace("脳", "*").replace("·", "*")
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    value = re.sub(r"\bln\s*\(", "log(", value)
    value = re.sub(r"\barcsin\s*\(", "asin(", value)
    value = re.sub(r"\barccos\s*\(", "acos(", value)
    value = re.sub(r"\barctan\s*\(", "atan(", value)
    return value


class SafeEvaluator(ast.NodeVisitor):
    def visit_Expression(self, node: ast.Expression) -> float:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("only numeric constants are allowed")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id in SAFE_NAMES:
            return float(SAFE_NAMES[node.id])
        raise ValueError(f"unknown name: {node.id}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        if not isinstance(node.op, ALLOWED_UNARYOPS):
            raise ValueError("unsupported unary operator")
        value = self.visit(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value

    def visit_BinOp(self, node: ast.BinOp) -> float:
        if not isinstance(node.op, ALLOWED_BINOPS):
            raise ValueError("unsupported binary operator")
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            if abs(right) > 1000:
                raise ValueError("exponent too large")
            return left**right
        return left % right

    def visit_Call(self, node: ast.Call) -> float:
        if not isinstance(node.func, ast.Name) or node.func.id not in SAFE_FUNCTIONS:
            raise ValueError("unsupported function")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [self.visit(arg) for arg in node.args]
        return float(SAFE_FUNCTIONS[node.func.id](*args))

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"unsupported syntax: {type(node).__name__}")


def safe_eval_expression(expression: str) -> str:
    normalized = normalize_expression(expression)
    if len(normalized) > 300:
        raise ValueError("expression too long")
    tree = ast.parse(normalized, mode="eval")
    value = SafeEvaluator().visit(tree)
    return format_number(value)


def extract_json_array(text: str) -> list[Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        data = json.loads(stripped)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def parse_calculation_requests(response: str, *, limit: int = 6) -> list[CalculationRequest]:
    requests: list[CalculationRequest] = []
    for item in extract_json_array(response):
        if not isinstance(item, dict):
            continue
        expression = str(item.get("expression", "")).strip()
        if not expression:
            continue
        requests.append(
            CalculationRequest(
                label=str(item.get("label", f"calc_{len(requests) + 1}")).strip() or f"calc_{len(requests) + 1}",
                expression=expression,
                source=str(item.get("source", "")).strip(),
                purpose=str(item.get("purpose", "")).strip(),
            )
        )
        if len(requests) >= limit:
            break
    return requests


def evaluate_calculation_requests(requests: list[CalculationRequest]) -> list[CalculationResult]:
    results: list[CalculationResult] = []
    for request in requests:
        try:
            value = safe_eval_expression(request.expression)
            results.append(
                CalculationResult(
                    label=request.label,
                    expression=request.expression,
                    value=value,
                    source=request.source,
                    purpose=request.purpose,
                )
            )
        except Exception as exc:
            results.append(
                CalculationResult(
                    label=request.label,
                    expression=request.expression,
                    value=None,
                    source=request.source,
                    purpose=request.purpose,
                    error=str(exc),
                )
            )
    return results
