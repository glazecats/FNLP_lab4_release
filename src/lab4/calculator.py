from __future__ import annotations

import ast
import math
from dataclasses import dataclass


class CalculatorError(ValueError):
    pass


ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
    "pow": pow,
    "abs": abs,
}

ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "c": 299792458.0,
    "h": 6.62607015e-34,
    "hbar": 1.054571817e-34,
    "k_B": 1.380649e-23,
    "kB": 1.380649e-23,
    "kB_eV": 8.617333262145e-5,
    "N_A": 6.02214076e23,
    "NA": 6.02214076e23,
    "R": 8.31446261815324,
    "F": 96485.33212331001,
    "e_charge": 1.602176634e-19,
    "epsilon0": 8.8541878128e-12,
    "mu0": 1.25663706212e-6,
    "G": 6.67430e-11,
    "g": 9.80665,
    "sigma": 5.670374419e-8,
    "amu": 1.66053906660e-27,
}


@dataclass(frozen=True)
class Calculation:
    expression: str
    value: float | int
    text: str


def evaluate_expression(expression: str) -> Calculation:
    expression = expression.strip()
    if not expression:
        raise CalculatorError("empty expression")
    if len(expression) > 500:
        raise CalculatorError("expression too long")
    tree = ast.parse(expression, mode="eval")
    value = _eval_node(tree.body)
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("non-finite result")
    return Calculation(expression=expression, value=value, text=format_number(value))


def format_number(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if value == 0:
        return "0"
    abs_value = abs(value)
    if 1e-4 <= abs_value < 1e6:
        return f"{value:.12g}"
    return f"{value:.12e}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        raise CalculatorError(f"unknown name: {node.id}")
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise CalculatorError("unsupported unary operator")
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
        raise CalculatorError("unsupported binary operator")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("only direct function calls are allowed")
        func = ALLOWED_FUNCS.get(node.func.id)
        if func is None:
            raise CalculatorError(f"unknown function: {node.func.id}")
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        args = [_eval_node(arg) for arg in node.args]
        return func(*args)
    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")

