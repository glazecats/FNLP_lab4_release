from __future__ import annotations

import ast
import math
from dataclasses import dataclass

import sympy as sp


class CalculatorError(ValueError):
    pass


ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": lambda x: math.sin(math.radians(x)) if abs(x) > 2 * math.pi else math.sin(x),
    "cos": lambda x: math.cos(math.radians(x)) if abs(x) > 2 * math.pi else math.cos(x),
    "tan": lambda x: math.tan(math.radians(x)) if abs(x) > 2 * math.pi else math.tan(x),
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "arcsin": math.asin,
    "arccos": math.acos,
    "arctan": math.atan,
    "radians": math.radians,
    "degrees": math.degrees,
    "sind": lambda x: math.sin(math.radians(x)),
    "cosd": lambda x: math.cos(math.radians(x)),
    "tand": lambda x: math.tan(math.radians(x)),
    "asind": lambda x: math.degrees(math.asin(x)),
    "acosd": lambda x: math.degrees(math.acos(x)),
    "atand": lambda x: math.degrees(math.atan(x)),
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "fabs": math.fabs,
    "pow": pow,
    "abs": abs,
    "min": min,
    "max": max,
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
    "u": 1.66053906660e-27,
    "m_e": 9.1093837015e-31,
    "m_p": 1.67262192369e-27,
    "m_n": 1.67492749804e-27,
}

SYMPY_FUNCS = {
    "sqrt": sp.sqrt,
    "log": sp.log,
    "ln": sp.log,
    "log10": lambda x: sp.log(x, 10),
    "exp": sp.exp,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sind": lambda x: sp.sin(sp.pi * x / 180),
    "cosd": lambda x: sp.cos(sp.pi * x / 180),
    "tand": lambda x: sp.tan(sp.pi * x / 180),
    "abs": sp.Abs,
}


@dataclass(frozen=True)
class Calculation:
    expression: str
    value: float | int
    text: str


def evaluate_expression(expression: str) -> Calculation:
    expression = expression.strip()
    expression = expression.replace("^", "**")
    if not expression:
        raise CalculatorError("empty expression")
    if len(expression) > 500:
        raise CalculatorError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(str(exc)) from exc
    _reject_ambiguous_power(tree)
    if isinstance(tree.body, ast.Compare):
        value = _eval_equation(tree.body)
        return Calculation(expression=expression, value=value, text=format_number(value))
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


def _eval_node(node: ast.AST, variables: dict[str, float | int] | None = None) -> float | int:
    variables = variables or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.id]
        raise CalculatorError(f"unknown name: {node.id}")
    if isinstance(node, ast.Attribute):
        prefix = getattr(node.value, "id", None)
        if prefix in {"math", "np", "numpy"} and node.attr in ALLOWED_NAMES:
            return ALLOWED_NAMES[node.attr]
        raise CalculatorError("only math/np constants are allowed")
    if isinstance(node, ast.UnaryOp):
        value = _eval_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.USub):
            return -value
        raise CalculatorError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, variables)
        right = _eval_node(node.right, variables)
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
        func_name = _call_name(node.func)
        if func_name in {"integral", "integrate"}:
            return _eval_integral(node, variables)
        func = ALLOWED_FUNCS.get(func_name)
        if func is None:
            raise CalculatorError(f"unknown function: {func_name}")
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        args = [_eval_node(arg, variables) for arg in node.args]
        return func(*args)
    raise CalculatorError(f"unsupported expression element: {type(node).__name__}")


def _eval_equation(node: ast.Compare) -> float:
    if len(node.ops) != 1 or not isinstance(node.ops[0], ast.Eq) or len(node.comparators) != 1:
        raise CalculatorError("only single equality equations are supported")
    unknowns = sorted(_unknown_names(node.left) | _unknown_names(node.comparators[0]))
    if len(unknowns) != 1:
        raise CalculatorError("equations must contain exactly one unknown variable")
    variable = sp.Symbol(unknowns[0])
    variables = {unknowns[0]: variable}
    left = _sympy_node(node.left, variables)
    right = _sympy_node(node.comparators[0], variables)
    solutions = sp.solve(sp.Eq(left, right), variable)
    real_solutions = []
    for solution in solutions:
        numeric = complex(sp.N(solution))
        if abs(numeric.imag) < 1e-9 and math.isfinite(numeric.real):
            real_solutions.append(float(numeric.real))
    if not real_solutions:
        raise CalculatorError("equation has no real numeric solution")
    if len(real_solutions) == 1:
        return real_solutions[0]
    nonnegative = [value for value in real_solutions if value >= 0]
    if len(nonnegative) == 1:
        return nonnegative[0]
    raise CalculatorError("equation has multiple real solutions; isolate the desired root")


def _unknown_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_NAMES and node.id not in ALLOWED_FUNCS and node.id not in {"math", "np", "numpy"}:
            names.add(node.id)
        return names
    if isinstance(node, ast.Call):
        for arg in node.args:
            names.update(_unknown_names(arg))
        return names
    for child in ast.iter_child_nodes(node):
        names.update(_unknown_names(child))
    return names


def _sympy_node(node: ast.AST, variables: dict[str, sp.Symbol]) -> sp.Expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sp.Float(node.value) if isinstance(node.value, float) else sp.Integer(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return variables[node.id]
        if node.id in ALLOWED_NAMES:
            return sp.Float(ALLOWED_NAMES[node.id])
        raise CalculatorError(f"unknown name: {node.id}")
    if isinstance(node, ast.Attribute):
        prefix = getattr(node.value, "id", None)
        if prefix in {"math", "np", "numpy"} and node.attr in ALLOWED_NAMES:
            return sp.Float(ALLOWED_NAMES[node.attr])
        raise CalculatorError("only math/np constants are allowed")
    if isinstance(node, ast.UnaryOp):
        value = _sympy_node(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return -value
        raise CalculatorError("unsupported unary operator")
    if isinstance(node, ast.BinOp):
        left = _sympy_node(node.left, variables)
        right = _sympy_node(node.right, variables)
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
        raise CalculatorError("unsupported binary operator")
    if isinstance(node, ast.Call):
        func_name = _call_name(node.func)
        func = SYMPY_FUNCS.get(func_name)
        if func is None:
            raise CalculatorError(f"unknown function in equation: {func_name}")
        if node.keywords:
            raise CalculatorError("keyword arguments are not allowed")
        args = [_sympy_node(arg, variables) for arg in node.args]
        return func(*args)
    raise CalculatorError(f"unsupported equation element: {type(node).__name__}")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id in {"math", "np", "numpy"}
    ):
        return node.attr
    raise CalculatorError("only direct function calls or math/np.<func> calls are allowed")


def _reject_ambiguous_power(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow) and isinstance(node.right, ast.BinOp):
            if isinstance(node.right.op, ast.Pow):
                raise CalculatorError("ambiguous chained exponentiation; use sqrt((...)**3) or split the powers")


def _eval_integral(node: ast.Call, variables: dict[str, float | int]) -> float:
    if node.keywords:
        raise CalculatorError("integral does not accept keyword arguments")
    if len(node.args) != 4:
        raise CalculatorError("integral syntax is integral(expr, x, lower, upper)")
    expr_node, var_node, lower_node, upper_node = node.args
    if not isinstance(var_node, ast.Name):
        raise CalculatorError("integral variable must be a bare name, for example x")
    lower = float(_eval_node(lower_node, variables))
    upper = float(_eval_node(upper_node, variables))
    if not (math.isfinite(lower) and math.isfinite(upper)):
        raise CalculatorError("integral bounds must be finite")
    if lower == upper:
        return 0.0

    steps = 4096
    width = (upper - lower) / steps

    def sample(x_value: float) -> float:
        scoped = dict(variables)
        scoped[var_node.id] = x_value
        value = float(_eval_node(expr_node, scoped))
        if not math.isfinite(value):
            raise CalculatorError("integral sampled a non-finite value")
        return value

    total = sample(lower) + sample(upper)
    odd_sum = 0.0
    even_sum = 0.0
    for i in range(1, steps):
        value = sample(lower + i * width)
        if i % 2:
            odd_sum += value
        else:
            even_sum += value
    return width * (total + 4 * odd_sum + 2 * even_sum) / 3
