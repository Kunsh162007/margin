"""Exact arithmetic for worked answers, without eval().

The expression is parsed into a syntax tree and only numbers, + - * / // % **,
unary signs, pi, e and a short list of maths functions are evaluated. Names,
attributes, calls to anything else, and oversized exponents are refused, so a
model-written expression cannot run code or hang the machine.
"""

from __future__ import annotations

import ast
import math
import operator

MAX_LENGTH = 200
MAX_EXPONENT = 1000

_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {name: getattr(math, name) for name in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "asin", "acos", "atan", "radians", "degrees", "floor", "ceil")}
_FUNCTIONS |= {"abs": abs, "round": round}
_CONSTANTS = {"pi": math.pi, "e": math.e}


class CalcError(ValueError):
    pass


def calculate(expression: str) -> float:
    text = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    if len(text) > MAX_LENGTH:
        raise CalcError(f"expression longer than {MAX_LENGTH} characters")
    try:
        return float(_evaluate(ast.parse(text, mode="eval").body))
    except CalcError:
        raise
    except (SyntaxError, ZeroDivisionError, OverflowError, ValueError, TypeError) as exc:
        raise CalcError(f"cannot evaluate {expression!r}: {exc}") from exc


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise CalcError("exponent too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise CalcError(f"not allowed in a calculation: {type(node).__name__}")
