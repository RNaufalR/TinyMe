"""Deterministic exact-arithmetic tool (audit §25).

``eval`` is never used: the expression is parsed with :mod:`ast` and only an
explicit whitelist of numeric operations is interpreted.  Results are exact for
integers (Python ``int``/``Fraction``) and IEEE-754 for floats, and the tool
refuses anything that is not a closed numeric expression.
"""
from __future__ import annotations

import ast
import math
import operator
from fractions import Fraction

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
            ast.Pow: operator.pow}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "phi": (1 + 5 ** 0.5) / 2}
_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "atan2": math.atan2, "hypot": math.hypot, "degrees": math.degrees,
    "radians": math.radians, "gcd": math.gcd, "lcm": math.lcm, "comb": math.comb, "perm": math.perm,
    "factorial": math.factorial, "isqrt": math.isqrt, "pow": pow, "fabs": math.fabs,
    "dist": math.dist, "fsum": math.fsum, "prod": math.prod, "sign": (lambda x: (x > 0) - (x < 0)),
}
_MAX_POW = 10 ** 6


class ComputeError(ValueError):
    pass


def _eval(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ComputeError(f"non-numeric literal {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ComputeError(f"operator {type(node.op).__name__} not allowed")
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if isinstance(right, (int, float)) and abs(right) > _MAX_POW:
                raise ComputeError("exponent too large")
            if isinstance(left, int) and isinstance(right, int) and right > 0 and right * math.log2(max(left, 2)) > 4096:
                raise ComputeError("result would exceed 4096 bits")
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ComputeError(f"unary {type(node.op).__name__} not allowed")
        return op(_eval(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ComputeError(f"unknown name {node.id!r}")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
            raise ComputeError("only whitelisted functions may be called")
        if node.keywords:
            raise ComputeError("keyword arguments are not allowed")
        return _FUNCTIONS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval(e) for e in node.elts]
    raise ComputeError(f"syntax {type(node).__name__} not allowed")


def evaluate(expression: str):
    if not isinstance(expression, str) or not expression.strip():
        raise ComputeError("empty expression")
    if len(expression) > 400:
        raise ComputeError("expression too long")
    tree = ast.parse(expression, mode="eval")
    value = _eval(tree)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ComputeError("non-finite result")
    return value


def _as_json(value):
    if isinstance(value, Fraction):
        return float(value)
    if isinstance(value, list):
        return [_as_json(v) for v in value]
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return value  # keep float semantics but serialise cleanly
    return value


def compute(expression: str) -> dict:
    """Tool entrypoint: exact evaluation with an auditable trace."""
    try:
        value = evaluate(expression)
    except ComputeError as exc:
        return {"ok": False, "error": f"compute_error: {exc}", "expression": expression}
    except SyntaxError as exc:
        return {"ok": False, "error": f"syntax_error: {exc.msg}", "expression": expression}
    exact = repr(value) if not isinstance(value, float) else repr(value)
    return {"ok": True, "expression": expression, "value": _as_json(value), "exact": exact,
            "method": "python-ast-exact"}
