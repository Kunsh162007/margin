"""Compare formulas and numbers by value, not by spelling.

The question and marking checks compare words, so a correct answer written as a
formula shared few words with anything: a question whose model answer was
M = gR²/G was rejected for exactly that. Here a formula — LaTeX or plain text —
is parsed with SymPy and compared by evaluating both sides at fixed random
points, so 1/√3 matches √3/3, and M = gR²/G matches g = GM/R², the same relation
solved for another letter. Text that does not parse is never equal to anything:
this can add credit only where two formulas agree numerically.

SymPy's ``parse_expr`` evaluates Python, so text is first reduced to a small
alphabet, names of four or more letters (prose, ``exit``) are refused, and every
remaining name becomes a plain symbol.
"""

from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Any

DECIMAL_TOL = 1e-6  # a written decimal is rounded, so it matches within a relative tolerance
EXACT_TOL = 1e-20  # exact numbers must agree to 20 digits: 11111111100 is not 11111111101
DIGITS = 40
SAMPLES = 4
SEED = 7
MAX_CHARS = 160
MAX_SYMBOLS = 6
MAX_SOLUTIONS = 4

MARK_NOTE = re.compile(r"\(\s*(?:\d+|one|two|three|half)\s+marks?\b[^)]*\)", re.IGNORECASE)
_MATH_SPAN = re.compile(r"\$\$(.+?)\$\$|\$(.+?)\$|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
_PLAIN_EQUATION = re.compile(r"(?<![\w\\$])([A-Za-z][A-Za-z0-9_]*\s*=\s*[-A-Za-z0-9_+*/^().√²³π·× ]+)")
_PROSE_WORD = re.compile(r"\s[A-Za-z]{3,}\b.*$")
_SAFE = re.compile(r"[A-Za-z0-9_+\-*/^().,= ]*")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_WORD = re.compile(r"[a-z]{3,}")
FUNCTIONS = frozenset("sqrt sin cos tan asin acos atan sinh cosh tanh log exp pi oo".split())
GREEK = frozenset(
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lamda mu nu xi rho sigma tau "
    "upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Phi Psi Omega".split()
)
DROPPED_COMMANDS = frozenset({"text", "mathrm", "textrm", "textsf"})  # units and words: "9.8\,\text{m/s}^2" compares as 9.8
KEPT_COMMANDS = frozenset({"boxed", "operatorname", "mathbf", "mathit", "boldsymbol", "vec", "hat", "widehat", "bar", "overline",
                           "overrightarrow", "tilde", "dot", "ddot"})  # \vec{\bf F} is the letter F
SWITCH_COMMANDS = frozenset({"bf", "rm", "it", "cal", "displaystyle", "textstyle", "scriptstyle", "scriptsize", "small", "quad", "qquad"})
MARKING_WORDS = frozenset(
    "the and for with its that states state stated stating give gives giving write writes writing use uses using "
    "correct correctly rearranged rearranges rearranging rearrangement formula formulas equation equations expression "
    "relation relationship final answer value result mark marks show shows derive derives obtain obtains".split()
)
_THOUSANDS = re.compile(r"(?<=\d),\\!\s*(?=\d{3}(?!\d))")  # 10,\!080 is ten thousand and eighty, not a pair
_REPLACE = (
    ("^{\\circ}", ""), ("^\\circ", ""), ("\\circ", ""), ("°", ""),  # degrees: 90^\circ compares as 90
    ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\cdot", "*"), ("\\times", "*"), ("\\div", "/"), ("\\left", ""),
    ("\\right", ""), ("\\infty", " oo "), ("\\ln", " log"), ("\\lambda", " lamda "), ("×", "*"), ("·", "*"), ("−", "-"),
    ("÷", "/"), ("π", " pi "), ("√", "sqrt"), ("²", "^2"), ("³", "^3"),
)


def _skip(text: str, i: int) -> int:
    while i < len(text) and text[i] == " ":
        i += 1
    return i


def _group(text: str, start: int) -> tuple[str, int]:
    """The ``{...}`` argument starting at ``start`` (or one character, as in ``\\frac12``) and the index after it."""
    if start >= len(text):
        raise ValueError("missing argument")
    if text[start] != "{":
        return text[start], start + 1
    depth = 0
    for i in range(start, len(text)):
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        if depth == 0:
            return text[start + 1:i], i + 1
    raise ValueError("unbalanced braces")


def _command(text: str, i: int) -> tuple[str, int]:
    """Expand the LaTeX command at ``text[i]`` (a backslash)."""
    name = re.match(r"\\([A-Za-z]*)", text[i:]).group(1)
    end = i + 1 + len(name)
    if name == "frac":
        num, j = _group(text, _skip(text, end))
        den, j = _group(text, _skip(text, j))
        return f"(({_expand(num)})/({_expand(den)}))", j
    if name == "sqrt":
        j, index = _skip(text, end), None
        if j < len(text) and text[j] == "[":
            close = text.index("]", j)
            index, j = text[j + 1:close], _skip(text, close + 1)
        arg, j = _group(text, j)
        return (f"sqrt({_expand(arg)})" if index is None else f"(({_expand(arg)})**(1/({_expand(index)})))"), j
    if name in DROPPED_COMMANDS or name in KEPT_COMMANDS:
        arg, j = _group(text, _skip(text, end))
        return ("" if name in DROPPED_COMMANDS else _expand(arg)), j
    if name in SWITCH_COMMANDS:
        return " ", end
    return (f" {name} " if name else " "), end  # \sin, \pi, \theta; "\," and "\%" become spaces


def _expand(text: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i] == "\\":
            piece, i = _command(text, i)
        elif text[i] in "^_" and i + 1 < len(text) and text[i + 1] == "{":
            marker = text[i]
            arg, i = _group(text, i + 1)
            piece = f"**({_expand(arg)})" if marker == "^" else "_" + re.sub(r"\\[A-Za-z]+|\W", "", arg)  # F_{\mathrm{net}} is F_net
        else:
            piece, i = text[i], i + 1
        out.append(piece)
    return "".join(out)


def _names(text: str) -> str | None:
    """Split short unknown names into letters (``gR`` → ``g R``); refuse prose and anything longer."""
    def split(match: re.Match[str]) -> str:
        name = match.group(0)
        head, _, tail = name.partition("_")
        if tail and len(head) > 1 and head.isalpha() and head not in GREEK:
            return " ".join(head[:-1]) + f" {head[-1]}_{tail}"  # mM_E is m times M_E
        if "_" in name or name in FUNCTIONS or name in GREEK or re.fullmatch(r"[A-Za-z]\d*", name):
            return name
        if len(name) >= 4:
            raise ValueError(f"not maths: {name}")
        return " ".join(name)

    try:
        return _NAME.sub(split, text)
    except ValueError:
        return None


def to_text(formula: str) -> str | None:
    """LaTeX or plain maths as SymPy input text, or None when it is not safely parsable."""
    text = _THOUSANDS.sub("", formula.strip().strip("$").strip())
    for old, new in _REPLACE:
        text = text.replace(old, new)
    try:
        text = _expand(text)
    except (ValueError, IndexError, AttributeError):
        return None
    text = re.sub(r"\s+", " ", text.replace("{", "(").replace("}", ")")).strip().rstrip(".,;")
    if not text or len(text) > MAX_CHARS or "__" in text or not _SAFE.fullmatch(text):
        return None
    return _names(text)


@lru_cache(maxsize=4096)
def parse(formula: str) -> tuple[str, Any] | None:
    """``("eq", (lhs, rhs))``, ``("tuple", items)`` or ``("expr", expr)``; None when the text is not maths."""
    text = to_text(formula)
    if text is None or text.count("=") > 1:
        return None
    import sympy
    from sympy.parsing.sympy_parser import convert_xor, implicit_multiplication_application, parse_expr, standard_transformations

    transforms = standard_transformations + (implicit_multiplication_application, convert_xor)
    symbols = {n: sympy.Symbol(n) for n in set(_NAME.findall(text)) - FUNCTIONS}
    try:
        sides = [parse_expr(side, local_dict=symbols, transformations=transforms) for side in text.split("=")]
        if len(sides) == 2:
            left, right = sympy.sympify(sides[0]), sympy.sympify(sides[1])
            if not (isinstance(left, sympy.Expr) and isinstance(right, sympy.Expr)):
                return None  # "x = 1, 2" lists values; it is not a relation to compare
            return ("eq", (left, right))
        if isinstance(sides[0], tuple):
            return ("tuple", tuple(sympy.sympify(x) for x in sides[0]))
        return ("expr", sympy.sympify(sides[0]))
    except Exception:  # parse_expr raises many unrelated types for text that is not maths
        return None


def _points(symbols: list[Any]) -> list[dict[Any, Any]]:
    """Exact rational sample points between 1.1 and 2.9, the same on every run."""
    import sympy

    rng = random.Random(SEED)
    return [{s: sympy.Rational(rng.randint(110, 290), 100) for s in symbols} for _ in range(SAMPLES if symbols else 1)]


def _tolerance(*exprs: Any) -> float:
    import sympy

    return DECIMAL_TOL if any(e.has(sympy.Float) for e in exprs) else EXACT_TOL


def _agree(a: Any, b: Any, point: dict[Any, Any], tol: float) -> bool:
    """Both sides evaluated to 40 digits at one point agree within ``tol`` of their size."""
    import sympy

    try:
        va, vb = sympy.N(a.subs(point), DIGITS), sympy.N(b.subs(point), DIGITS)
        if va.free_symbols or vb.free_symbols or not (va.is_finite and vb.is_finite):
            return False
        return bool(abs(va - vb) <= tol * max(1, abs(va), abs(vb)))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError, AttributeError):
        return False


def _same_value(a: Any, b: Any) -> bool:
    if a == b:  # identical structure, including values with no finite number such as oo
        return True
    symbols = sorted(a.free_symbols | b.free_symbols, key=str)
    tol = _tolerance(a, b)
    return len(symbols) <= MAX_SYMBOLS and all(_agree(a, b, p, tol) for p in _points(symbols))


def _solutions(left: Any, right: Any, symbol: Any) -> list[Any]:
    import sympy

    try:
        found = sympy.solve(left - right, symbol)
    except (NotImplementedError, ValueError, TypeError):
        return []
    return found if len(found) <= MAX_SOLUTIONS else []


def _same_relation(first: tuple[Any, Any], second: tuple[Any, Any]) -> bool:
    """Solve both relations for a letter they share; they are the same relation when a solution agrees.

    A letter that cancels — the m in the book's mg = GmM/r² — disappears in the solving,
    so that form matches a student's g = GM/r².
    """
    (l1, r1), (l2, r2) = first, second
    own, other = (l1 - r1).free_symbols, (l2 - r2).free_symbols
    shared = own & other
    if not shared or len(own | other) > MAX_SYMBOLS + 2:
        return False
    for target in sorted(shared, key=str):
        mine, theirs = _solutions(l1, r1, target), _solutions(l2, r2, target)
        if any(_same_value(a, b) for a in mine for b in theirs):
            return True
    return False


def _as_value(parsed: tuple[str, Any]) -> tuple[str, Any]:
    """``x = 5`` compared with a bare value is its right-hand side."""
    import sympy

    if parsed[0] == "eq" and isinstance(parsed[1][0], sympy.Symbol):
        return ("expr", parsed[1][1])
    return parsed


@lru_cache(maxsize=8192)
def equivalent(a: str, b: str) -> bool:
    pa, pb = parse(a), parse(b)
    if pa is None or pb is None:
        return False
    if pa[0] == "eq" and pb[0] == "eq":
        return _same_relation(pa[1], pb[1])
    pa, pb = _as_value(pa), _as_value(pb)
    if pa[0] != pb[0] or pa[0] == "eq":
        return False
    if pa[0] == "tuple":
        return len(pa[1]) == len(pb[1]) and all(map(_same_value, pa[1], pb[1]))
    return _same_value(pa[1], pb[1])


def _checkable(formula: str) -> bool:
    """Relations and numbers; a bare letter like $g$ names a quantity and proves nothing."""
    parsed = parse(formula)
    if parsed is None:
        return False
    if parsed[0] == "eq":
        return True
    items = parsed[1] if parsed[0] == "tuple" else (parsed[1],)
    return all(not item.free_symbols for item in items)


def formulas(text: str) -> list[str]:
    """Checkable formulas in text: ``$...$``, ``\\(...\\)`` and ``\\[...\\]`` spans, then plain equations like ``F = ma``."""
    spans = [next(g for g in m.groups() if g is not None).strip() for m in _MATH_SPAN.finditer(text)]
    plain = [_PROSE_WORD.sub("", m.group(1)).strip() for m in _PLAIN_EQUATION.finditer(_MATH_SPAN.sub(" ", text))]
    return [f for f in spans + plain if f and _checkable(f)]


def relations(text: str) -> list[str]:
    """Formulas stating a relation between at least two letters (F = ma); a value like v = 3.0 m/s is not one."""
    found = []
    for f in formulas(text):
        parsed = parse(f)
        if parsed is not None and parsed[0] == "eq" and len((parsed[1][0] - parsed[1][1]).free_symbols) >= 2:
            found.append(f)
    return found


def _contrary(first: tuple[Any, Any], second: tuple[Any, Any]) -> bool:
    """Solved for a quantity both use, the two give different expressions in exactly the same letters."""
    (l1, r1), (l2, r2) = first, second
    own, other = (l1 - r1).free_symbols, (l2 - r2).free_symbols
    if len(own | other) > MAX_SYMBOLS + 2:
        return False
    for target in sorted(own & other, key=str):
        for mine in _solutions(l1, r1, target):
            for theirs in _solutions(l2, r2, target):
                if mine.free_symbols and mine.free_symbols == theirs.free_symbols and not _same_value(mine, theirs):
                    return True
    return False


@lru_cache(maxsize=8192)
def contradicts(a: str, b: str) -> bool:
    """True when relation ``a`` states something different about the same quantities as relation ``b``.

    g = GM/r contradicts the book's mg = GmM/r² (solved for g, both use G, M and r). A relation
    the book does not show, such as v = v₀ + at beside v̄ = (v₀ + v)/2, does not: it uses other
    letters. Requiring every formula to *appear* in the section rejected correct ones — the formula
    reader misses or misreads some equations, and the model sees only part of a long section.
    """
    pa, pb = parse(a), parse(b)
    if pa is None or pb is None or pa[0] != "eq" or pb[0] != "eq":
        return False
    return not _same_relation(pa[1], pb[1]) and _contrary(pa[1], pb[1])


def relations_consistent(text: str, source_relations: list[str]) -> bool:
    """No relation in ``text`` contradicts the source, unless it also matches one there. True when the
    source has no relations, as in a book read before its displayed equations were recovered."""
    for f in relations(text):
        if any(contradicts(f, s) for s in source_relations) and not any(equivalent(f, s) for s in source_relations):
            return False
    return True


def strip_maths(text: str) -> str:
    """Text without formula spans or "(1 mark for ...)" notes, for the word-overlap checks."""
    return MARK_NOTE.sub(" ", _MATH_SPAN.sub(" ", text))


def _formula_only(point: str) -> bool:
    return not [w for w in _WORD.findall(strip_maths(point).lower()) if w not in MARKING_WORDS]


def points_met_by_formula(points: tuple[str, ...], answer: str) -> tuple[bool, ...]:
    """Marking points that are only a formula ("States $M = gR^2/G$"), met when the answer has an equivalent one."""
    found = formulas(answer)

    def met(point: str) -> bool:
        needed = formulas(point)
        return bool(needed and found and _formula_only(point)) and all(any(equivalent(f, g) for g in found) for f in needed)

    return tuple(met(p) for p in points)
