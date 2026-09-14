import pytest

from margin.practice.maths import contradicts, equivalent, formulas, parse, points_met_by_formula


def test_a_contradiction_needs_the_same_letters_solved_for_the_same_quantity():
    book = r"mg=G\frac{mM_{\mathrm{E}}}{r^{2}}"
    assert contradicts(r"g = \frac{G M_E}{r}", book)  # solved for g, both use G, M_E and r, and they differ
    assert not contradicts(r"g = \frac{G M_E}{r^2}", book)  # the same relation
    assert not contradicts("v = v_0 + a t", r"\overline{v}=\frac{v_0+v}{2}")  # a different relation, not a contrary one
    assert not contradicts("hello", book) and not contradicts("x = 1, 2", book)


@pytest.mark.parametrize(("a", "b"), [
    (r"\frac{1}{\sqrt{3}}", r"\frac{\sqrt{3}}{3}"),
    (r"M = \frac{g R^2}{G}", r"g = \frac{G M}{R^2}"),  # the same relation solved for another letter
    ("v^2 = 2as", r"v = \sqrt{2 a s}"),
    (r"T = 2\pi\sqrt{\frac{L}{g}}", r"T^2 = \frac{4 \pi^2 L}{g}"),
    (r"288 \pi", "288*pi"),
    (r"\left( 3, \frac{\pi}{2} \right)", "(3, pi/2)"),
    ("x = 5", "5"),
    ("E = mc^2", "E = m c²"),
    (r"5.97 \times 10^{24}", "5.97e24"),
    (r"\dfrac{17}{50}", "0.34"),
    (r"F_{net} = m a", "F_net = a m"),
    (r"\boxed{\frac{3}{2}}", "1.5"),
    (r"mg = G \frac{m M_{\mathrm{E}}}{r^2}", r"g = \frac{G M_E}{r^2}"),  # the book's form, with m cancelled
    (r"\vec{\bf F}_{\mathrm{net}} = m \vec{\bf a}", "F_{net} = m a"),  # vector notation reads as the letters
    (r"mg=G\frac{mM_{\mathrm{E}}}{r^{2}}", r"g = \frac{G M_E}{r^2}"),  # unspaced, as the formula reader writes it
    (r"90^\circ", "90"),
    (r"10,\!080", "10080"),
    (r"11,\! 111,\! 111,\! 100", "11111111100"),
])
def test_equivalent_forms_match_both_ways(a, b):
    assert equivalent(a, b) and equivalent(b, a)


@pytest.mark.parametrize(("a", "b"), [
    (r"M = \frac{g R^2}{G}", r"M = \frac{g R}{G}"),
    (r"\frac{1}{3}", "0.34"),
    ("11111111100", "11111111101"),  # exact numbers differ even when the relative gap is tiny
    ("11111111100 x", "11111111101 x"),
    ("v = at", "v = a + t"),
    ("(3, pi/2)", "(pi/2, 3)"),
    ("x = 5", "y = 5"),
    (r"g = \frac{G M}{r^2}", r"a = \frac{G M}{r^2}"),
    ("hello world", "hello world"),  # prose is not maths, even when identical
    ("x = 1, 2", "x = 1"),  # a list on one side is not a relation; it must not raise
    ("y = 3 x", "x = 1, 2"),
    ("", "0"),
])
def test_different_or_unparsable_text_does_not_match(a, b):
    assert not equivalent(a, b)


def test_parse_refuses_code_prose_and_oversized_text():
    assert parse("__import__('os').system('echo hi')") is None
    assert parse("exit()") is None
    assert parse("the answer is obviously five") is None
    assert parse("x+" * 200 + "1") is None
    assert parse(r"\frac{1}{2") is None  # unbalanced braces


def test_formulas_finds_relations_and_values_but_not_bare_symbols():
    text = r"The mass is $M = \frac{g R^2}{G}$ where $g$ and $R$ are known; plainly F = ma holds, giving $5.97 \times 10^{24}$ kg."
    assert formulas(text) == [r"M = \frac{g R^2}{G}", r"5.97 \times 10^{24}", "F = ma"]


def test_only_formula_marking_points_are_met_by_an_equivalent_formula():
    points = (
        r"States the correct rearranged formula: $M = \frac{g R^2}{G}$ (1 mark).",
        r"Correctly identifies that $g$ and $R$ are the known inputs used to find $M$ (1 mark).",
        r"Explains that $F = ma$ means acceleration doubles when force doubles.",
    )
    answer = r"Since $g = GM/R^2$, the mass follows, and F = m a."
    assert points_met_by_formula(points, answer) == (True, False, False)
    assert points_met_by_formula(points, "") == (False, False, False)
