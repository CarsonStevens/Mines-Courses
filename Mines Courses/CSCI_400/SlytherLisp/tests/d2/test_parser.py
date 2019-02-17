import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from slyther.types import (Symbol as s, String as sl,       # noqa
                           Quoted, SExpression, NIL)
from slyther.parser import LParen, RParen, Quote, parse

lp, rp, q = LParen(), RParen(), Quote()


def s_expressions(*args, **kwargs):
    return st.builds(SExpression.from_iterable, st.lists(*args, **kwargs))


def quoteds(strategy):
    return st.builds(Quoted, strategy)


symbols = st.builds(
    s,
    st.from_regex(r'''[^0-9"'();.\s][^"'();\s]*''', fullmatch=True))
strings = st.builds(sl, st.text())
ast_objects = st.deferred(
    lambda: (symbols
             | strings
             | st.integers()
             | st.floats(allow_nan=False, allow_infinity=False)
             | quoteds(ast_objects)
             | s_expressions(ast_objects)))
ast_lists = st.lists(ast_objects)


def deparse(exprs):
    for expr in exprs:
        if type(expr) in {int, float, s, sl}:
            yield expr
        elif isinstance(expr, Quoted):
            yield q
            yield from deparse([expr.elem])
        elif expr is NIL:
            yield lp
            yield rp
        elif isinstance(expr, SExpression):
            yield lp
            yield from deparse(expr)
            yield rp
        else:
            raise ValueError("bad deparse")


@settings(deadline=None)
@given(ast_lists)
def test_valid_parses(ast):
    assert list(parse(deparse(ast))) == ast


def test_missing_rp():
    parser = parse(iter([lp, rp, lp, lp, rp]))
    assert next(parser) is NIL
    with pytest.raises(SyntaxError):
        next(parser)


def test_missing_lp():
    parser = parse(iter([10, rp]))
    assert next(parser) is 10
    with pytest.raises(SyntaxError):
        next(parser)


def test_bad_quotation_end():
    parser = parse(iter([q, -15, q, q]))
    assert next(parser) == Quoted(-15)
    with pytest.raises(SyntaxError):
        next(parser)


def test_bad_quotation_rp():
    parser = parse(iter([q, s('a'), lp, q, q, q, rp]))
    assert next(parser) == Quoted(s('a'))
    with pytest.raises(SyntaxError):
        next(parser)


@settings(deadline=None)
@given(st.integers(min_value=1, max_value=75))
def test_many_quotes(quotes):
    parser = parse(iter([q] * quotes + [lp, rp]))
    r = next(parser)
    for _ in range(quotes):
        assert isinstance(r, Quoted)
        r = r.elem
    assert r is NIL
    with pytest.raises(StopIteration):
        next(parser)
