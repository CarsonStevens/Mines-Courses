import pytest
from hypothesis import given
from hypothesis.strategies import from_regex
from slyther.types import Symbol, String
from slyther.parser import LParen, RParen, Quote, lex


@given(from_regex(r'\s*-?([0-9]*\.[0-9]+|[0-9]+\.[0-9]*)\s*', fullmatch=True))
def test_fp_parse(s):
    num = next(lex(s))
    assert num == float(s.strip())


@given(from_regex(r'\s*"(\\"|[^"])*', fullmatch=True))
def test_unclosed_string(s):
    with pytest.raises(SyntaxError):
        next(lex(s))


@given(from_regex(r'\s*\.[^\d]*', fullmatch=True))
def test_invalid_period(s):
    with pytest.raises(SyntaxError):
        next(lex(s))
