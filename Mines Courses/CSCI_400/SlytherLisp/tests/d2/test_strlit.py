from ast import literal_eval
from hypothesis import given
import hypothesis.strategies as st
from slyther.types import String
from slyther.parser import parse_strlit


@given(st.from_regex(r'"(\\[abfnrtv"\\]|[^"\\])*"', fullmatch=True))
def test_pycompat_escapes(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert literal_eval(s) == result


@given(st.from_regex(r'"(\\[0e]|[^0-9"\\])*"', fullmatch=True))
def test_replacement_escapes(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert s[1:-1].replace(r'\0', '\0').replace(r'\e', '\x1b') == result


@given(st.from_regex(r'"\\0[0-7][0-7]"', fullmatch=True))
def test_octal(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert len(result) == 1
    assert int(s[3:5], base=8) == ord(result)


@given(st.from_regex(r'"\\0[89][89]"', fullmatch=True))
def test_bad_octal(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert result == '\x00' + s[3:5]


@given(st.from_regex(r'"\\x[0-9A-Fa-f][0-9A-Fa-f]"', fullmatch=True))
def test_hex(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert len(result) == 1
    assert int(s[3:5], base=16) == ord(result)


@given(st.from_regex(r'"\\x[G-Zg-z][A-Za-z0-9]"', fullmatch=True))
def test_bad_hex(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert result == '\\x' + s[3:5]


@given(st.from_regex(r'"\\X[0-9A-Fa-f][0-9A-Fa-f]"', fullmatch=True))
def test_bad_hex_cap(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert result == '\\X' + s[3:5]


@given(st.from_regex(r'"(\\[A-Z]|[^"\\])*"', fullmatch=True))
def test_nocaps(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert s[1:-1] == result


@given(st.from_regex(r'"(\\[cdghijklmopqsuwyz]|[^"\\])*"', fullmatch=True))
def test_nobadlower(s):
    result = parse_strlit(s)
    assert type(result) is String
    assert s[1:-1] == result
