from hypothesis import given
import hypothesis.strategies as st
from slyther.types import ConsCell

simple_objects = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.floats(),
    st.text())


@given(simple_objects, simple_objects)
def test_cons_init(car, cdr):
    cell = ConsCell(car, cdr)
    assert cell.car is car
    assert cell.cdr is cdr


@given(simple_objects, simple_objects)
def test_cons_eq_self(car, cdr):
    cell = ConsCell(car, cdr)
    if car == car and cdr == cdr:
        assert cell == cell
    else:
        assert cell != cell


def test_cons_eq_others():
    a = ConsCell(1, 2)
    b = ConsCell(0, 1)
    c1 = ConsCell(a, b)
    c2 = ConsCell(a, b)
    d = ConsCell(b, a)
    e = ConsCell(c1, c2)

    assert a == a
    assert b == b
    assert a != b
    assert c1 == c2
    assert c2 == c1
    assert c1 != d
    assert e == ConsCell(ConsCell(a, b), ConsCell(a, b))
    assert a != d
    assert d != a


@given(simple_objects, simple_objects, simple_objects)
def test_cons_eq_noncell(car, cdr, other):
    cell = ConsCell(car, cdr)
    assert cell != other


@given(simple_objects, simple_objects)
def test_cons_repr(car, cdr):
    cell = ConsCell(car, cdr)
    r = repr(cell)
    assert r.startswith('(cons ')
    assert r.endswith(')')
    assert repr(car) in r
    assert repr(cdr) in r
