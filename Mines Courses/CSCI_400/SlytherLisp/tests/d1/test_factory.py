from hypothesis import given
import hypothesis.strategies as st
from slyther.types import cons, ConsCell, ConsList, SExpression, NIL

simple_objects = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text())


@given(simple_objects)
def test_cons_onto_nil(car):
    cell = cons(car, NIL)
    assert cell.car is car
    assert cell.cdr is NIL
    assert type(cell) is ConsList


@given(simple_objects, simple_objects, simple_objects)
def test_cons_onto_conscell(car, cadr, cddr):
    cdr = ConsCell(cadr, cddr)
    cell = cons(car, cdr)

    assert cell.car is car
    assert cell.cdr is cdr
    assert type(cell) is ConsCell


@given(simple_objects, simple_objects)
def test_cons_onto_conslist(car, cadr):
    cdr = ConsList(cadr, NIL)
    cell = cons(car, cdr)

    assert cell.car is car
    assert cell.cdr is cdr
    assert type(cell) is ConsList


@given(simple_objects, simple_objects)
def test_cons_onto_se(car, cadr):
    cdr = SExpression(cadr, NIL)
    cell = cons(car, cdr)

    assert cell.car is car
    assert cell.cdr is cdr
    assert type(cell) is SExpression


@given(simple_objects, simple_objects)
def test_cons_onto_noncons(car, cdr):
    cell = cons(car, cdr)

    assert cell.car is car
    assert cell.cdr is cdr
    assert type(cell) is ConsCell
