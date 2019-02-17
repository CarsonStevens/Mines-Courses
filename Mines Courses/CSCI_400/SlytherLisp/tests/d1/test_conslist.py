import pytest
from hypothesis import given
import hypothesis.strategies as st
from slyther.types import ConsList, NIL

simple_objects = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(),
    st.text())

simple_objects_nonone = st.one_of(
    st.booleans(),
    st.integers(),
    st.text())


@given(simple_objects, simple_objects_nonone)
def test_init_fail(car, cdr):
    with pytest.raises(TypeError):
        ConsList(car, cdr)


@given(simple_objects)
def test_init_nil(car):
    cell_a = ConsList(car, NIL)
    cell_b = ConsList(car)
    assert cell_a.car is car
    assert cell_b.car is car
    assert cell_a.cdr is NIL
    assert cell_b.cdr is NIL


def test_init_nil_nil():
    cell_a = ConsList(NIL, NIL)
    cell_b = ConsList(NIL)
    assert cell_a.car is NIL
    assert cell_b.car is NIL
    assert cell_a.cdr is NIL
    assert cell_b.cdr is NIL


@given(simple_objects, simple_objects)
def test_twolist_init(car, cadr):
    last = ConsList(cadr)
    front = ConsList(car, last)
    assert front.car is car
    assert front.cdr is last
    assert front.cdr.car is cadr
    assert front.cdr.cdr is NIL


@given(simple_objects)
def test_nestlist_init(car):
    inner = ConsList(car)
    outer = ConsList(inner)
    assert outer.car is inner
    assert inner.car is car
    assert outer.cdr is NIL
    assert inner.cdr is NIL


@given(st.lists(simple_objects))
def test_fromiterable(lst):
    def gen(lst):
        yield from lst

    for typ in list, iter, gen:
        cell = ConsList.from_iterable(typ(lst))
        for itm in lst:
            assert cell.car is itm
            cell = cell.cdr
        assert cell is NIL


@given(st.one_of(st.sets(simple_objects), st.frozensets(simple_objects)))
def test_fromiterable_set(s):
    cell = ConsList.from_iterable(s)
    seen = set()
    while cell is not NIL:
        assert cell.car not in seen
        seen.add(cell.car)
        cell = cell.cdr
    assert seen == s


def test_fromiterable_big():
    cell = ConsList.from_iterable(range(10000))
    for i in range(10000):
        assert cell.car == i
        cell = cell.cdr
    assert cell is NIL


def test_fromiterable_uses_cls():
    # this test is used to determine if the cls argument to
    # from_iterable is used correctly, or fail if ConsList was
    # hard-coded.

    class ExtType(ConsList):
        pass

    cell = ExtType.from_iterable(range(10))
    for i in range(10):
        assert isinstance(cell, ExtType)
        assert cell.car == i
        cell = cell.cdr
    assert cell is NIL


@given(st.lists(simple_objects))
def test_getitem(lst):
    cell = ConsList.from_iterable(lst)
    for i, itm in enumerate(lst):
        assert cell[i] is lst[i]


@given(st.lists(simple_objects))
def test_iter(lst):
    it = iter(ConsList.from_iterable(lst))
    for itm in lst:
        assert itm is next(it)
    with pytest.raises(StopIteration):
        next(it)


@given(st.lists(simple_objects))
def test_cells(lst):
    it = ConsList.from_iterable(lst).cells()
    for itm in lst:
        cell = next(it)
        assert itm is cell.car
    with pytest.raises(StopIteration):
        next(it)


@given(st.lists(simple_objects))
def test_len(lst):
    cell = ConsList.from_iterable(lst)
    assert len(lst) == len(cell)


def test_len_big():
    # build a list of length 10000
    head = ConsList(NIL)
    c = head
    for _ in range(9999):
        c.cdr = ConsList(NIL)
        c = c.cdr

    # if they implemented recursively, this would most certainly fail
    assert len(head) == 10000


@given(st.lists(simple_objects), simple_objects)
def test_contains(lst, other):
    cell = ConsList.from_iterable(lst)
    if other in lst:
        assert other in cell
    else:
        assert other not in cell


@given(st.lists(simple_objects, min_size=1))
def test_repr(lst):
    r = repr(ConsList.from_iterable(lst))
    assert r.startswith('(list')
    idx = 6
    for item in map(repr, lst):
        assert r[idx - 1] == ' '
        assert r[idx:idx + len(item)] == item
        idx += len(item) + 1
    assert r[idx - 1] == ')'
