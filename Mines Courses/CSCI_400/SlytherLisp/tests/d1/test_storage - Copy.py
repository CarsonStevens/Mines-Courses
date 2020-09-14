import pytest
from hypothesis import given
import hypothesis.strategies as st
from slyther.types import Variable, LexicalVarStorage, NIL

simple_objects = st.one_of(
    st.just(NIL),
    st.none(),
    st.booleans(),
    st.integers(),
    st.text())

variables = st.builds(Variable, simple_objects)
stg_dictionaries = st.dictionaries(st.text(), variables)


@given(stg_dictionaries, stg_dictionaries)
def test_fork(environ, local):
    stg = LexicalVarStorage(environ)
    stg.local = local
    save_local = dict(stg.local)
    save_environ = dict(stg.environ)
    fork = stg.fork()
    assert isinstance(fork, dict)

    # should not modify locals or environ during fork
    assert stg.local == save_local
    assert stg.environ == save_environ

    for k, v in stg.environ.items():
        if k not in stg.local.keys():
            assert fork[k] is v

    for k, v in stg.local.items():
        assert fork[k] is v
