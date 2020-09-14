import operator
from slyther.types import (BuiltinFunction, UserFunction, LexicalVarStorage,
                           Variable, Boolean, NIL)
from slyther.parser import lisp
from slyther.evaluator import lisp_eval

collected = {}


@BuiltinFunction
def collect(name, value):
    collected[name] = value
    return value


stg = LexicalVarStorage({
    'collect': Variable(collect),
    'add': Variable(BuiltinFunction(operator.add)),
    '#t': Variable(Boolean(True)),
    '#f': Variable(Boolean(False)),
    'NIL': Variable(NIL)})


def test_function_args():
    environ = stg.fork()
    func = UserFunction(
        params=lisp('(x y z)'),
        body=lisp('''((collect 'z z)
                      (collect 'x x)
                      (collect 'y y)
                      ;; return NIL so that TCO can be implemented
                      ;; without breaking this test
                      NIL)'''),
        environ=environ.copy())

    assert environ == func.environ

    func(1, 2, 3)

    # If this fails, then calling the function modified it's environ
    # dictionary. This should NEVER happen.
    assert environ == func.environ

    # look at captures
    assert collected['x'] == 1
    assert collected['y'] == 2
    assert collected['z'] == 3


def test_function_return():
    environ = stg.fork()
    func = UserFunction(
        params=lisp('(a b)'),
        body=lisp('''((collect 'add-result (add a b))
                      (add a b))'''),
        environ=environ.copy())

    assert environ == func.environ

    stg.put('func', func)
    result = lisp_eval(lisp('(func 10 50)'), stg)

    # If this fails, then calling the function modified it's environ
    # dictionary. This should NEVER happen.
    assert environ == func.environ

    assert collected['add-result'] == 60
    assert result == 60
