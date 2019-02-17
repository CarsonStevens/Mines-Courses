from slyther.types import (Quoted, NIL, SExpression, ConsList, Symbol,
                           Macro, NilType, LexicalVarStorage)


def lisp_eval(expr, stg: LexicalVarStorage):
    """
    Takes a **single** AST element (such as a SExpression, NIL, or
    number) and evaluates it, returning the result, if any.

    Depending on what type ``expr`` is, the evaluation will be computed
    differently:

    +---------------------+-------------------------------------------------+
    | If ``expr`` is a... | return...                                       |
    +=====================+=================================================+
    | ``NIL``             | ``NIL``                                         |
    +---------------------+-------------------------------------------------+
    | Quoted s-expression | A ``ConsList`` with each of the elements in the |
    |                     | s-expression quoted and then ``lisp_eval``-ed.  |
    +---------------------+-------------------------------------------------+
    | Quoted non-SE       | The element unquoted.                           |
    +---------------------+-------------------------------------------------+
    | Symbol              | The value of the corresponding ``Variable`` in  |
    |                     | the lexical variable storage (``stg``).         |
    +---------------------+-------------------------------------------------+
    | S-Expression        | ``lisp_eval`` the CAR and if that's a...        |
    |                     |                                                 |
    |                     | :Macro:                                         |
    |                     |     call the macro with the unevaluated         |
    |                     |     arguments and ``stg`` and return the        |
    |                     |     ``lisp_eval``-ed result.                    |
    |                     | :Function:                                      |
    |                     |     ``lisp_eval`` each of the arguments and     |
    |                     |     call the function. Return the result.       |
    |                     | :Something else:                                |
    |                     |     Raise a ``TypeError``.                      |
    +---------------------+-------------------------------------------------+
    | Something else      | Return it as is.                                |
    +---------------------+-------------------------------------------------+

    Here is some examples:

    >>> from slyther.types import *
    >>> empty_stg = LexicalVarStorage({})
    >>> lisp_eval(NIL, empty_stg)
    NIL
    >>> lisp_eval(3, empty_stg)
    3
    >>> lisp_eval(Quoted(3), empty_stg)
    3
    >>> some_stg = LexicalVarStorage({
    ...     'x': Variable(2),
    ...     'NIL': Variable(NIL)})
    >>> lisp_eval(Symbol('x'), some_stg)
    2
    >>> lisp_eval(Quoted(Symbol('x')), some_stg)
    x
    >>> lisp_eval(Symbol('NIL'), some_stg)
    NIL
    >>> from slyther.parser import lisp
    >>> def test(code):                 # test function
    ...     return lisp_eval(lisp(code), some_stg)
    >>> test("3")
    3
    >>> test("'(1 2 3)")
    (list 1 2 3)
    >>> l = test("'(x y z)")
    >>> l
    (list x y z)
    >>> type(l.car)
    <class 'slyther.types.Symbol'>
    >>> test("'(x y z (a b c))")
    (list x y z (list a b c))
    >>> test("'((a b (c)) (1 (2) 3))")
    (list (list a b (list c)) (list 1 (list 2) 3))

    Function calls should take *evaluated parameters*, do something
    with them, and return a result (which, unlike macros, ``lisp_eval``
    does not need called on to compute).

    >>> identity = UserFunction(
    ...     params=SExpression(Symbol('x')),
    ...     body=SExpression(Symbol('x')),
    ...     environ=some_stg.fork())
    >>> some_stg.put('identity', identity)
    >>> test("(identity 4)")
    4
    >>> test("(identity ())")
    NIL
    >>> test("(identity NIL)")
    NIL
    >>> test("(identity x)")
    2
    >>> import operator
    >>> some_stg.put('add', BuiltinFunction(operator.add))
    >>> test("(add 10 20)")
    30
    >>> test("(add '10 '20)")
    30
    >>> test('(add "10" "20")')
    "1020"
    >>> test('(add x x)')
    4
    >>> test('((identity add) (identity x) (identity x))')
    4

    Macros should take *unevaluated paramaters* and the storage, do
    something with them, and return an expression *to be evaluated*!

    >>> def testmac(se, stg):
    ...     for i, arg in enumerate(se):
    ...         print(i, arg)
    ...     stg.put('f', se.car)
    ...     return se
    >>> some_stg.put('testmac', BuiltinMacro(testmac))
    >>> test("(testmac add 3 6)")
    0 add
    1 3
    2 6
    9
    >>> type(test('f'))
    <class 'slyther.types.Symbol'>
    >>> test('(f 1 2)')
    Traceback (most recent call last):
        ...
    TypeError: 'Symbol' object is not callable
    >>> test('((identity testmac) (identity add) (identity 1) 2)')
    0 (identity add)
    1 (identity 1)
    2 2
    3

    """
    raise NotImplementedError("Deliverable 3")
