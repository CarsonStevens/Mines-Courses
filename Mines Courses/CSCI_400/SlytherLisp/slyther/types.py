import collections.abc as abc
from typing import Dict
from functools import partial, update_wrapper


class ConsCell:
    """
    A simple cons cell data structure:

    >>> cell = ConsCell(1, 2)
    >>> cell.car
    1
    >>> cell.cdr
    2
    >>> cell.car = 4
    >>> cell.car
    4
    """
# Falsifying example: test_cons_init(car=None, cdr=None)
    def __init__(self, car, cdr):
        if car is not None: self.car = car
        else: self.car = NIL

        if cdr is not None: self.cdr = cdr
        else: self.cdr = NIL

        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_cons_eq_self(car=None, cdr=None)
# Falsifying example: test_cons_eq_noncell(car=None, cdr=None, other=None)
    def __eq__(self, other):
        """
        Two cons cells are equal if each of their ``car`` and
        ``cdr`` are equal:

        >>> a = ConsCell(1, 2)
        >>> b = ConsCell(2, 1)
        >>> c = ConsCell(1, 1)
        >>> d = ConsCell(1, 2)
        >>> a == a
        True
        >>> a == b
        False
        >>> b == c
        False
        >>> b == d
        False
        >>> a == d
        True

        Should return ``False`` if ``other`` is not an instance of a
        ``ConsCell``.
        """
        if (isinstance(other, ConsCell)):
            if(self.car == other.car and self.cdr == other.cdr):
                return True

        return False

        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_cons_repr(car=None, cdr=None)
    def __repr__(self):
        """
        A cons cell should ``repr`` itself in a format that would
        be parsable and evaluable to our language.

        >>> ConsCell(1, 2)
        (cons 1 2)
        >>> ConsCell(ConsCell(2, 1), 1)
        (cons (cons 2 1) 1)

        .. hint::

            The string formatting specifier ``!r`` will get you the
            ``repr`` of an object.
        """
        result = "(cons "
        # result += self.car
        # if(isinstance(self.cdr, ConsCell)):
        #     result += __repr_(self.cdr)
        # else:
        result += str(self.car) + " " + str(self.cdr) + ")"

        return result
        raise NotImplementedError("Deliverable 1")


class ConsList(ConsCell, abc.Sequence):
    """
    A ``ConsList`` inherits from a ``ConsCell``, but the ``cdr`` must
    be a ``ConsList`` or any structure which inherits from that.

    >>> cell = ConsList(1, ConsList(2, NIL))
    >>> cell.car
    1
    >>> cell.cdr.car
    2
    >>> cell = ConsList(1, 2)
    Traceback (most recent call last):
        ...
    TypeError: cdr must be a ConsList
    """


# Falsifying example: test_init_fail(car=None, cdr=False)
# Falsifying example: test_init_nil(car=None)
# Falsifying example: test_twolist_init(car=None, cadr=None)
# Falsifying example: test_nestlist_init(car=None)
    def __init__(self, car, cdr=None):
        """
        If the ``cdr`` was not provided, assume to be ``NIL``.

        >>> cell = ConsList(1)
        >>> cell.cdr
        NIL
        """
        self.car = car
        if(cdr is None): self.cdr = NIL
        else: self.cdr = cdr

        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_fromiterable(lst=[])
# Falsifying example: test_fromiterable_set(s={None})
    @classmethod
    def from_iterable(cls, it):
        """
        Create an instance of ``cls`` from an iterable (anything that can go
        on the right hand side of a ``for`` loop).

        >>> lst = ConsList.from_iterable(iter(range(3)))
        >>> [lst.car, lst.cdr.car, lst.cdr.cdr.car, lst.cdr.cdr.cdr]
        [0, 1, 2, NIL]
        >>> ConsList.from_iterable([])
        NIL

        Note: Your implementation is subject to the following constraints:

        :Time complexity: O(n), where n is length of ``it``
        :Space complexity: O(n) ``ConsList`` objects,
                           O(1) everything else (including stack frames!)
        """

        # try:
        #     #Need to return list of ConCells
        #     for i, value in enumerate(it):
        #         cell_list = ConsList(value)
        #         try:
        #             cell_list.cdr = cls[i-1]
        #         except:
        #             cell_list.cdr = NIL
        #
        #         cls.append(cell_list)
        # except:
        #     return NIL

        if(next(it, None)):
            cls = NIL
            return cls

        try:
            for i, value in enumerate(reversed(it)):
                if i == 0:
                    cls = ConsList(value)
                else:
                    cls = ConsList(value, cls)
        except:
            cls = NIL

        return cls


        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_getitem(lst=[None])
    def __getitem__(self, idx):
        """
        Get item at index ``idx``:

        >>> lst = [1, 1, 2, 3, 5, 8]
        >>> clst = ConsList.from_iterable(lst)
        >>> [lst[i] == clst[i] for i in range(len(lst))]
        [True, True, True, True, True, True]
        """
        item = self
        for i in range(idx):
            if i == idx:
                return item.car
            if item.cdr == NIL:
                break;
            #item = ConsList(item.cdr.car, item.cdr)
            item = item.cdr


        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_iter(lst=[])
    def __iter__(self):
        """
        Iterate over the ``car`` of each cell:

        >>> lst = [1, 1, 2, 3, 5, 8]
        >>> for itm in ConsList.from_iterable(lst):
        ...     print(itm)
        1
        1
        2
        3
        5
        8

        Note: Your implementation is subject to the following constraints:

        :Time complexity: O(1) for each yield
        :Space complexity: O(1)
        """
        item = self
        while(item.cdr != NIL):
            yield item.car
            #item = ConsList(item.cdr.car, item.cdr)
            item = item.cdr

        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_cells(lst=[])
    def cells(self):
        """
        Iterate over each cell (rather that the ``car`` of each):

        >>> lst = [1, 1, 2, 3, 5, 8]
        >>> for cell in ConsList.from_iterable(lst).cells():
        ...     print(cell.car)
        1
        1
        2
        3
        5
        8

        Note: Your implementation is subject to the following constraints:

        :Time complexity: O(1) for each yield
        :Space complexity: O(1)
        """
        item = self
        while(item.cdr != NIL):
            yield item
            #item = ConsList(item.cdr.car, item.cdr)
            item = item.cdr
        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_len(lst=[])
    def __len__(self):
        """
        Return the number of elements in the list:

        >>> lst = [1, 1, 2, 3, 5, 8]
        >>> len(ConsList.from_iterable(lst))
        6

        Note: Your implementation is subject to the following constraints:

        :Time complexity: O(n), where n is the length of the list.
        :Space complexity: O(1)
        """
        item = self
        length = 0
        while(item.cdr != NIL):
            length += 1
            #item = ConsList(item.cdr.car, item.cdr)
            item = item.cdr

        return length
        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_contains(lst=[], other=None)
    def __contains__(self, p):
        """
        Return ``True`` if the list contains an element ``p``, ``False``
        otherwise. A list is said to contain an element ``p`` iff there is any
        element ``a`` in the list such that ``a == p``.

        >>> lst = [1, 1, 2, 3, 5, 8]
        >>> clst = ConsList.from_iterable(lst)
        >>> 1 in clst
        True
        >>> 3 in clst
        True
        >>> 8 in clst
        True
        >>> NIL in clst
        False
        >>> 9 in clst
        False

        Note: Your implementation is subject to the following constraints:

        :Time complexity: O(n), where n is the length of the list.
        :Space complexity: O(1)
        """
        item = self

        while(item.cdr != NIL):
            if item.car == p:
                return True
            #item = ConsList(item.cdr.car, item.cdr)
            item = item.cdr
        return False

        raise NotImplementedError("Deliverable 1")

# CHECK
    def __reversed__(self):
        """
        Iterate over the elements of our list, reversed.

        >>> lst = [1, 1, 2, 3, 5, 8]
        >>> clst = ConsList.from_iterable(lst)
        >>> for x in reversed(clst):
        ...     print(x)
        8
        5
        3
        2
        1
        1

        Note: Your implementation is subject to the following constraints:

        :Time complexity: O(n), where n is the length of the list.
        :Space complexity: O(n)
        """

        for element in self[::-1]:
            yield element.car


        raise NotImplementedError("Deliverable 1")

    def __bool__(self):
        """ NilType overrides this to be ``False``. """
        return True

# TODO (1/2)
    def __eq__(self, other):
        """
        Test if two lists have the same elements in the same order.

        >>> l1, l2 = map(
        ...     ConsList.from_iterable,
        ...     ([1, 2, 10, 3, 4, 7], [1, 2, 10, 3, 4, 7]))
        >>> l1 == l2
        True
        >>> l1, l2 = map(
        ...     ConsList.from_iterable,
        ...     ([1, 2, 10, 4, 3, 7], [1, 2, 10, 3, 4, 7]))
        >>> l1 == l2
        False
        >>> l1, l2 = map(
        ...     ConsList.from_iterable,
        ...     ([1, 2, 10, 3, 4, 7], [1, 2, 10, 3, 4, 7, 1]))
        >>> l1 == l2
        False
        >>> l2 == l1
        False
        >>> l1 = NIL
        >>> l1 == l2
        False
        >>> l2 == l1
        False
        >>> SExpression.from_iterable(l2) == NIL
        False
        """
        #Check if lengths of lists are the same
        if(len(self.cls) != len(other.cls)): return False
        #Might need to check if list or sexpression here
        else:
            for idx, element in enumerate(self.cls):
                if(self[idx].car != other[idx]):
                    return False

        return True
        raise NotImplementedError("Deliverable 1")

# Falsifying example: test_repr(lst=[None])
    def __repr__(self):
        """
        Represent ourselves in a format evaluable in SlytherLisp.

        >>> ConsList.from_iterable([1, 2, 3])
        (list 1 2 3)
        """
        result = "(list"
        item = self
        while(item.cdr != NIL):
            result += " " + str(item.car)
            #item = ConsList(item.cdr.car, item.cdr)
            item = item.cdr
        result += ")"

        return result
        raise NotImplementedError("Deliverable 1")


class NilType(ConsList):
    """
    The type for the global ``NIL`` object.
    """
    def __new__(cls):
        """
        If already constructed, don't make another. Just
        return our already existing instance.
        """
        if 'NIL' in globals().keys():
            return NIL
        return super().__new__(cls)

    def __init__(self):
        """
        The ``car`` and ``cdr`` of ``NIL`` are ``NIL``.
        """
        self.car = self
        self.cdr = self

    def __bool__(self):
        """
        Empty lists are implicitly falsy.
        """
        return False

    def __eq__(self, other):
        """
        There is only one ``NIL`` instance anyway...
        """
        return self is other

    def __repr__(self):
        """
        Represent ourselves in SlytherLisp evaluable format
        """
        return 'NIL'


NIL = NilType()


class Boolean:
    """
    Type for a boolean with SlytherLisp evaluable representation.
    """
    # too bad we can't subclass bool...
    class LispTrue:
        def __bool__(self):
            return True

        def __repr__(self):
            return '#t'

    class LispFalse:
        def __bool__(self):
            return False

        def __repr__(self):
            return '#f'

    t_instance = LispTrue()
    f_instance = LispFalse()

    def __new__(cls, v=False):
        """
        There shall only be one true, and one false!
        """
        if v:
            return Boolean.t_instance
        return Boolean.f_instance


class SExpression(ConsList):
    """
    ConsList which we use to store s-expressions. Has an alternate
    representation.

    >>> SExpression(4)
    (4)
    """
    def __repr__(self):
        return '({})'.format(' '.join(map(repr, self)))

# TODO
# Falsifying example: test_cons_onto_nil(car=None)
# Falsifying example: test_cons_onto_conscell(car=None, cadr=None, cddr=None)
# Falsifying example: test_cons_onto_conslist(car=None, cadr=None)
# Falsifying example: test_cons_onto_noncons(car=None, cdr=None)
def cons(car, cdr) -> ConsCell:
    """
    Factory for cons cell like things. Tries to make a ``ConsList`` or
    ``SExpression`` if it can (if ``cdr`` is...), otherwise makes a
    plain old ``ConsCell``.

    >>> cons(5, ConsList(4, NIL))
    (list 5 4)
    >>> cons(5, NIL)
    (list 5)
    >>> cons(5, 4)
    (cons 5 4)
    >>> cons(5, SExpression(4, NIL))
    (5 4)
    """
    if isinstance(ConsList, cdr) or cdr == NIL:
        return '(list ' + str(car) + ')'
    raise NotImplementedError("Deliverable 1")


class Variable:
    """
    A simple wrapper to reference an object. The reference may change using the
    ``set`` method.

    The reason for this is so that ``(set! ...)`` works, even in different
    environments. Also lets Python's garbage collection do the dirty work for
    us.

    Note: ``Variable`` will never appear in an abstract syntax tree. Its sole
    purpose is to be used with the ``LexicalVariableStorage``.
    """
    def __init__(self, value):
        self.set(value)

    def set(self, value):
        self.value = value


class LexicalVarStorage:
    """
    Storage for lexically scoped variables. Has two parts:

    * An ``environ`` part: a dictionary of the containing
      environment (closure).
    * A ``local`` part: a dictionary of the local variables
      in the function.
    """
    def __init__(self, environ: Dict[str, Variable]):
        self.environ = environ
        self.local = {}

# TODO
# Falsifying example: test_fork(environ={}, local={})
    def fork(self) -> Dict[str, Variable]:
        """
        Return the union of the ``local`` part and the ``environ``
        part. Should not modify either part.

        >>> environ = {k: Variable(v) for k, v in (('x', 10), ('y', 11))}
        >>> stg = LexicalVarStorage(environ)
        >>> stg.put('y', 12)
        >>> stg.put('z', 13)
        >>> for k, v in stg.fork().items():
        ...     print(k, v.value)
        x 10
        y 12
        z 13
        """
        raise NotImplementedError("Deliverable 1")

    def put(self, name: str, value) -> None:
        """
        Put a **new** variable in the local environment, giving
        it a value ``value``.
        """
        self.local[name] = Variable(value)

# TODO
    def __getitem__(self, key: str) -> Variable:
        """
        Return a Variable object, first checking the local
        environment, then checking the containing environment,
        otherwise raising a ``KeyError``.

        >>> environ = {k: Variable(v) for k, v in (('x', 10), ('y', 11))}
        >>> stg = LexicalVarStorage(environ)
        >>> stg.put('y', 12)
        >>> stg.put('z', 13)
        >>> stg['x'].value
        10
        >>> stg['y'].value
        12
        >>> stg['z'].value
        13
        >>> stg['x'].set(11)
        >>> stg['foo'].value
        Traceback (most recent call last):
            ...
        KeyError: "Undefined variable 'foo'"
        >>> stg['bar'].set(10)
        Traceback (most recent call last):
            ...
        KeyError: "Undefined variable 'bar'"
        """
        raise NotImplementedError("Deliverable 1")


class Quoted:
    """
    A simple wrapper for a quoted element in the abstract syntax tree.
    """
    def __init__(self, elem):
        self.elem = elem

    def __repr__(self):
        return "'{!r}".format(self.elem)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.elem == other.elem


class Symbol(str):
    """
    A type for symbols, like a ``str``, but alternate representation.
    """
    def __repr__(self):
        return str(self)


class String(str):
    """
    A type for SlytherLisp strings, like a ``str``, but alternate
    representation: always use double quotes since SlytherLisp only
    allows double quoted strings.
    """
    def __repr__(self):
        r = super().__repr__()
        if r.startswith("'"):
            return '"{}"'.format(r[1:-1].replace('"', '\\"')
                                        .replace("\\'", "'"))
        return r


class Function(abc.Callable):
    """
    Base class for user and builtin functions. No implementation needed.
    """


class UserFunction(Function):
    """
    Type for user defined functions.

    * ``params`` is an s-expression of the parameters, like so:
      (a b c)
    * ``body`` is an SExpression with the body of the function. The
      result of the last element in the body should be returned when
      the function is called.
    * ``environ`` is a dictionary created by calling ``.fork()`` on a
      ``LexicalVarStorage`` when the function was created.

    """
    def __init__(self, params: SExpression, body: SExpression, environ: dict):
        """
        >>> from slyther.parser import lisp
        >>> f = UserFunction(
        ...     params=lisp('(a b c)'),
        ...     body=lisp('((print a b) (print c))'),
        ...     environ={})
        >>> f
        (lambda (a b c) (print a b) (print c))
        >>> f.params
        (a b c)
        >>> f.body
        ((print a b) (print c))
        >>> f.environ
        {}
        """
        raise NotImplementedError("Deliverable 3")

    def __call__(self, *args):
        """
        Call the function with arguments ``args``.

        Make use of ``lisp_eval``. Note that a fully working ``lisp_eval``
        implementation will require this function to work properly, and this
        will require a working ``lisp_eval`` to work properly, so you must
        write both before you can test it.

        Warning: Do not make any attempt to modify ``environ`` here. That is
        not how lexical scoping works. Instead, construct a new
        ``LexicalVarStorage`` from the existing environ.
        """
        # avoid circular imports
        from slyther.evaluator import lisp_eval

        raise NotImplementedError("Deliverable 3")

    def __repr__(self):
        """
        Represent in self-evaluable form.
        """
        return "(lambda ({}) {})".format(
            ' '.join(self.params),
            ' '.join(repr(x) for x in self.body))


class Macro(abc.Callable):
    """
    Base class for all macros. No implementation needed.
    """


class BuiltinCallable(abc.Callable):
    """
    Base class for builtin callables (functions and macros)
    """
    py_translations = {
        bool: Boolean,
        str: String,
        list: ConsList.from_iterable,
        tuple: ConsList.from_iterable,
    }

    def __new__(cls, arg=None, name=None):
        if isinstance(arg, str):
            return partial(cls, name=arg)
        obj = super().__new__(cls)
        obj.func = arg
        update_wrapper(obj, obj.func)
        obj.__name__ = name or obj.func.__name__
        return obj

    def __call__(self, *args, **kwargs):
        # Note: this function was provided for you in the starter code
        # and you do NOT need to change it.
        result = self.func(*args, **kwargs)
        if result is None:
            return NIL
        if type(result) in self.py_translations.keys():
            return self.py_translations[type(result)](result)
        return result


class BuiltinFunction(BuiltinCallable, Function):
    """
    Builtin functions have this type. Unlike macros, functions cannot
    return s-expressions, and they should be downgraded to cons lists.
    """

    py_translations = dict(BuiltinCallable.py_translations)
    py_translations.update({SExpression: ConsList.from_iterable})


class BuiltinMacro(BuiltinCallable, Macro):
    """
    Builtin macros have this type. No implementation needed.
    """
