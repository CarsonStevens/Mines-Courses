#!/usr/bin/env python3
"""
This script extracts docstrings from modules and generates the appropriate
files. It is intended to be run by the instructor before distributing the
starter code.

It really has no reason to exist (pytest supports running from the modules
directly) other than:

1. Students tend to be silly sometimes and remove their own docstrings (or,
   type above them!) and loose their unit tests
2. I can grade your answers against the extracted docstrings
"""
import slyther.types
import slyther.parser
import slyther.interpreter
import slyther.builtins
import slyther.evaluator


def trim(docstring):
    """
    Normalize indentation on a docstring, per PEP-257.
    """
    if not docstring:
        return ''

    # Convert tabs to spaces (following the normal Python rules)
    # and split into a list of lines:
    lines = docstring.expandtabs().splitlines()

    # Determine minimum indentation (first line doesn't count):
    try:
        indent = min(len(l) - len(l.lstrip()) for l in lines[1:] if l)
    except ValueError:
        indent = 0

    # Remove indentation (first line is special):
    trimmed = [lines[0].strip()]
    for line in lines[1:]:
        trimmed.append(line[indent:].rstrip())

    # Strip off trailing and leading blank lines:
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    while trimmed and not trimmed[0]:
        trimmed.pop(0)

    return '\n'.join(trimmed) + '\n'


deliverables = {
    'd1': [
        slyther.types.ConsCell,
        slyther.types.ConsCell.__eq__,
        slyther.types.ConsCell.__repr__,
        slyther.types.ConsList,
        slyther.types.ConsList.__init__,
        slyther.types.ConsList.from_iterable,
        slyther.types.ConsList.__getitem__,
        slyther.types.ConsList.cells,
        slyther.types.ConsList.__len__,
        slyther.types.ConsList.__contains__,
        slyther.types.ConsList.__reversed__,
        slyther.types.ConsList.__eq__,
        slyther.types.ConsList.__repr__,
        slyther.types.SExpression,
        slyther.types.cons,
        slyther.types.LexicalVarStorage,
        slyther.types.LexicalVarStorage.fork,
        slyther.types.LexicalVarStorage.put,
        slyther.types.LexicalVarStorage.__getitem__,
    ],
    'd2': [
        slyther.parser.lex,
        slyther.parser.parse,
        slyther.parser.parse_strlit,
        slyther.parser,
    ],
    'd3': [
        slyther.evaluator,
        slyther.evaluator.lisp_eval,
        slyther.types.UserFunction,
        slyther.types.UserFunction.__init__,
        slyther.types.UserFunction.__call__,
        slyther.types.UserFunction.__repr__,
        slyther.builtins,
        slyther.builtins.add,
        slyther.builtins.sub,
        slyther.builtins.mul,
        slyther.builtins.div,
        slyther.builtins.floordiv,
        slyther.builtins.list_,
        slyther.builtins.car,
        slyther.builtins.cdr,
    ],
    'd4': [
        slyther.builtins.define,
        slyther.builtins.lambda_func,
        slyther.builtins.let,
        slyther.builtins.if_expr,
        slyther.builtins.cond,
        slyther.builtins.and_,
        slyther.builtins.or_,
        slyther.builtins.setbang,
        slyther.builtins.eval_,
        slyther.builtins.parse_string,
    ],
}


def fullname(obj):
    if hasattr(obj, '__qualname__'):
        return obj.__module__ + '.' + obj.__qualname__
    return obj.__name__


if __name__ == '__main__':
    for d, items in deliverables.items():
        for item in items:
            if hasattr(item, "__doc__") and item.__doc__:
                f = open("tests/{}/test__docstring_{}.txt".format(
                    d, fullname(item).replace('/', 'div')), "w")
                mod = getattr(item, "__module__", getattr(item, "__name__"))
                f.write('>>> from {} import *\n\n'.format(mod))
                f.write(trim(item.__doc__))
                f.close()
