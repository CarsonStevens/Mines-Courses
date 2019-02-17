"""
This module defines parsing utilities in SlytherLisp.

>>> parser = parse(lex('(print "Hello, World!")'))
>>> se = next(parser)
>>> se
(print "Hello, World!")
>>> type(se)
<class 'slyther.types.SExpression'>

"""
import re
from slyther.types import SExpression, Symbol, String, Quoted, NIL

__all__ = ['lex', 'parse', 'lisp', 'parse_strlit', 'ControlToken', 'LParen',
           'RParen', 'Quote']


class ControlToken:
    """
    "Control tokens" are tokens emitted by the lexer, but never appear in the
    resultant abstract syntax tree. The define information for the parser to
    complete the parse. For example, parentheses are emitted by the lexer, but
    are not valid in the abstract syntax tree.

    This is the base class for all control token types. You don't need to
    instantiate it directly, instead, you should instantiate its subclasses.
    """
    instance = None

    def __new__(cls, *args, **kwargs):
        # Control tokens are single instance per class. In other words,
        # ``LParen() is LParen()`` will always be true.
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __repr__(self):
        return self.__class__.__name__


class LParen(ControlToken):
    pass


class RParen(ControlToken):
    pass


class Quote(ControlToken):
    pass


def lex(code):
    r"""
    IMPORTANT: read this entire docstring before implementing this function!
    Please ask for help on Piazza or come to office hours if you don't
    understand something in here.

    This is a *generator function* that splits a piece of code into
    lexical tokens, emitting whole tokens each time one is encountered.

    A lexical token is not just a string: this function should assign a
    type to each piece of data as appropriate.

    For a high level overview, consider this invocation:

    >>> toks = list(lex("(define (do-123 fun) \n (map fun '(1 2 3.0))) \n"))
    >>> toks    # doctest: +NORMALIZE_WHITESPACE
    [LParen, define, LParen, do-123, fun, RParen,
        LParen, map, fun, Quote, LParen, 1, 2, 3.0, RParen, RParen, RParen]

    Let's look at the types of each piece of data here. The ``LParen``,
    ``RParen``, and ``Quote`` bits are *instances of* the subclasses of
    the ``ControlToken`` class above. ``1`` and ``2`` are Python integers,
    and ``3.0`` is a Python ``float``. Finally, everything else here is a
    ``Symbol``.

    >>> print(*(t.__class__.__name__ for t in toks))
    ...                                 # doctest: +NORMALIZE_WHITESPACE
    LParen Symbol LParen Symbol Symbol RParen
    LParen Symbol Symbol Quote LParen int int float RParen RParen RParen

    So what goes into this process? First, we need a pretty good definition
    of each of the tokens in the language. The lexer will emit what matches
    the following:

    :Control Tokens:
        Left parenthesis, right parenthesis, and single quotes.
    :String Literals:
        A double quote (``"``) followed by any (including zero) amount of
        characters which are not a double quote. The double quote
        character ends a string, *unless* a backslash precedes it. These
        string literals should be parsed by the ``parse_strlit`` function
        before they are emitted by the lexer.
    :Integer Literals:
        An optional negative sign, followed by 1 or more digits. There cannot
        be a period following the digits, as it should be parsed as a floating
        point number (see below). This should be emitted as Python's ``int``
        type.
    :Floating Point Literals:
        An optional negative sign, followed by zero or more digits, followed
        by a period (decimal point), followed by zero or more digits. All
        floating point numbers must have at least one digit. These should be
        emitted as Python's ``float`` type.
    :Symbols:
        These are a collection of 1 or more of **any characters**, not
        including single/double quotes, parenthesis, semicolons, or
        whitespace. In addition, symbols cannot start with digits or
        periods.

    The following are ignored, and not omitted by the lexer:

    :Whitespace:
        Groups of whitespace characters in-between tokens.
    :Comments:
        A semicolon, followed by any amount of characters until the end
        of a line.
    :Shebang Lines:
        A line at the top of the file which starts with ``#!``. For example::

            #!/usr/bin/env slyther

        Note that it only a shebang line if it's at the top of the file.
        Elsewhere, this would look like two symbols.

    Anything which does not match above should raise a ``SyntaxError``. Think
    for a second: what might not match the above?

    That's the end of the technical specification... what follows is examples
    which reading may give you an idea on what to do here.

    Strings should be parsed using ``parse_strlit``:

    >>> toks = list(lex(r'(print "hello \"world\"!")'))
    >>> toks
    [LParen, print, "hello \"world\"!", RParen]
    >>> type(toks[1])
    <class 'slyther.types.Symbol'>
    >>> type(toks[2])
    <class 'slyther.types.String'>

    Symbols are defined based on what characters cannot be in them, not which
    can. This means you can have symbols like this:

    >>> list(lex(r'lambda-is-λ ¯\_[ツ]_/¯'))
    [lambda-is-λ, ¯\_[ツ]_/¯]

    Since symbols cannot start with a digit, this function should separate the
    numerical literal from the symbol when things like this happen:

    >>> list(lex('(print 8-dogcows)'))
    [LParen, print, 8, -dogcows, RParen]
    >>> list(lex('(print -8.0-dogcows)'))
    [LParen, print, -8.0, -dogcows, RParen]

    And since symbols can have digits and dots in the middle, make sure these
    are parsed properly:

    >>> list(lex('(print dogcows-8.0)'))
    [LParen, print, dogcows-8.0, RParen]

    Quotes can't occur in the middle of symbols and numbers, so these quotes
    separate the tokens below, even without whitespace:

    >>> list(lex("(print'hello-world'4'8.0)"))
    [LParen, print, Quote, hello-world, Quote, 4, Quote, 8.0, RParen]

    Shebang lines **only at the front of the string, before any whitespace**
    should be ignored, and not emitted as a token:

    >>> list(lex("#!/usr/bin/env slyther\n(print 1)\n"))
    [LParen, print, 1, RParen]
    >>> list(lex("#!slyther\n(print 1)"))
    [LParen, print, 1, RParen]
    >>> list(lex(" #!/usr/bin/env slyther\n(print 1)\n"))
    [#!/usr/bin/env, slyther, LParen, print, 1, RParen]

    Comments start at a semicolon and go until the end of line (or,
    potentially the end of the input string). Note that string literals
    might contain semicolons: these don't start a comment, beware.
    Comments should not be emitted from the lexer.

    >>> list(lex('(here-comes; a comment!\n "no comment ; here";comment()\n)'))
    [LParen, here-comes, "no comment ; here", RParen]
    >>> list(lex('; commments can contain ; inside them\n'))
    []

    When an error is encountered, ``SyntaxError`` should be raised:

    >>> list(lex(r'(print "Hello, World!\")'))        # unclosed string
    Traceback (most recent call last):
        ...
    SyntaxError: malformed tokens in input
    >>> list(lex(r'.symbol'))        # symbols cannot start with period
    Traceback (most recent call last):
        ...
    SyntaxError: malformed tokens in input

    Don't worry about handling unmatched parenthesis or single quotes in an
    invalid position. This will be the parser's job! In other words, the
    lexer should be no smarter than it needs to be to do its job.

    >>> list(lex("((("))
    [LParen, LParen, LParen]
    >>> list(lex("(')"))
    [LParen, Quote, RParen]
    >>> list(lex("'"))
    [Quote]
    """
    raise NotImplementedError("Deliverable 2")


def parse_strlit(tok):
    r"""
    This function is a helper method for ``lex``. It takes a string literal,
    raw, just like it is in the source code, and converts it to a
    ``slyther.types.String``.

    It should support the following translations:

    +-----------------+--------------------+
    | Escape Sequence | Resulting Output   |
    +=================+====================+
    | ``\0``          | ASCII Value 0      |
    +-----------------+--------------------+
    | ``\a``          | ASCII Value 7      |
    +-----------------+--------------------+
    | ``\b``          | ASCII Value 8      |
    +-----------------+--------------------+
    | ``\e``          | ASCII Value 27     |
    +-----------------+--------------------+
    | ``\f``          | ASCII Value 12     |
    +-----------------+--------------------+
    | ``\n``          | ASCII Value 10     |
    +-----------------+--------------------+
    | ``\r``          | ASCII Value 13     |
    +-----------------+--------------------+
    | ``\t``          | ASCII Value 9      |
    +-----------------+--------------------+
    | ``\v``          | ASCII Value 11     |
    +-----------------+--------------------+
    | ``\"``          | ASCII Value 34     |
    +-----------------+--------------------+
    | ``\\``          | ASCII Value 92     |
    +-----------------+--------------------+
    | ``\x##``        | Hex value ``##``   |
    +-----------------+--------------------+
    | ``\0##``        | Octal value ``##`` |
    +-----------------+--------------------+

    >>> parse_strlit(r'"\0"')
    "\x00"
    >>> parse_strlit(r'"\e"')
    "\x1b"
    >>> parse_strlit(r'"\x41"')
    "A"
    >>> parse_strlit(r'"\x53\x6c\x79\x74\x68\x65\x72\x4C\x69\x73\x70"')
    "SlytherLisp"
    >>> parse_strlit(r'"this is my\" fancy\n\estring literal"')
    "this is my\" fancy\n\x1bstring literal"

    Patterns which do not match the translations should be left alone:

    >>> parse_strlit(r'"\c\d\xzz"')
    "\\c\\d\\xzz"

    Octal values should only expand when octal digits (0-7) are used:

    >>> parse_strlit(r'"\077"')
    "?"
    >>> parse_strlit(r'"\088"') # a \0, followed by two 8's
    "\x0088"

    Even though this is similar to Python's string literal format,
    you should not use any of Python's string literal processing
    utilities for this: tl;dr do it yourself.
    """
    raise NotImplementedError("Deliverable 2")


def parse(tokens):
    r"""
    This *generator function* takes a generator object from the ``lex``
    function and generates AST elements.

    These are some convenient constants to make the examples more readable.
    Make a note of them while you read the examples.

    >>> from slyther.types import (Symbol as s, Quoted, SExpression)
    >>> lp = LParen()
    >>> rp = RParen()
    >>> q = Quote()

    Here is a simple example:

    >>> tokens = [lp, s('define'), lp, s('do-123'), s('fun'), rp,
    ...             lp, s('map'), s('fun'), q, lp, 1, 2, 3, rp, rp, rp]
    >>> parser = parse(iter(tokens))
    >>> se = next(parser)
    >>> se                  # what you see below is from __repr__
    (define (do-123 fun) (map fun '(1 2 3)))
    >>> type(se)
    <class 'slyther.types.SExpression'>
    >>> type(se.car)
    <class 'slyther.types.Symbol'>

    Let's grab out that quoted list and take a look at it.

    >>> quoted_list = se.cdr.cdr.car.cdr.cdr.car
    >>> quoted_list
    '(1 2 3)
    >>> type(quoted_list)
    <class 'slyther.types.Quoted'>
    >>> type(quoted_list.elem)
    <class 'slyther.types.SExpression'>

    It was a quoted s-expression. Those can be constructed like this:

    >>> Quoted(SExpression.from_iterable([1, 2, 3]))
    '(1 2 3)

    Not only can s-expressions be quoted, but practically anything can. In
    addition, things can be quoted multiple times.

    >>> tokens = [q, lp, s('print'), 1, q, -2, q, q, 3.0, rp]
    >>> parser = parse(iter(tokens))
    >>> qse = next(parser)
    >>> qse
    '(print 1 '-2 ''3.0)
    >>> numbers = qse.elem.cdr
    >>> numbers
    (1 '-2 ''3.0)

    See that doubly-quoted three? It was constructed like this:

    >>> three = Quoted(Quoted(3.0))
    >>> three
    ''3.0
    >>> three.elem
    '3.0
    >>> three.elem.elem
    3.0

    You could imagine something similar for triply-quoted, quad-quoted, or
    even n-quoted. Your parser should be able to handle any amount of quotes.

    When the input token stream is starting to form a valid parse, but ends
    before the parse is complete, a ``SyntaxError`` should be raised:

    >>> tokens = [lp, rp, lp, s('print')]
    >>> parser = parse(iter(tokens))
    >>> next(parser)
    NIL
    >>> next(parser)
    Traceback (most recent call last):
        ...
    SyntaxError: incomplete parse

    >>> tokens = [lp, rp, q]
    >>> parser = parse(iter(tokens))
    >>> next(parser)
    NIL
    >>> next(parser)
    Traceback (most recent call last):
        ...
    SyntaxError: incomplete parse

    Notice in both of the previous examples, we got complete elements as soon
    as they were fully formed, and the syntax error did not come until there
    was an error.

    When there's too many closing parens, you should raise another error:

    >>> tokens = [lp, rp, s('print'), rp]
    >>> parser = parse(iter(tokens))
    >>> next(parser)
    NIL
    >>> next(parser)
    print
    >>> next(parser)
    Traceback (most recent call last):
        ...
    SyntaxError: too many closing parens

    Finally, right parenthesis cannot be quoted. We have one more type of
    error this can raise:

    >>> tokens = [q, 1, lp, s('print'), q, rp]
    >>> parser = parse(iter(tokens))
    >>> next(parser)
    '1
    >>> next(parser)
    Traceback (most recent call last):
        ...
    SyntaxError: invalid quotation
    """
    raise NotImplementedError("Deliverable 2")


def lisp(code: str):
    """
    Helper function (mostly for test functions, etc.) to lex and parse
    code representing a single AST element.
    """
    return next(parse(lex(code)))
