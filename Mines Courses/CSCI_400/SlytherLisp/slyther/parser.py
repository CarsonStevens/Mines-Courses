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
    r"""
    patterns = r'''
        (^\\#\![^\n]*\n)                             # shebang line at start 1
        | (\\()                                  # lparen 2
        | (\\))                                   # rparen 3
        | (\\')                                  # quote 4
        | (\s)                                   # whitespace 5
        | ("(?:[^"\\]|\\.)*")                       # string 6
        | ([+-]?(?:\d+\.\d*)|([+-]?\d*\.\d+)       # float 7
        | (?:[-+]?[0-9]+)                           # int 8
        | ([^\s\d\.\'"\(\)\;][^\s\'"\(\);]*)        # symbol 9
        | (;.*$)                                     # comments 10
        | (.)                                       # errors 11
    '''
    p = re.compile(patterns, re.VERBOSE)
    for pattern in p.finditer(code):
        # Ignore group 1 (shebang lines)
        if pattern.group(1):
            pass
        # If a Control Token, find which one
        elif pattern.group(2):
            yield LParen()
        elif pattern.group(3):
            yield RParen()
        elif pattern.group(4):
            yield Quote()
        # Whitespace
        elif pattern.group(5):
            pass
        elif pattern.group(6):
            yield parse_strlit(pattern[0])
        elif pattern.group(7):
            try:
                yield float(pattern[0])
            except ValueError:
                print("Value Error with value: ", pattern[0])
        elif pattern.group(8):
            try:
                yield int(pattern[0])
            except ValueError:
                print("Value Error with value: ", pattern[0])
        elif pattern.group(9):
            yield Symbol(pattern[0])
        elif pattern.group(10):
            pass
        else:
            raise SyntaxError("malformed tokens in input")
    """                                                     # Pattern    |Index
    patterns = [
        re.compile(r'\('),                                  # LParen       | 0
        re.compile(r'\)'),                                  # RParen       | 1
        re.compile(r'\''),                                  # Quote        | 2
        re.compile(r'\s+'),                                 # Whitespace   | 3
        re.compile(r'"([^"\\]|\\.)*"'),                     # StringLit    | 4
        re.compile(r'(-?\d+\.\d*)|(-?\d*\.\d+)'),           # FloatLit     | 5
        re.compile(r'[-+]?[\d]+'),                          # IntegerLit   | 6
        re.compile(r'^#.*?[$\n]'),                          # Shebang      | 7
        re.compile(r'[^\s\d\.\'"\(\)\;][^\s\'"\(\);]*'),    # Symbol       | 8
        re.compile(r';.*$', re.MULTILINE)                   # Comments     | 9
    ]

    # To keep track when in code the last match was yielded from
    index = 0

    """ D3 Updated version
    Go through each part of the input code (index update through match.end() to
    find where the last match ended). if yielded, break from for loop to
    restart patterns at new updated index while there is still input left.
    """

    while index < len(code):

        # Compare to each of the patterns (pattern = index #)
        for pattern in patterns:

            # Get the match to compare
            match = patterns[pattern].match(code, index)
            if match is not None:
                # LParen Pattern
                if pattern == 0:
                    yield LParen()
                    index = match.end()
                    break

                # RParen
                elif pattern == 1:
                    yield RParen()
                    index = match.end()
                    break

                # Quote
                elif pattern == 2:
                    yield Quote()
                    index = match.end()
                    break

                # Whitespace (ignore, but update index)
                elif pattern == 3:
                    index = match.end()
                    break

                # String Literal
                elif pattern == 4:
                    yield parse_strlit(match.group(0))
                    index = match.end()
                    break

                # Floating Point Literal
                elif pattern == 5:
                    yield float(match.group(0))
                    index = match.end()
                    break

                # Integer Literal
                elif pattern == 6:
                    yield int(match.group(0))
                    index = match.end()
                    break

                # Shebang Line (ignore, but update index)
                elif pattern == 7:
                    index = match.end()
                    break

                # Symbol
                elif pattern == 8:
                    yield Symbol(match.group(0))
                    index = match.end()
                    break
                # Comments (ignore, but update index)
                elif pattern == 9:
                    index = match.end()
                    break

            # No match means an invalid sequence that wasn't caught in regex
            else:
                raise SyntaxError("malformed tokens in input")

    r""" CHANGED TO ABOVE AT D3, D2 BELOW WORKS: minus float/int
    # Error in pattern group(4). Everything interpreted as float; failed INT
    # Conversion
    
    patterns = r'''
        (^\#\![^\n]*\n)                                   # Shebang lines(at
                                                          #     start)
        | ([()'])                                         # Control Tokens
        | ("(?:[^\\"]|\\.)*")                             # String Literals
        | ([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?) # Floating Point/
                                                          #     Integer Literal
        | ([^\.\d\s;\'\"\(\)][^\s;\'\"\(\)]*)             # Symbols
        | (;.*)                                           # Comments
        | (\s)                                            # Whitespace
        | (.)                                             # Errors/Anything
                                                          #     else
        '''

    p = re.compile(patterns, re.VERBOSE)
    for pattern in p.finditer(code):
        # Ignore group 1 (shebang lines)
        # If a Control Token, find which one
        if pattern.group(2):
            if pattern[0] == '(':
                yield LParen()
            elif pattern[0] == ')':
                yield RParen()
            else:
                yield Quote()
        # If a String Literal, send to parse_strlit first
        elif pattern.group(3):
            yield parse_strlit(pattern[0])
        # If is a number
        elif pattern.group(4):
            # Error if can't interpret number
            try:
                # See if number was int, else yield float
                if isinstance(pattern[0], int):
                    yield int(pattern[0])
                else:
                    yield float(pattern[0])
            except ValueError:
                print("Value Error with value: ", pattern[0])
        # If token was a Symbol
        elif pattern.group(5):
            yield Symbol(pattern[0])
        elif pattern.group(6):
            pass
        elif pattern.group(7):
            pass
        else:
            raise SyntaxError("malformed tokens in input")
        """

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

    """Regex pattern for each of the listed possibilities in the table above"""
    result = ""         # Final String to return/convert to String type
    token = tok[1:-1]   # Inverse list for iterating, because appends reversed
    pattern = r"""
                  (\\a)                                 # ASCII Value 7
                | (\\b)                                 # ASCII Value 8
                | (\\e)                                 # ASCII Value 27
                | (\\f)                                 # ASCII Value 12
                | (\\n)                                 # ASCII Value 10
                | (\\r)                                 # ASCII Value 13
                | (\\t)                                 # ASCII Value 9
                | (\\v)                                 # ASCII Value 11
                | (\\\")                                # ASCII Value 34
                | (\\\\)                                # ASCII Value 92
                | (\\x([0-9a-fA-F]{2}))                 # Hex Value
                | (\\0([0-7]{2}))                       # Octal Value
                | (\\0)                                 # ASCII Value 0
                | (\\)                                  # Unused
                | (.)                                   # Others
                """
    p = re.compile(pattern, re.VERBOSE)

    # Apply pattern to resulting string; tokens in reverse for append
    for item in p.finditer(token):
        if item.group(1):
            result += "\x07"
        elif item.group(2):
            result += "\x08"
        elif item.group(3):
            result += "\x1b"
        elif item.group(4):
            result += "\x0c"
        elif item.group(5):
            result += "\x0a"
        elif item.group(6):
            result += "\x0d"
        elif item.group(7):
            result += "\x09"
        elif item.group(8):
            result += "\x0b"
        elif item.group(9):
            result += "\x22"
        elif item.group(10):
            result += "\x5c"
        # For next two cases, chr(int) returns unicode 16=HEX, 8=OCTAL
        elif item.group(11):
            result += chr(int(item.group(12), 16))
        elif item.group(13):
            result += chr(int(item.group(14), 8))
        elif item.group(15):
            result += "\x00"
        elif item.group(16):
            result += "\\"
        # Anything else
        elif item.group(17):
            result += item.group(17)

    # Convert to SlytherLisp String data type and return
    return String(result)


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

    result = []

    """ Length of result is used to keep track of parenthesis in both
        directions and if in between two parenthesis, process into the AST.
        If uneven parenthesis, raise error based on rule invalidated."""
    for item in tokens:
        """ If the element isn't a RParen, add it to the result.
            closed parens counted in the else"""
        if not isinstance(item, RParen):
            result.append(item)
        else:
            # Start of new list
            cdr = NIL
            # Process Contents inside Parenthesis
            while True:

                """ Save contents of last entry and process them below or
                    raise error of too many closing parens"""
                if len(result) > 0:
                    previous_token = result[-1]
                    del result[-1]
                else:
                    raise SyntaxError("too many closing parens")

                # Put contents between both parens together
                if isinstance(previous_token, LParen):
                    result.append(cdr)
                    break

                # Case to make sure that quote isn't followed by RParen
                elif isinstance(previous_token, Quote):
                    raise SyntaxError("invalid quotation")

                # If the previous token wasn't a Control Token, then create
                # an SExpression for it
                else:
                    cdr = SExpression(previous_token, cdr)

            """ If unprocessed symbols still inside parenthesis,
                process insides"""
            if len(result) > 1:
                # Get the quote in a result list
                while isinstance(result[len(result) - 2], Quote):
                    # Breaks while if less than two args left in result
                    # Should be the quotes
                    if len(result) < 2:
                        break
                    # Grab next token to append and remove from result
                    # for reassigning as Quoted
                    previous_token = result[-1]
                    del result[-1]
                    # Delete the quote from the result list
                    del result[-1]
                    result.append(Quoted(previous_token))
            # Pop extra parenthesis if there is one
            if len(result) == 1:
                yield result.pop()

        """ Case for if a Quote shows up before a paren. Should process
            quote first and then the contents in the paren"""
        if not isinstance(item, ControlToken):
            if len(result) > 1:
                # Breaks while if less than two args left in result
                # Should be the quotes
                while isinstance(result[len(result) - 2], Quote):
                    if len(result) < 2:
                        break
                    # Grab next token to append/remove & reassign as Quoted
                    previous_token = result[-1]
                    del result[-1]
                    # Delete the quote from the result list
                    del result[-1]
                    result.append(Quoted(previous_token))
            # Pop extra parenthesis if there is one
            if len(result) == 1:
                yield result.pop()
    # Input was invalid
    if len(result) != 0:
        raise SyntaxError("incomplete parse")


def lisp(code: str):
    """
    Helper function (mostly for test functions, etc.) to lex and parse
    code representing a single AST element.
    """
    return next(parse(lex(code)))
