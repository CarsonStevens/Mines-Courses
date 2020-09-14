import hypothesis.strategies as strat
from slyther.types import Symbol as s, String as st     # noqa
from slyther.parser import ControlToken, LParen, RParen, Quote, lex

random = strat.randoms().example()

lp, rp, q = LParen(), RParen(), Quote()

token_sequences = [
    [lp, lp, lp, lp, lp, lp],
    [lp, 10, 30, 50.0, rp, s('hello-world'), q, q, rp, rp, lp],
    [rp, -10, -70.0, s('&123456789'), q, s('#!#!#!#!#!#!#!#'), s('#t'),
        s('NIL'), s('初音ミクでてむてららるちょむちょめぷ'), q, q, s('+'), q,
        s('ZZZZ'), q, s('`'), q, rp, rp, q, lp, rp, lp, q, q, rp],
    [s('OHEA'), st('OHEA'), s('OHEA'), st('OHEA'), st('OHEA'), q, q],
    [st(';;;;;;;;;;;;;;;;;;;;;;'), s('λιαoheaενορτ'), st('λιαoheaενορτ'), q,
        -40, s('λ')],
    [st('"""""  ohea  """"'), s('*****')]]


comment_s = strat.from_regex(r'\s*;[^\n]*\n\s*', fullmatch=True)
whitespace_s = strat.from_regex(r'\s+', fullmatch=True)
needed_transitions = {(s, s), (s, int), (s, float)}
dash_transitions = {(int, int), (int, float), (float, int), (float, float)}


def random_whitespace(comment=0.1, ws=0.1, one=False):
    while True:
        r = random.random()
        if r < comment:
            yield comment_s.example()
            if one:
                return
        elif r < comment + ws:
            yield whitespace_s.example()
            if one:
                return
        elif not one:
            return


def tok_repr(token):
    if isinstance(token, ControlToken):
        return {
            LParen: '(',
            RParen: ')',
            Quote: "'",
        }[type(token)]
    return repr(token)


def random_repr(tokens, minws=False):
    if not minws:
        yield from random_whitespace()
    for tok, nxt in zip(tokens, tokens[1:]):
        yield tok_repr(tok)
        tt = (type(tok), type(nxt))
        if (tt in needed_transitions
                or (tt in dash_transitions and not repr(nxt).startswith('-'))):
            yield from random_whitespace(0.5, 0.5, one=True)
        if not minws:
            yield from random_whitespace()
    yield tok_repr(nxt)
    if not minws:
        yield from random_whitespace()


def reprs(tokens, n=100):
    yield ' '.join(map(tok_repr, tokens))
    yield ''.join(random_repr(tokens, minws=True))
    for _ in range(n - 2):
        yield ''.join(random_repr(tokens))


def test_token_sequences():
    for tokens in token_sequences:
        for code in reprs(tokens):
            result = list(lex(code))
            assert tokens == result, "failed to lex: {}".format(code)
