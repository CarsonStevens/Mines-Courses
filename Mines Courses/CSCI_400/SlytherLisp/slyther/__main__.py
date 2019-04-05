import os
import sys
import importlib
import argparse
import traceback
from slyther.interpreter import Interpreter


def main():
    """
    The entry point for the ``slyther`` command.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pypy',
        action='store_true',
        help='Run using PyPy (experimental and not required to work)')
    parser.add_argument(
        '--load',
        action='append',
        default=[],
        type=argparse.FileType('r'),
        help='Source code to evaluate before dropping to a REPL')
    parser.add_argument(
        'source',
        type=argparse.FileType('r'),
        nargs='?',
        help='Source code to run')
    args = parser.parse_args()

    if args.pypy and sys.implementation.name != 'pypy':
        env = dict(os.environ)
        env['PYTHONPATH'] = os.path.dirname(os.path.dirname(
            importlib.util.find_spec('slyther').origin))
        try:
            os.execvpe('pypy3', ['pypy3', '-m', 'slyther'] + sys.argv[1:], env)
        except FileNotFoundError:
            print("The pypy3 command must be available on your system "
                  "for this feature to work.", file=sys.stderr)
            sys.exit(1)

    # This is just an easy way to allow no exception catching when pdb
    # is loaded. This allows the implementer to use python -m pdb and
    # do easy post-mortem debugging.
    interp = Interpreter()

    def run(debug=False):
        for f in args.load:
            interp.exec(f.read())
        if args.source:
            interp.exec(args.source.read())
        else:
            from slyther.repl import repl
            repl(interp, debug=debug)

    if any(m in sys.modules.keys() for m in ('pdb', 'pudb')):
        run(debug=True)
    else:
        try:
            run()
        except KeyboardInterrupt:
            sys.exit(1)
        except Exception:
            traceback.print_exc(limit=10, chain=False)


if __name__ == '__main__':
    main()
