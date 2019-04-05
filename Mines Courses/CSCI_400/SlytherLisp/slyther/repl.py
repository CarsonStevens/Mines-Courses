import threading
import atexit

def repl(interpreter, debug=False):
    """
    Take an interpreter object (see ``slyther/interpreter.py``) and give a REPL
    on it. Should not return anything: just a user interface at the terminal.
    For example::

        $ slyther
        > (print "Hello, World!")
        Hello, World!
        NIL
        > (+ 10 10 10)
        30

    When the user presses ^D at an empty prompt, the REPL should exit.

    When the user presses ^C at any prompt (whether there is text or
    not), the input should be cancelled, and the user prompted again::

        $ slyther
        > (blah bla^C
        >                   <-- ^C resulted in new prompt line

    Should be pretty easy. No unit tests for this function, but I will
    test the interface works when I grade it.

    Optionally, you may want to prevent the REPL from raising an exception
    when an exception in user code occurs, and allow the user to keep
    typing further expressions. This is not required, just a suggestion.
    If you do this, you should probably disable this behavior when ``debug``
    is set to ``True``, as it allows for easy post-mortem debugging with pdb
    or pudb.
    """

    # ^C exits, so calls atexit which calls the interpreter again
    atexit.register(repl(interpreter))
    t = WorkerThread()
    t.daemon = True
    t.start()

    # User input and send to interpreter
    expr = input("> ")
    print(interpreter.exec(expr))
    repl(interpreter)

class WorkerThread(threading.Thread):
    def __init__(self):
        super(WorkerThread, self).__init__()
        self.quit = False

    def run(self):
        while not self.quit:
            pass

    def stop(self):
        self.quit = True




        # User hit control C, start new command line
        # except KeyboardInterrupt:
        #     print()
        #     continue
        # User hit control D, exit program
        # except EOFError:
        #     print(">>>\tEXITING\t<<<")
        #     exit(0)
