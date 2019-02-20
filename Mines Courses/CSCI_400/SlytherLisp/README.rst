SlytherLisp
===========

.. This README is in reStructuredText format. If you have Docutils installed,
   you can validate the format and make an HTML copy by typing:
     $ rst2html README.rst >README.html

.. Replace with your own names and Mines email addresses.

:Implemented By:
   * Carson Stevens <carsonstevens@Mines.EDU>


.. Replace with the current deliverable you are working on. For example, if you
   are submitting the first deliverable, this should be 1 (as shown).

:Deliverable: 1

SlytherLisp is a lexically-scoped Scheme-like Lisp dialect for CSCI-400. Lisp
dialects use s-expressions as their primary syntax structure, just like we have
learned in class. For example, adding 3 and 4 is written as::

    (+ 3 4)

The ``+`` is the function we are calling, and ``3`` and ``4`` are the
arguments. This language only uses prefix notation, no infix or postfix
notation is used.

In SlytherLisp, there are two kinds of things we can call by placing in the
parentheses:

:Functions: Whose arguments are evaluated, and the function returns a value to
            substitute where it was written.
:Macros:    Whose arguments are not evaluated, and the macro returns an
            expression to fill in its place.

For example, the ``define`` macro defines functions::

   (define (f x y)
     (* (+ x y) y))

In this case, we defined a function named ``f`` which takes arguments ``x`` and
``y`` and returns ``(x + y) * y``. It is necessary for ``define`` to be a macro
since it needs the arguments unevaluated to create the definition.

Further examples of SlytherLisp code can be found in the ``examples``
directory. Also, as a beginner reference to Scheme and Lisp-like languages, the
`Structure and Interpretation of Computer Programs`__ (available *free* online)
book is recommended, even though SlytherLisp is not exactly the same as Scheme.

__ https://mitpress.mit.edu/sicp/full-text/book/book.html

Getting Started with the Starter Code
-------------------------------------

1. Open this ``README.rst`` file in a text editor and make changes to the name
   and email up top. Save the file before proceeding to step 2.

2. Install an editable copy of the application::

      $ python3.6 -m pip install -e . --user

   or::

      $ python3.7 -m pip install -e . --user

   - ``-e`` says to install an editable copy. If you omit this, you'll need to
     re-install every time you make changes.

   - ``--user`` says to install for the local user only. This way you do not
     need to use ``sudo``. The binaries should end up in something like
     ``$HOME/.local/bin``: make sure this is in your ``PATH``.

3. After installing the application, you should have access to the ``slyther``
   program anywhere on your system. Confirm you can run the ``slyther``
   command (even though it may result in a ``NotImplementedError``). Check
   ``slyther --help`` for potential command line arguments.

   - If you are still getting ``slyther: command not found``, please ask on
     Piazza or come to office hours for help.

4. Open up the files in the ``slyther`` directory and familiarize yourself with
   their structure.

   - The ``raise NotImplementedError(...)`` lines are for you to replace with
     working code. Typically, they will state which deliverable you need to
     complete them for.

   - The functions typically have a description of how they should work at the
     top in a docstring. This docstring usually has doctests in it too: that
     is, those lines that start with ``>>>`` in the docstring are actually unit
     tests as well! You are free to change these docstrings or remove as you
     please.

   - You are free to implement helper functions, etc., as you need: in fact,
     this project would be very hard to do without doing so.

   - If you wish to change how a particular interface works in the application,
     leave the original interface in, and have that call the modified
     interface. This is how (you, and I) can test your code even if you change
     the application structure.

5. Start coding!

Running Tests
-------------

To run the tests, type ``pytest`` from the base directory (where ``README.rst``
is located). You will need to specify at least one of four flags: ``--d1``,
``--d2``, ``--d3``, or ``--d4``, which runs the tests for Deliverable 1, 2, 3,
or 4. For example,

Running D1 tests::

   $ pytest --d1

Running all tests::

   $ pytest --d1 --d2 --d3 --d4

.. warning::

   Your grade is not solely based on unit tests. The instructor or grader
   *will* perform additional testing on your code.

Pretty Formatting
~~~~~~~~~~~~~~~~~

If you wish to show unit test results in your browser, a plugin for ``pytest``
is available.

https://pypi.org/project/pytest-html/

I highly recommend this plugin to help make looking at the test results easier.

Style Checking
--------------

The file ``.flake8`` defines a set of style checks (PEP 8, plus a few others).
To run the style checks, type ``flake8`` from the base directory. This is the
same way that the code style will be checked when graded.

With ``flake8``, no news is good news as well. If you want to make sure it's
working, add some bad style to your code for a second and see if it errors at
you.

Submitting your Deliverables
----------------------------

1. Make sure all code for your submission is completed.

2. Run the ``make_submission.sh`` script.

3. Upload the resulting ``submission.tar.bz2`` to Canvas.
