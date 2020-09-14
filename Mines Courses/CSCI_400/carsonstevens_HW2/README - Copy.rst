Homework 2: Functional Programming
==================================

The purpose of this assignment is to:

1. Familiarize yourself with the Racket programming language
2. Practice and learn the fundamentals of functional programming

Getting Started
---------------

Open up ``hw-2.rkt`` in either Dr. Racket or another s-expression aware editor.
See the slides for recommendations on how to make Vim or EMACS easier to work
with Lisp dialects.

You are working with a restricted subset of Racket, as defined in
``language.rkt``.  This subset of Racket only lets you write pure-functional
code. Here are the functions, macros, and symbols you have access to::

   define
   lambda
   match
   define/match
   let
   let*
   if
   cond
   case
   list
   list*
   < > = <= >=
   even?
   odd?
   sin cos tan abs expt
   remainder modulo
   + * - /
   sub1 add1
   eqv? eq? equal?
   car cdr cadr cdar
   cons
   null?
   curry curryr

You will want to replace the parts of the code that say ``(todo)``. Note that
when ``(todo)`` was placed in a match statement, you may need to change the
cases or add more cases to the match (so change the entire match statement,
not just the ``(todo)``.

You can define any helper functions your code needs.

Running your Code
-----------------

If you are using Dr. Racket, you can run your code by clicking "Run" and
entering expressions into the REPL at the bottom.

If you are using Racket from another editor, type "racket" at your terminal to
get a REPL. You can enter into your code using the following command::

   ,en hw-2.rkt

Run any commands you need. To reload, you can type the same ``,en`` command
again.

Note: the grader will run additional test cases from the ones you were given.
Please be sure to test all of your code. Create additional test cases to the
ones you were given.

Submitting your Code
--------------------

1. Make sure all code for your submission is committed.

2. Double check that step 1 is *actually* complete.

3. Submit your ``hw-2.rkt`` file on Canvas.
