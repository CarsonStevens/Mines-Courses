#!/usr/bin/env slyther
(define (calculate op args)
  (eval (cons op args)))

(define (enter-args)
  (define prompt
    "Please enter an operand, then press enter. End with blank line.\n")
  (let ((arg (input prompt)))
    (if arg
        (cons (parse arg) (enter-args))
        NIL)))

(define (enter-op)
  (define prompt
    "Please enter the function you would like to use (e.g., +): ")
  (make-symbol (input prompt)))

(define (calculator)
  (let ((op (enter-op))
        (args (enter-args)))
    (print "The result is:" (calculate op args))
    (if (eval
          (make-symbol
            (input "Would you like to make another calculation (#t,#f)? ")))
        (calculator)
        (print "Have a nice day!"))))

(calculator)
