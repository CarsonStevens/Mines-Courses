#!/usr/bin/env slyther
; If your interpreter features tail-call optimization (for extra
; credit) you should not get a RecursionError with this example.

(define (print-fibs a b)
  (print a)
  (print-fibs b (+ a b)))

(print-fibs 0 1)
