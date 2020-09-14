#!/usr/bin/env slyther
(define (gcd a b)
  (if (= b 0)
      a
      (gcd b (remainder a b))))

(define (print-gcds a b)
  (print
    (format "The gcd of {} and {} is" a b)
    (gcd a b))
  (if (< a b)
      (print-gcds (+ 1 a) b)
      (print-gcds 1 (+ 1 b))))

(print-gcds 1 1)
