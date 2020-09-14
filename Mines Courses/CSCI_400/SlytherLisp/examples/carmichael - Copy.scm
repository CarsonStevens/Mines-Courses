#!/usr/bin/env slyther
; Prints all Carmichael numbers: 561, 1105, 1729, 2465, ...
; See https://en.wikipedia.org/wiki/Carmichael_number

; This program almost certainly requires tail call optimization to work!
; Great way to stress-test your TCO. I found bugs in mine using this!

(define (divides? a b)
  ; return #t if a divides b, #f otherwise
  (= (remainder b a) 0))

(define (isqrt n)
  ; compute (floor (sqrt n)), but "efficently"
  (define (isqrt-iter guess)
    (let ((next (/ (+ guess (/ n guess)) 2)))
      (if (< (abs (- next guess)) 1)
          (floor next)
          (isqrt-iter next))))
  (isqrt-iter (/ n 2)))

(define (prime? n)
  (define stop (isqrt n))
  (define (prime-iter x)
    (and (not (divides? x n))
         (if (<= x stop)
             (prime-iter (+ 2 x))
             #t)))
  (cond
    ((> n 3) (and
               (not (divides? 2 n))
               (prime-iter 3)))
    ((>= n 2) #t)
    (#t #f)))

(define (congruent a b m)
  (= (remainder a m) (remainder b m)))

(define (fermat-prime? n)
  (define (prime-iter b)
    (and (congruent (expt b n) b n)
         (if (< b n)
             (prime-iter (+ 1 b))
             #t)))
  (prime-iter 2))

(define (print-carmichaels x)
  (if (and (fermat-prime? x) (not (prime? x)))
      (print x)
      NIL)
  (print-carmichaels (+ 2 x)))

(print-carmichaels 5)
