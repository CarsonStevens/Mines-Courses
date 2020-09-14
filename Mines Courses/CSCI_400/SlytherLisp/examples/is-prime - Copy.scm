#!/usr/bin/env slyther
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

(define (print-primes x)
  (if (prime? x)
      (print x)
      NIL)
  (print-primes (+ 2 x)))

(print 2)
(print-primes 3)
