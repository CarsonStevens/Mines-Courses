#!/usr/bin/env slyther

;===========================================================
; The following should be the output of this program:
;===========================================================
; Testing rng1...
; 16807
; 282475249
; 1622650073
; 984943658
; 1144108930
; 470211272
; 101027544
; 1457850878
;
; Testing rng2...
; 16807
; 282475249
; 1622650073
; 984943658
; 1144108930
; 470211272
; 101027544
; 1457850878
;
; The original pRNG should be unharmed!
; 1458777923
; 2007237709
; 823564440
; 1115438165
; 1784484492
; 74243042
; 114807987
; 1137522503
;
; Should be the same as:
; 1458777923
; 2007237709
; 823564440
; 1115438165
; 1784484492
; 74243042
; 114807987
; 1137522503
;===========================================================
; If what you see above is not the output, your SlytherLisp
; impementation is incorrect.
;===========================================================

(define (prng seed)
  ; Lehmer random number generator
  (lambda ()
    (set! seed (remainder (* 16807 seed) 2147483647))
    seed))

(define (n-calls f args n)
  (if (< n 1)
      NIL
      ((lambda ()
         (print (eval (cons f args)))
         (n-calls f args (- n 1))))))

(define rng1 (prng 1))
(print "Testing rng1...")
(n-calls rng1 () 8)

(define rng2 (prng 1))
(print "\nTesting rng2...")
(n-calls rng2 () 8)

(print "\nThe original pRNG should be unharmed!")
(n-calls rng1 () 8)

(print "\nShould be the same as:")
(n-calls rng2 () 8)
