#!/usr/bin/env slyther
(define base (make-float
               (input "What is the length of the base of your triangle? ")))

(define height (make-float
                 (input "What is the height of your triangle? ")))

(print "The area is" (/ (* base height) 2))
(print "Have a great day!")
