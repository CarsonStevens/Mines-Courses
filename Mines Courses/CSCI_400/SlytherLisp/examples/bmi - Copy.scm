#!/usr/bin/env slyther
(define (read-float prompt)
  (make-float (input prompt)))

(define weight (read-float "What is your weight in lbs? "))
(define height (read-float "What is your height in inches? "))

(define bmi
  (let ((weight (* weight 0.454))
        (height (* height 0.0254))
        (square (lambda (n) (expt n 2))))
        (/ weight (square height))))

(print (* 50 '*))
(print (format
         "Body Profile: {} lbs, {} inches tall"
         weight height))

(print "Your BMI is:" bmi)
(print
  "Your rating is:"
  (cond
    ((<= bmi 18.5) 'UNDERWEIGHT)
    ((<= bmi 25) 'NORMAL)
    ((<= bmi 30) 'OVERWEIGHT)
    (#t 'OBESE)))
(print (* 50 '*))
