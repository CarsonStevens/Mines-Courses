#lang racket/base
(require racket/match)
(require racket/function)
(provide
  todo
  define
  lambda
  match
  define/match
  let let*
  if
  cond
  case
  else
  list list*
  < > = <= >=
  even? odd?
  sin cos tan abs expt
  remainder modulo
  + * - /
  sub1 add1
  eqv? eq? equal?
  car cdr cadr cdar
  cons
  null?
  curry curryr
  quote
  quasiquote
  unquote
  unquote-splicing
  #%app
  #%module-begin
  #%top-interaction
  #%datum)

(define-syntax-rule (todo)
  (error
    (caar (continuation-mark-set->context
            (current-continuation-marks)))
    "not implemented"))
