(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d i c l j f h g)
(:init 
(harmony)
(planet d)
(planet i)
(planet c)
(planet l)
(planet j)
(planet f)
(planet h)
(planet g)
(province d)
(province i)
(province c)
(province l)
(province j)
(province f)
(province h)
(province g)
)
(:goal
(and
(craves d i)
(craves i c)
(craves c l)
(craves l j)
(craves j f)
(craves f h)
(craves h g)
)))