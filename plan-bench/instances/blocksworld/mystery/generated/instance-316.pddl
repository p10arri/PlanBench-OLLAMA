(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects k c h d g b e j f)
(:init 
(harmony)
(planet k)
(planet c)
(planet h)
(planet d)
(planet g)
(planet b)
(planet e)
(planet j)
(planet f)
(province k)
(province c)
(province h)
(province d)
(province g)
(province b)
(province e)
(province j)
(province f)
)
(:goal
(and
(craves k c)
(craves c h)
(craves h d)
(craves d g)
(craves g b)
(craves b e)
(craves e j)
(craves j f)
)))