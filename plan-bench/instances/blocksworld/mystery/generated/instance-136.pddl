(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d g i j l e b f)
(:init 
(harmony)
(planet d)
(planet g)
(planet i)
(planet j)
(planet l)
(planet e)
(planet b)
(planet f)
(province d)
(province g)
(province i)
(province j)
(province l)
(province e)
(province b)
(province f)
)
(:goal
(and
(craves d g)
(craves g i)
(craves i j)
(craves j l)
(craves l e)
(craves e b)
(craves b f)
)))