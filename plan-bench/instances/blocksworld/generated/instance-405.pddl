(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c l i a j h f k b d)
(:init 
(handempty)
(ontable c)
(ontable l)
(ontable i)
(ontable a)
(ontable j)
(ontable h)
(ontable f)
(ontable k)
(ontable b)
(ontable d)
(clear c)
(clear l)
(clear i)
(clear a)
(clear j)
(clear h)
(clear f)
(clear k)
(clear b)
(clear d)
)
(:goal
(and
(on c l)
(on l i)
(on i a)
(on a j)
(on j h)
(on h f)
(on f k)
(on k b)
(on b d)
)))