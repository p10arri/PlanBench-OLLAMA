(define (problem BW-generalization-4)
(:domain blocksworld-4ops)(:objects c l d a g i b f h j e)
(:init 
(handempty)
(ontable c)
(ontable l)
(ontable d)
(ontable a)
(ontable g)
(ontable i)
(ontable b)
(ontable f)
(ontable h)
(ontable j)
(ontable e)
(clear c)
(clear l)
(clear d)
(clear a)
(clear g)
(clear i)
(clear b)
(clear f)
(clear h)
(clear j)
(clear e)
)
(:goal
(and
(on c l)
(on l d)
(on d a)
(on a g)
(on g i)
(on i b)
(on b f)
(on f h)
(on h j)
(on j e)
)))