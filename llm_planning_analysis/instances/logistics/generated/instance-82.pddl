(define (problem LG-generalization)
(:domain logistics-strips)(:objects c2 t2 a2 l2-1 p5 l2-3 p4 l2-2 l2-4 l2-5 l2-0 c0 t0 a0 l0-1 p1 l0-3 p0 l0-2 l0-4 l0-5 l0-0 c1 t1 a1 l1-1 p3 l1-3 p2 l1-2 l1-4 l1-5 l1-0)
(:init 
(CITY c2)
(TRUCK t2)
(AIRPLANE a2)
(LOCATION l2-1)
(in-city l2-1 c2)
(OBJ p5)
(at p5 l2-1)
(at t2 l2-1)
(LOCATION l2-3)
(in-city l2-3 c2)
(OBJ p4)
(at p4 l2-3)
(LOCATION l2-2)
(in-city l2-2 c2)
(LOCATION l2-4)
(in-city l2-4 c2)
(LOCATION l2-5)
(in-city l2-5 c2)
(LOCATION l2-0)
(in-city l2-0 c2)
(CITY c0)
(TRUCK t0)
(AIRPLANE a0)
(LOCATION l0-1)
(in-city l0-1 c0)
(OBJ p1)
(at p1 l0-1)
(at t0 l0-1)
(LOCATION l0-3)
(in-city l0-3 c0)
(OBJ p0)
(at p0 l0-3)
(LOCATION l0-2)
(in-city l0-2 c0)
(LOCATION l0-4)
(in-city l0-4 c0)
(LOCATION l0-5)
(in-city l0-5 c0)
(LOCATION l0-0)
(in-city l0-0 c0)
(CITY c1)
(TRUCK t1)
(AIRPLANE a1)
(LOCATION l1-1)
(in-city l1-1 c1)
(OBJ p3)
(at p3 l1-1)
(at t1 l1-1)
(LOCATION l1-3)
(in-city l1-3 c1)
(OBJ p2)
(at p2 l1-3)
(LOCATION l1-2)
(in-city l1-2 c1)
(LOCATION l1-4)
(in-city l1-4 c1)
(LOCATION l1-5)
(in-city l1-5 c1)
(LOCATION l1-0)
(in-city l1-0 c1)
(AIRPORT l2-0)
(at a2 l2-0)
(AIRPORT l0-0)
(at a0 l0-0)
(AIRPORT l1-0)
(at a1 l1-0)
)
(:goal
(and
(at p5 l2-3)
(at p1 l0-3)
(at p3 l1-3)
(at p4 l0-0)
(at p0 l2-0)
(at p2 l2-0)
)))