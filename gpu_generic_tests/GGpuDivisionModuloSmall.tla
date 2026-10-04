---- MODULE GGpuDivisionModuloSmall ----
EXTENDS Integers

VARIABLES x, quotient, remainder

Init ==
  /\ x = -7
  /\ quotient = (-7) \div 3
  /\ remainder = (-7) % 3

Next ==
  /\ x' \in -2..2
  /\ quotient' = x' \div 2
  /\ remainder' = x' % 2

TypeOK ==
  /\ x \in {-7} \union (-2..2)
  /\ quotient \in -3..1
  /\ remainder \in 0..2

====
