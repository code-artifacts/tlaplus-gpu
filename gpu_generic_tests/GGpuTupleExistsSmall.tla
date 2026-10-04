---- MODULE GGpuTupleExistsSmall ----
EXTENDS Naturals

VARIABLES x, y

PairDomain == IF x = 0 THEN {<<1, 2>>, <<2, 1>>} ELSE {<<0, 0>>, <<x, y>>}

Init ==
  /\ x = 0
  /\ y = 0

Next ==
  \E <<a, b>> \in PairDomain:
    /\ x' = a
    /\ y' = b

TypeOK ==
  /\ x \in 0..2
  /\ y \in 0..2

====
