---- MODULE GGpuSetFunctionOpsSmall ----
EXTENDS Naturals, Integers

VARIABLES x, y, z, d, v, picked

A == 1..3
B == {2, 3, 4}
F == [i \in 1..3 |-> i * 10]

Init ==
  /\ x = 1
  /\ y = 2
  /\ z = 4
  /\ d = 1
  /\ v = 10
  /\ picked = {}

Next ==
  /\ x' \in A \union B
  /\ y' \in A \intersect B
  /\ z' \in B \ A
  /\ d' \in DOMAIN F
  /\ v' = F[d']
  /\ picked' \in SUBSET (A \intersect B)
  /\ x' >= y'
  /\ picked' \subseteq A

====
