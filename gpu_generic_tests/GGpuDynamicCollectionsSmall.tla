---- MODULE GGpuDynamicCollectionsSmall ----
EXTENDS Naturals

VARIABLES f, i, pairs, matrix

vars == <<f, i, pairs, matrix>>

GPU_DOMAIN_f == [0..2 -> 0..2]

Init ==
  /\ f = [x \in 0..2 |-> x]
  /\ i = 0
  /\ pairs = {}
  /\ matrix = [u \in 0..0, v \in 0..0 |-> u + v]

NestedDomains ==
  /\ i \in 0..2
  /\ \E a \in 0..i:
       \E b \in a..2:
         /\ f' = [f EXCEPT ![a] = b]
         /\ i' = b
         /\ pairs' = {<<u, v>> : u \in 0..a, v \in 0..b}
         /\ matrix' = [u \in 0..a, v \in 0..b |-> u + v]

GeneralUnchanged ==
  /\ i' \in 0..2
  /\ pairs' = UNION {{<<u, v>> : v \in u..2} : u \in 0..i'}
  /\ matrix' = [u \in 0..i', v \in 0..2 |-> u + v]
  /\ UNCHANGED f[i]

Next == NestedDomains \/ GeneralUnchanged

TypeOK ==
  /\ f \in [0..2 -> 0..2]
  /\ i \in 0..2
  /\ pairs \subseteq ((0..2) \X (0..2))
  /\ DOMAIN matrix \subseteq ((0..2) \X (0..2))

====
