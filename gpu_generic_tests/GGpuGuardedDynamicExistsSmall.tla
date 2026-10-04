---- MODULE GGpuGuardedDynamicExistsSmall ----
EXTENDS Naturals

VARIABLES pc, domain, x

vars == <<pc, domain, x>>

Init ==
  /\ pc = "off"
  /\ domain = 0
  /\ x = 0

Start ==
  /\ pc = "off"
  /\ pc' = "on"
  /\ domain' = {1, 2}
  /\ x' = 0

Pick ==
  /\ pc = "on"
  /\ \E value \in domain:
       /\ x' = value
       /\ UNCHANGED <<pc, domain>>

Next == Start \/ Pick

TypeOK ==
  \/ /\ pc = "off"
     /\ domain = 0
     /\ x = 0
  \/ /\ pc = "on"
     /\ domain = {1, 2}
     /\ x \in 0..2

====
