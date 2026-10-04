---- MODULE GGpuActionIfSmall ----
EXTENDS Naturals

VARIABLES x, y

vars == <<x, y>>

Init ==
  /\ x = 0
  /\ y = 0

Next ==
  /\ IF x = 0
        THEN /\ x' = 1
             /\ IF y = 0 THEN y' = 1 ELSE y' = 0
        ELSE /\ x' = 0
             /\ y' = 0

TypeOK == x \in 0..1 /\ y \in 0..1

====
