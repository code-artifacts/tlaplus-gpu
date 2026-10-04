---- MODULE GGpuActionWrappersSmall ----
EXTENDS Naturals

VARIABLES x, y, phase

vars == <<x, y, phase>>

Init ==
  /\ x = 0
  /\ y = 10
  /\ phase = "ready"

Inc ==
  /\ x < 2
  /\ x' = x + 1
  /\ UNCHANGED <<y, phase>>

Reset ==
  /\ x' = 0
  /\ UNCHANGED <<y, phase>>

Next ==
  \/ [Inc]_vars
  \/ <<Reset>>_vars

TypeOK ==
  /\ x \in 0..2
  /\ y = 10
  /\ phase = "ready"

====
