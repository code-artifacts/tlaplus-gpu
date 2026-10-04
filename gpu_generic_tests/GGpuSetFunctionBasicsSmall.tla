---- MODULE GGpuSetFunctionBasicsSmall ----
EXTENDS Naturals, Integers

VARIABLES k, val, merged, picked

Vec == <<3, 5, 7>>
Blocks == {{1, 2}, {3}, 4..5}
Base == {1, 2}

Init ==
  /\ k = 1
  /\ val = 3
  /\ merged = 1
  /\ picked = {}

Next ==
  /\ k' \in DOMAIN Vec
  /\ val' = Vec[k']
  /\ merged' \in UNION Blocks
  /\ picked' \in SUBSET Base
  /\ k' \leq 3
  /\ merged' \in 1..5
  /\ picked' \subseteq Base

====
