---- MODULE GGpuIfThenElseSmall ----
EXTENDS Naturals

VARIABLES choice, out, pass

ChoiceSet == {1, 2, 3}
BoolSet == {TRUE, FALSE}
Good == {10, 20}

Init ==
  /\ choice = 1
  /\ out = 10
  /\ pass = TRUE

Next ==
  /\ choice' \in ChoiceSet
  /\ pass' \in BoolSet
  /\ out' = IF choice' = 1 THEN 10 ELSE IF choice' = 2 THEN 20 ELSE 30
  /\ IF pass' THEN out' \in Good ELSE out' /= 10

====
