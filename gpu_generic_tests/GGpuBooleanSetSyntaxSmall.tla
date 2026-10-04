---- MODULE GGpuBooleanSetSyntaxSmall ----
EXTENDS Naturals

VARIABLES token, bag, gate, mirror

TokenChoices == {1, 2, 3}
BagChoices == {{1, 2}, {2, 3}, {1, 2, 3}}
GateChoices == {TRUE, FALSE}
BlockedTokens == {3}
RequiredBag == {1, 2}

Init ==
  /\ token = 1
  /\ bag = {1, 2}
  /\ gate = FALSE
  /\ mirror = TRUE

Next ==
  /\ token' \in TokenChoices
  /\ bag' \in BagChoices
  /\ gate' \in GateChoices
  /\ mirror' \in GateChoices
  /\ token' /= 3
  /\ token' \notin BlockedTokens
  /\ RequiredBag \subseteq bag'
  /\ (gate' => token' \in bag')
  /\ (mirror' \equiv ~(token' \notin bag'))

====
