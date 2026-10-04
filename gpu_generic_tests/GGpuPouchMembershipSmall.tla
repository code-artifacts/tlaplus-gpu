---- MODULE GGpuPouchMembershipSmall ----
EXTENDS Naturals

VARIABLES pebble, pouch, seal

PebbleChoices == {1, 2, 3, 4}
PouchChoices == {{1, 3}, {2, 4}, {1, 2, 4}}
SealChoices == {FALSE, TRUE}

Init ==
  /\ pebble = 1
  /\ pouch = {1, 3}
  /\ seal = FALSE

Next ==
  /\ pouch' \in PouchChoices
  /\ pebble' \in PebbleChoices
  /\ seal' \in SealChoices
  /\ ((pebble' \in pouch') \/ seal' = TRUE)

====
