---- MODULE GGpuSwitchLaneSmall ----
EXTENDS Naturals

VARIABLES lane, lamp, anchor

LaneChoices == {0, 1, 2, 3}
LampChoices == {TRUE, FALSE}

Init ==
  /\ lane = 0
  /\ lamp = FALSE
  /\ anchor = 42

Next ==
  /\ lane' \in LaneChoices
  /\ lamp' \in LampChoices
  /\ (lane' = 0 \/ lane' = 2 \/ lamp' = TRUE)
  /\ UNCHANGED anchor

====
