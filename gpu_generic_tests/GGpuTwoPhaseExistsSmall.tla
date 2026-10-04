---- MODULE GGpuTwoPhaseExistsSmall ----
EXTENDS Naturals, Integers

VARIABLES rmState, msgs

RM == {"rm1", "rm2"}

Init ==
  /\ rmState = [rm \in RM |-> "working"]
  /\ msgs = {}

Prepare(rm) ==
  /\ rmState[rm] = "working"
  /\ rmState' = [rmState EXCEPT ![rm] = "prepared"]
  /\ msgs' = msgs \cup {[type |-> "Prepared", rm |-> rm]}

Next ==
  \E rm \in RM : Prepare(rm)

====
