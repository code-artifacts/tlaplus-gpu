---- MODULE GGpuTwoPhaseStructuresSmall ----
EXTENDS Naturals, Integers

VARIABLES rmState, msgs, seenType

RM == {"rm1", "rm2"}

Init ==
  /\ rmState = [rm \in RM |-> "working"]
  /\ msgs = {}
  /\ seenType = "none"

Next ==
  /\ rmState["rm1"] = "working"
     \/ ([type |-> "Prepared", rm |-> "rm1"] \in msgs /\ rmState["rm1"] = "prepared")
     \/ ([type |-> "Commit"] \in msgs /\ rmState["rm2"] = "working")
  /\ rmState' = IF rmState["rm1"] = "working" THEN [rmState EXCEPT !["rm1"] = "prepared"]
                ELSE IF [type |-> "Prepared", rm |-> "rm1"] \in msgs THEN [rmState EXCEPT !["rm1"] = "committed"]
                ELSE [rmState EXCEPT !["rm2"] = "committed"]
  /\ msgs' = IF rmState["rm1"] = "working" THEN msgs \cup {[type |-> "Prepared", rm |-> "rm1"]}
             ELSE IF [type |-> "Prepared", rm |-> "rm1"] \in msgs THEN msgs \cup {[type |-> "Commit"]}
             ELSE msgs
  /\ seenType' = IF rmState["rm1"] = "working" THEN [type |-> "Prepared", rm |-> "rm1"].type
                 ELSE IF [type |-> "Prepared", rm |-> "rm1"] \in msgs THEN [type |-> "Commit"].type
                 ELSE seenType

====
