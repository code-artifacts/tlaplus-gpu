---- MODULE GGpuRecordFunctionSetSmall ----
EXTENDS Naturals, Integers

VARIABLES rec, tally

Ids == {"x", "y"}

Init ==
  /\ rec = [phase |-> "open"]
  /\ tally = [id \in Ids |-> 0]

OpenToMid ==
  /\ rec \in [phase : {"open"}]
  /\ rec' = [phase |-> "mid", id |-> "x"]
  /\ UNCHANGED tally

MidToClosed ==
  /\ rec \in [phase : {"mid"}, id : Ids]
  /\ tally \in [Ids -> 0..1]
  /\ rec' = [phase |-> "closed"]
  /\ UNCHANGED tally

Next == OpenToMid \/ MidToClosed

====
