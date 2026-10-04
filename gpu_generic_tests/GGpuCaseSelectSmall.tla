---- MODULE GGpuCaseSelectSmall ----
EXTENDS Naturals

VARIABLES key, bucket, flag

Keys == {1, 2, 3, 4}
Allowed == {11, 22, 44}

Init ==
  /\ key = 1
  /\ bucket = 11
  /\ flag = TRUE

Next ==
  /\ key' \in Keys
  /\ bucket' = CASE key' = 1 -> 11
                [] key' = 2 -> 22
                [] key' = 3 -> 33
                [] OTHER -> 44
  /\ flag' = CASE bucket' = 11 -> TRUE
              [] bucket' = 22 -> TRUE
              [] OTHER -> FALSE
  /\ CASE flag' -> bucket' \in Allowed
     [] OTHER -> bucket' /= 11

====
