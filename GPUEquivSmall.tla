--------------------------- MODULE GPUEquivSmall ---------------------------
EXTENDS Naturals, TLC

CONSTANT N
VARIABLES x, y

Init ==
    /\ x = 0
    /\ y = 0

Next ==
    \/ /\ x < N
       /\ x' = x + 1
       /\ y' = y
    \/ /\ y < N
       /\ x' = x
       /\ y' = y + 1
    \/ /\ x > 0
       /\ x' = x - 1
       /\ y' = y
    \/ /\ y > 0
       /\ x' = x
       /\ y' = y - 1

TypeOK ==
    /\ x \in 0..N
    /\ y \in 0..N

Spec == Init /\ [][Next]_<<x, y>>
=============================================================================
