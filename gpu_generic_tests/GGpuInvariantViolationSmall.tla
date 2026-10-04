---------------------- MODULE GGpuInvariantViolationSmall ---------------------
EXTENDS Naturals

VARIABLE x

Init == x = 0

Next == /\ x < 2
        /\ x' = x + 1

TypeOK == x \in 0..1

=============================================================================
