--------------- MODULE GGpuMultiInitialInvariantViolationSmall ---------------
EXTENDS Naturals

VARIABLE x

Init == x \in 0..1

Next == x' = x

OnlyFirstInitialIsValid == x = 0

=============================================================================
