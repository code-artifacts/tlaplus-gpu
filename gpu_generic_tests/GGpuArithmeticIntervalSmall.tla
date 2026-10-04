---- MODULE GGpuArithmeticIntervalSmall ----
EXTENDS Naturals, Integers

VARIABLES n, m, total

Init ==
  /\ n = 1
  /\ m = 3
  /\ total = 2

Next ==
  /\ n' \in 1..6
  /\ m' = n' * 2 + 1
  /\ total' = m' - n'
  /\ n' < m'
  /\ total' \leq 8
  /\ m' \geq 3

====
