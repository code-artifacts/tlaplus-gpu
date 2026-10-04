---- MODULE GGpuRelationsSmall ----
EXTENDS Naturals

VARIABLES x, y

vars == <<x, y>>

\* Explicit finite universes used only when a relation does not provide a
\* direct next-state generator for the corresponding variable.
\* Deliberately reverse the variable-index dependency: x is declared before y,
\* but its finite universe depends on y'.  The compiler must emit y first.
GPU_DOMAIN_x == y'..3
GPU_DOMAIN_y == 0..3

Init ==
  /\ x = 0
  /\ y = 0

IncX ==
  /\ x < 2
  /\ x' = x + 1
  /\ UNCHANGED y

CopyIntermediateX ==
  /\ y' = x
  /\ UNCHANGED x

Composed == IncX \cdot CopyIntermediateX

\* Neither next-state variable is introduced by x'=e or x'\in S.  The GPU
\* relation compiler enumerates the declared finite universes and applies the
\* formula as a relation filter.
GeneralRelation ==
  /\ y' < x'
  /\ x' + y' = 3

EnabledStutter ==
  /\ ENABLED Composed
  /\ UNCHANGED vars

Next == Composed \/ GeneralRelation \/ EnabledStutter

TypeOK == x \in 0..3 /\ y \in 0..3

====
