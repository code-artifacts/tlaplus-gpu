---- MODULE GGpuCardinalityConcatSmall ----
EXTENDS Integers, FiniteSets, Sequences

VARIABLE seq

Init == seq = <<>>

Next ==
    /\ seq' = IF seq = <<>>
                 THEN seq \o <<Cardinality({1, 2})>>
                 ELSE seq
    /\ Len(seq \o <<Cardinality({1, 2})>>) = Len(seq) + 1

Spec == Init /\ [][Next]_<<seq>>
====
