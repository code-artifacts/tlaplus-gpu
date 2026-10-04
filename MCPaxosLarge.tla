-------------------------------- MODULE MCPaxosLarge -----------------------------
EXTENDS Paxos, TLC
-----------------------------------------------------------------------------
CONSTANTS a1, a2, a3, a4  \* acceptors
CONSTANTS v1, v2, v3       \* values

MCAcceptor == {a1, a2, a3, a4}
MCValue    == {v1, v2, v3}

\* All 3-of-4 quorums.  Every pair of quorums intersects.
MCQuorum == {
  {a1, a2, a3},
  {a1, a2, a4},
  {a1, a3, a4},
  {a2, a3, a4}
}

MCMaxBallot == 4
MCBallot == 0..MCMaxBallot
MCSymmetry == Permutations(MCAcceptor) \cup Permutations(MCValue)

VotingSpecBar == V!Spec
-----------------------------------------------------------------------------
(***************************************************************************)
(* For checking liveness.                                                  *)
(***************************************************************************)
MCLSpec == /\ Spec
           /\ WF_vars(Phase1a(MCMaxBallot))
           /\ \A v \in Value : WF_vars(Phase2a(MCMaxBallot, v))
           /\ \A a \in {a1, a2} : WF_vars(Phase1b(a) \/ Phase2b(a))
MCLiveness == <>(V!chosen # {})
-----------------------------------------------------------------------------
(***************************************************************************)
(* For checking the inductive invariant.                                   *)
(***************************************************************************)

ITypeOK == /\ maxBal \in [Acceptor -> Ballot \cup {-1}]
           /\ maxVBal \in [Acceptor -> Ballot \cup {-1}]
           /\ maxVal \in [Acceptor -> Value \cup {None}]
           /\ msgs \in SUBSET Message

IInv == /\ ITypeOK
        /\ Inv!2
        /\ Inv!3
        /\ Inv!4

MCISpec == IInv /\ [][Next]_vars

Inv1 == Inv!1
Inv2 == Inv!2
Inv3 == Inv!3
Inv4 == Inv!4

MCIProp == [][V!Next]_<<votes, maxBal>>
=============================================================================
