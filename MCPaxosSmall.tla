--------------------------- MODULE MCPaxosSmall ---------------------------
EXTENDS Paxos, TLC

(***************************************************************************)
(* A finite, medium-sized instance of the Paxos specification.             *)
(* The protocol actions and invariants come from Paxos.tla.                *)
(* MCMaxBallot is a model parameter so the state-space size can be tuned   *)
(* without changing the specification.                                    *)
(***************************************************************************)

CONSTANTS a1, a2, a3, a4
CONSTANTS v1, v2
CONSTANT MCMaxBallot

MCAcceptor == {a1, a2, a3, a4}
MCValue == {v1, v2}

(* Every quorum has three of the four acceptors, so quorums intersect. *)
MCQuorum == {
    {a1, a2, a3},
    {a1, a2, a4},
    {a1, a3, a4},
    {a2, a3, a4}
}

MCBallot == 0..MCMaxBallot
MCSymmetry == Permutations(MCAcceptor) \cup Permutations(MCValue)

(* Optional checks and refinement expressions retained for compatibility. *)
VotingSpecBar == V!Spec

MCLSpec == /\ Spec
           /\ WF_vars(Phase1a(MCMaxBallot))
           /\ \A v \in Value : WF_vars(Phase2a(MCMaxBallot, v))
           /\ \A a \in {a1, a2} : WF_vars(Phase1b(a) \/ Phase2b(a))

MCLiveness == <> (V!chosen # {})

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
