---------------------------- MODULE PaxosCommit ----------------------------
EXTENDS Integers

Maximum(S) ==
    IF S = {} 
    THEN -1
    ELSE CHOOSE n \in S : \A m \in S : n>=m
    
CONSTANT RM, Acceptor, Majority, Ballot

ASSUME 
    /\Ballot \subseteq Nat
    /\0 \in Ballot
    /\Majority \subseteq SUBSET Acceptor
    /\ \A MS1, MS2 \in Majority : MS1 \cap MS2 /={}
   
    
Message == 
    [type |-> "phase1a", ins |-> RM, bal |-> Ballot \{0}]
    \/
    [type |-> "phase1b", ins |-> RM, mbal |-> Ballot, bal |->Ballot \/ {-1}, val|->{"prepared", "aborted", "none"}, acc |-> Acceptor]
    \/
    [type |-> "phase2a", ins |-> RM, bal |-> Ballot, val |-> {"prepared", "aborted"}]
    \/
    [type |-> "phase2b", acc |-> Acceptor, ins |-> RM, bal |-> Ballot, val |-> {"prepared", "aborted"}]
    \/
    [type |-> {"Commit", "Abort"}]
    
VARIABLES rmState, aState, msgs

PCTypeOK ==
    /\rmState \in [RM |-> {"working", "prepared", "committed", "aborted"}]
    /\aState \in [RM |-> [Acceptor |->[mbal : Ballot, bal: Ballot \cap {-1}, val : {"prepared", "aborted", "none"}]]]
    /\msgs \subseteq SUBSET Message
    
PCInit == 
    /\rmState = [rm \in RM |->"working"]
    /\aState = [ins \in RM |-> [ac \in Acceptor |-> [mbal |-> 0, bal |-> -1, val |-> "none"]]]
    /\msgs = {}
    
Send(m) == 
    /\msgs' = msgs \cup {m}

RMPrepared(rm) == 
    /\rmState[rm] = "working"
    /\rmState' = [rmState EXCEPT ![rm] = "prepared"]
    /\Send([type |-> "phase2a", ins |->rm, bal |->0, val |->"prepared"])
    /\UNCHANGED <<aState>>

RMRcvCommitMsg(rm) == 
    /\ [type |-> "Commit"] \in msgs
    /\ rmState' = [rmState EXCEPT ![rm] = "committed"]
    /\ UNCHANGED <<aState, msgs>>
    
RMChooseToAbort(rm) ==
    /\rmState[rm] = "working"
    /\rmState' = [rmState EXCEPT ![rm] = "aborted"]
    /\Send([type |-> "phase2a", ins |->rm, bal |->0, val |->"aborted"])
    /\ UNCHANGED <<aState>>
    
RMRcvAbortMsg(rm) ==
    /\ [type |-> "Abort"] \in msgs
    /\ rmState' = [rmState EXCEPT ![rm] = "aborted"]
    /\ UNCHANGED <<aState, msgs>>
    
Phase1a(bal, rm) == 
    /\Send([type |-> "phase1a", ins |-> rm, bal |-> bal])
    /\UNCHANGED <<rmState, aState>>
    
Phase2a(bal, rm) ==
    /\ ~\E m \in msgs :
        /\ m.type = "phase2a"
        /\ m.bal = bal
        /\ m.ins = rm
    /\ \E MS \in Majority : 
        LET mset == {m \in msgs : 
                        /\m.type = "phase1b"
                        /\m.ins = rm
                        /\m.mbal = bal
                        /\m.acc \in MS}
            mu == Maximum({m.bal : m \in mset})
            v == IF mu = -1
                 THEN "aborted"
                 ELSE (CHOOSE m \in mset : m.bal = mu).val
        IN   
            /\ \A ac \in MS : \E m \in mset : m.acc = ac
            /\ Send([type |-> "phase2a", ins |-> rm, bal |-> bal, val |-> v])
            /\ UNCHANGED <<rmState, aState>>

Decide == 
    /\LET Decided(rm , v) == 
            \E b \in Ballot, MS \in Majority :
                \A ac \in MS : [type |-> "phase2b",ins |-> rm, bal |-> b, val |-> v, acc |-> ac] \in msgs
                
      IN /\ ( \A rm \in RM : Decided(rm, "prepared")
            /\ Send([type |-> "Commit"]) )
         \/ ( \E rm \in RM : Decided(rm, "aborted")
            /\ Send([type |-> "Abort"]) )
    /\ UNCHANGED <<rmState, aState>>
    
Phase1b(acc) ==
    \E m \in msgs :
        /\ m.type = "phase1a"
        /\ aState[m.ins][acc].mbal < m.bal
        /\ aState' =
            [aState EXCEPT ![m.ins][acc].mbal = m.bal]
        /\ msgs' = msgs \cup {
            [ type |-> "phase1b",
              ins  |-> m.ins,
              mbal |-> m.bal,
              bal  |-> aState[m.ins][acc].bal,
              val  |-> aState[m.ins][acc].val,
              acc  |-> acc ]
          }
        /\ UNCHANGED <<rmState>>


Phase2b(acc) ==
    \E m \in msgs :
        /\ m.type = "phase2a"
        /\ aState[m.ins][acc].mbal <= m.bal
        /\ aState' =
            [ aState EXCEPT
                ![m.ins][acc].mbal = m.bal,
                ![m.ins][acc].bal  = m.bal,
                ![m.ins][acc].val  = m.val
            ]
        /\ msgs' = msgs \cup {
            [ type |-> "phase2b",
              ins  |-> m.ins,
              bal  |-> m.bal,
              val  |-> m.val,
              acc  |-> acc ]
          }
        /\ UNCHANGED <<rmState>>

PCNext == 
    \/ \E rm \in RM :
        \/ RMPrepared(rm)
        \/ RMChooseToAbort(rm)
        \/ RMRcvCommitMsg(rm)
        \/ RMRcvAbortMsg(rm)
    \/ \E bal \in Ballot \ {0} :
        \E rm \in RM :
            \/ Phase1a(bal, rm)
            \/ Phase2a(bal, rm)
    \/ Decide
    \/ \E acc \in Acceptor : Phase1b(acc) \/ Phase2b(acc)

PCSpec == PCInit /\[][PCNext]_<<rmState, aState, msgs>>
    
=============================================================================
\* Modification History
\* Last modified Tue Nov 04 12:09:54 CST 2025 by 老实人
\* Created Tue Nov 04 12:03:40 CST 2025 by 老实人
