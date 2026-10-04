---- MODULE GGpuTwoPhaseStructuresSmall_TTrace_1784086050 ----
EXTENDS Sequences, GGpuTwoPhaseStructuresSmall, TLCExt, Toolbox, Naturals, TLC

_expression ==
    LET GGpuTwoPhaseStructuresSmall_TEExpression == INSTANCE GGpuTwoPhaseStructuresSmall_TEExpression
    IN GGpuTwoPhaseStructuresSmall_TEExpression!expression
----

_trace ==
    LET GGpuTwoPhaseStructuresSmall_TETrace == INSTANCE GGpuTwoPhaseStructuresSmall_TETrace
    IN GGpuTwoPhaseStructuresSmall_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        msgs = ({[type |-> "Commit"], [type |-> "Prepared", rm |-> "rm1"]})
        /\
        rmState = ([rm1 |-> "committed", rm2 |-> "committed"])
        /\
        seenType = ("Commit")
    )
----

_init ==
    /\ msgs = _TETrace[1].msgs
    /\ rmState = _TETrace[1].rmState
    /\ seenType = _TETrace[1].seenType
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ msgs  = _TETrace[i].msgs
        /\ msgs' = _TETrace[j].msgs
        /\ rmState  = _TETrace[i].rmState
        /\ rmState' = _TETrace[j].rmState
        /\ seenType  = _TETrace[i].seenType
        /\ seenType' = _TETrace[j].seenType

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("GGpuTwoPhaseStructuresSmall_TTrace_1784086050.json", _TETrace)

=============================================================================

 Note that you can extract this module `GGpuTwoPhaseStructuresSmall_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `GGpuTwoPhaseStructuresSmall_TEExpression.tla` file takes precedence 
  over the module `GGpuTwoPhaseStructuresSmall_TEExpression` below).

---- MODULE GGpuTwoPhaseStructuresSmall_TEExpression ----
EXTENDS Sequences, GGpuTwoPhaseStructuresSmall, TLCExt, Toolbox, Naturals, TLC

expression == 
    [
        \* To hide variables of the `GGpuTwoPhaseStructuresSmall` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        msgs |-> msgs
        ,rmState |-> rmState
        ,seenType |-> seenType
        
        \* Put additional constant-, state-, and action-level expressions here:
        \* ,_stateNumber |-> _TEPosition
        \* ,_msgsUnchanged |-> msgs = msgs'
        
        \* Format the `msgs` variable as Json value.
        \* ,_msgsJson |->
        \*     LET J == INSTANCE Json
        \*     IN J!ToJson(msgs)
        
        \* Lastly, you may build expressions over arbitrary sets of states by
        \* leveraging the _TETrace operator.  For example, this is how to
        \* count the number of times a spec variable changed up to the current
        \* state in the trace.
        \* ,_msgsModCount |->
        \*     LET F[s \in DOMAIN _TETrace] ==
        \*         IF s = 1 THEN 0
        \*         ELSE IF _TETrace[s].msgs # _TETrace[s-1].msgs
        \*             THEN 1 + F[s-1] ELSE F[s-1]
        \*     IN F[_TEPosition - 1]
    ]

=============================================================================



Parsing and semantic processing can take forever if the trace below is long.
 In this case, it is advised to uncomment the module below to deserialize the
 trace from a generated binary file.

\*
\*---- MODULE GGpuTwoPhaseStructuresSmall_TETrace ----
\*EXTENDS IOUtils, GGpuTwoPhaseStructuresSmall, TLC
\*
\*trace == IODeserialize("GGpuTwoPhaseStructuresSmall_TTrace_1784086050.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE GGpuTwoPhaseStructuresSmall_TETrace ----
EXTENDS GGpuTwoPhaseStructuresSmall, TLC

trace == 
    <<
    ([msgs |-> {},rmState |-> [rm1 |-> "working", rm2 |-> "working"],seenType |-> "none"]),
    ([msgs |-> {[type |-> "Prepared", rm |-> "rm1"]},rmState |-> [rm1 |-> "prepared", rm2 |-> "working"],seenType |-> "Prepared"]),
    ([msgs |-> {[type |-> "Commit"], [type |-> "Prepared", rm |-> "rm1"]},rmState |-> [rm1 |-> "committed", rm2 |-> "working"],seenType |-> "Commit"]),
    ([msgs |-> {[type |-> "Commit"], [type |-> "Prepared", rm |-> "rm1"]},rmState |-> [rm1 |-> "committed", rm2 |-> "committed"],seenType |-> "Commit"])
    >>
----


=============================================================================

---- CONFIG GGpuTwoPhaseStructuresSmall_TTrace_1784086050 ----

INVARIANT
    _inv

CHECK_DEADLOCK
    \* CHECK_DEADLOCK off because of PROPERTY or INVARIANT above.
    FALSE

INIT
    _init

NEXT
    _next

CONSTANT
    _TETrace <- _trace

ALIAS
    _expression
=============================================================================
\* Generated on Wed Jul 15 11:27:31 ULAT 2026