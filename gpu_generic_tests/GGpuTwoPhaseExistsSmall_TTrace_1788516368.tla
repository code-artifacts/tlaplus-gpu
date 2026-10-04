---- MODULE GGpuTwoPhaseExistsSmall_TTrace_1788516368 ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, GGpuTwoPhaseExistsSmall

_expression ==
    LET GGpuTwoPhaseExistsSmall_TEExpression == INSTANCE GGpuTwoPhaseExistsSmall_TEExpression
    IN GGpuTwoPhaseExistsSmall_TEExpression!expression
----

_trace ==
    LET GGpuTwoPhaseExistsSmall_TETrace == INSTANCE GGpuTwoPhaseExistsSmall_TETrace
    IN GGpuTwoPhaseExistsSmall_TETrace!trace
----

_inv ==
    ~(
        TLCGet("level") = Len(_TETrace)
        /\
        msgs = ({[rm |-> "rm1", type |-> "Prepared"], [rm |-> "rm2", type |-> "Prepared"]})
        /\
        rmState = ([rm1 |-> "prepared", rm2 |-> "prepared"])
    )
----

_init ==
    /\ msgs = _TETrace[1].msgs
    /\ rmState = _TETrace[1].rmState
----

_next ==
    /\ \E i,j \in DOMAIN _TETrace:
        /\ \/ /\ j = i + 1
              /\ i = TLCGet("level")
        /\ msgs  = _TETrace[i].msgs
        /\ msgs' = _TETrace[j].msgs
        /\ rmState  = _TETrace[i].rmState
        /\ rmState' = _TETrace[j].rmState

\* Uncomment the ASSUME below to write the states of the error trace
\* to the given file in Json format. Note that you can pass any tuple
\* to `JsonSerialize`. For example, a sub-sequence of _TETrace.
    \* ASSUME
    \*     LET J == INSTANCE Json
    \*         IN J!JsonSerialize("GGpuTwoPhaseExistsSmall_TTrace_1788516368.json", _TETrace)

=============================================================================

 Note that you can extract this module `GGpuTwoPhaseExistsSmall_TEExpression`
  to a dedicated file to reuse `expression` (the module in the 
  dedicated `GGpuTwoPhaseExistsSmall_TEExpression.tla` file takes precedence 
  over the module `GGpuTwoPhaseExistsSmall_TEExpression` below).

---- MODULE GGpuTwoPhaseExistsSmall_TEExpression ----
EXTENDS Sequences, TLCExt, Toolbox, Naturals, TLC, GGpuTwoPhaseExistsSmall

expression == 
    [
        \* To hide variables of the `GGpuTwoPhaseExistsSmall` spec from the error trace,
        \* remove the variables below.  The trace will be written in the order
        \* of the fields of this record.
        msgs |-> msgs
        ,rmState |-> rmState
        
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
\*---- MODULE GGpuTwoPhaseExistsSmall_TETrace ----
\*EXTENDS IOUtils, TLC, GGpuTwoPhaseExistsSmall
\*
\*trace == IODeserialize("GGpuTwoPhaseExistsSmall_TTrace_1788516368.bin", TRUE)
\*
\*=============================================================================
\*

---- MODULE GGpuTwoPhaseExistsSmall_TETrace ----
EXTENDS TLC, GGpuTwoPhaseExistsSmall

trace == 
    <<
    ([msgs |-> {},rmState |-> [rm1 |-> "working", rm2 |-> "working"]]),
    ([msgs |-> {[rm |-> "rm1", type |-> "Prepared"]},rmState |-> [rm1 |-> "prepared", rm2 |-> "working"]]),
    ([msgs |-> {[rm |-> "rm1", type |-> "Prepared"], [rm |-> "rm2", type |-> "Prepared"]},rmState |-> [rm1 |-> "prepared", rm2 |-> "prepared"]])
    >>
----


=============================================================================

---- CONFIG GGpuTwoPhaseExistsSmall_TTrace_1788516368 ----

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
\* Generated on Fri Sep 04 18:06:09 ULAT 2026