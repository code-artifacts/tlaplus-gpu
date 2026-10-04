# GPU Generic TLA+ Semantic Objects

This document describes the migration path from the current TwoPhase-specific GPU resident BFS backend to a generic GPU backend that can carry TLA+ semantic values.

## Current state

The verified fast path is specialized for `TwoPhase.tla`:

```text
TLCState init
  -> TwoPhase Java encoder
  -> 64-bit packed state
  -> CUDA resident frontier
  -> CUDA resident fingerprint set
  -> CUDA hand-written TPNext slots
```

This is fast because it does not materialize general TLC `Value` objects on the GPU. It encodes the complete TwoPhase state into one `uint64` and applies `TPNext` with bit operations.

## Target architecture

The generic backend should not copy Java object graphs to the GPU. Java object pointers are meaningless on the device and the object overhead would destroy the memory advantage. The target is a compact ABI:

```text
TLC Value graph
  -> flat GPU value heap
  -> state root table
  -> GPU IR evaluator
  -> GPU successor enumerator
  -> GPU fingerprint/frontier/trace tables
```

The CPU remains responsible for parsing, semantic analysis, model configuration, and final reporting. The GPU becomes responsible for evaluating compiled Init/Next/Invariant/Constraint IR over the value heap.

## Phase 1 implemented here: semantic value ABI

Implemented files:

- `tlatools/org.lamport.tlatools/src/tlc2/tool/gpu/ValueEncoder.java`
- `tlatools/org.lamport.tlatools/src/tlc2/tool/gpu/GPUNextState.cu`

The Java side now exposes a structured encoder in addition to the old scalar hash encoder:

```java
ValueEncoder.encodeSemanticValue(IValue)
ValueEncoder.encodeSemanticState(TLCState)
```

The encoded heap is a `long[]`. Each object starts with one header word:

```text
bits 63..56: tag
bits 55..32: arity
bits 31..0 : aux
```

Compound payload words store child heap offsets. Primitive payload words store immediate values or stable symbol ids.

Supported first-pass tags:

```text
0   NULL
1   BOOL
2   INT
3   MODEL
4   TUPLE
5   SET_ENUM
6   FCN_RCD
7   RECORD
8   INTERVAL
255 OPAQUE fallback
```

Function records support two domain encodings:

```text
0 explicit domain: [domain child offsets][value child offsets]
1 interval domain: [low][high][value child offsets]
```

The CUDA side now has matching constants and helpers:

```cpp
gpuTlaTag(header)
gpuTlaArity(header)
gpuTlaAux(header)
gpuTlaChild(ref, index)
```

This ABI is intentionally separate from the TwoPhase resident kernels, so the verified TwoPhase path is preserved.

## Phase 2: GPU semantic operators

Implemented first-pass device functions over the ABI:

```text
gpuTlaEquals
gpuTlaMember
gpuTlaFcnApply
gpuTlaFingerprint
```

Still to add before the IR evaluator can cover a broad TLA+ subset:

```text
gpuTlaCompare where TLC allows comparison
gpuTlaRecordSelect
gpuTlaTupleSelect
gpuTlaSetUnion
gpuTlaSetInsert
```

The first practical subset should cover protocol specs:

```text
Bool, Int, ModelValue, Tuple, Record, finite SetEnum, finite FcnRcd, Interval
```

Opaque fallback values must be rejected by GPU evaluation unless an operation explicitly permits hash-only treatment.

## Phase 3: TLA+ expression IR

Implemented first-pass stack IR and evaluator files:

```text
GpuTlaIR.java
GpuTlaIRCompiler.java
GpuTlaEvaluator.java
GPUKernel.evalIR(...)
gpuTlaEvalIrKernel(...)
```

Currently supported IR operations:

```text
LOAD_CONST
LOAD_VAR
LOAD_NEXT_VAR
EQ
MEMBER
FCN_APPLY
AND
OR
NOT
RETURN
```

The compiler currently covers a bounded predicate subset:

```text
state variables
primed state variables
constant expressions evaluable by TLC
=
\in
/\ and conjunction lists
\/ and disjunction lists
function application
UNCHANGED var
UNCHANGED <<vars>>
```

The evaluator currently checks one candidate `(s0, s1)` pair against an IR predicate. It does not yet enumerate successor bindings for `x' \in S` or bounded existential branches; that is the next step for full generic successor generation.

Original target shape for full `Next` generation remains:

```text
frontier state id
  -> load state roots
  -> evaluate Next IR
  -> enumerate existential bindings
  -> allocate successor state roots
  -> evaluate constraints/invariants
  -> fingerprint/dedup
  -> append next frontier
```

## Phase 4: Generic successor enumeration

Implemented first-pass successor IR and GPU state builder files:

```text
GpuTlaSuccessorIR.java
GpuTlaSuccessorCompiler.java
GpuTlaSuccessorEnumerator.java
GPUKernel.expandSuccessors(...)
gpuTlaExpandSuccessorsKernel(...)
```

Currently supported Next-generation subset:

```text
x' = expr
x' \in finite SetEnum
UNCHANGED var
UNCHANGED <<vars>>
/\ conjunction branches
\/ disjunction branches
extra boolean predicates as FILTER
```

The state builder starts each branch by copying current roots, overwrites assigned primed variables, enumerates finite set choices, filters candidates, and returns successor root tables plus structure fingerprints. It currently reuses existing heap values and constants; it does not yet allocate newly constructed compound values on GPU. Interval enumeration and bounded existential local variables are also not enabled in this first pass.

## Phase 5: TLC compatibility work

To approach general TLC semantics, add:

```text
predecessor/action trace tables
checkpoint/recovery metadata
fingerprint collision recovery or full-state equality fallback
invariant and action-property reporting
liveness handoff or GPU graph support
symmetry/view-map policy
```

## Non-goals of phase 1

Phase 1 does not replace `residentStepKernel`, does not evaluate arbitrary TLA+ actions, and does not make the existing GPU path a full TLC-compatible backend. It establishes the value/state representation needed for the next phase.
