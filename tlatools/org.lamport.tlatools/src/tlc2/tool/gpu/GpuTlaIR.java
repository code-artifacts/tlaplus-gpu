package tlc2.tool.gpu;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;

import tla2sany.semantic.SymbolNode;

public final class GpuTlaIR {
    public static final int OP_LOAD_CONST = 1;
    public static final int OP_LOAD_VAR = 2;
    public static final int OP_LOAD_NEXT_VAR = 3;
    public static final int OP_EQ = 4;
    public static final int OP_MEMBER = 5;
    public static final int OP_FCN_APPLY = 6;
    public static final int OP_AND = 7;
    public static final int OP_OR = 8;
    public static final int OP_NOT = 9;
    public static final int OP_NEQ = 10;
    public static final int OP_NOT_MEMBER = 11;
    public static final int OP_IMPLIES = 12;
    public static final int OP_EQUIV = 13;
    public static final int OP_SUBSETEQ = 14;
    public static final int OP_JUMP_IF_FALSE = 15;
    public static final int OP_JUMP = 16;
    public static final int OP_RETURN = 17;
    public static final int OP_ADD = 18;
    public static final int OP_SUB = 19;
    public static final int OP_MUL = 20;
    public static final int OP_LT = 21;
    public static final int OP_LE = 22;
    public static final int OP_GT = 23;
    public static final int OP_GE = 24;
    public static final int OP_INTERVAL = 25;
    public static final int OP_NEG = 26;
    public static final int OP_DOMAIN = 27;
    public static final int OP_UNION = 28;
    public static final int OP_SUBSET = 29;
    public static final int OP_SET_UNION = 30;
    public static final int OP_SET_INTERSECT = 31;
    public static final int OP_SET_DIFF = 32;
    public static final int OP_SET_ENUM = 33;
    public static final int OP_RECORD_ONE = 34;
    public static final int OP_RECORD_APPEND = 35;
    public static final int OP_RECORD_SELECT = 36;
    public static final int OP_EXCEPT_UPDATE = 37;
    public static final int OP_RECORD_SET_ONE = 38;
    public static final int OP_RECORD_SET_APPEND = 39;
    public static final int OP_SET_OF_FCNS = 40;
    public static final int OP_LOAD_LOCAL = 41;
    public static final int OP_TUPLE = 42;
    public static final int OP_EXISTS_BEGIN = 43;
    public static final int OP_EXISTS_NEXT = 44;
    public static final int OP_FORALL_BEGIN = 45;
    public static final int OP_FORALL_NEXT = 46;
    public static final int OP_FILTER_BEGIN = 47;
    public static final int OP_FILTER_NEXT = 48;
    public static final int OP_MAP_BEGIN = 49;
    public static final int OP_MAP_NEXT = 50;
    public static final int OP_EXCEPT_SAVE_AT = 51;
    public static final int OP_SAVE_LOCAL = 52;
    public static final int OP_CHOOSE_CHECK = 53;
    public static final int OP_CHOOSE_BEGIN = 54;
    public static final int OP_CHOOSE_NEXT = 55;
    public static final int OP_FCN_BEGIN = 56;
    public static final int OP_FCN_NEXT = 57;
    public static final int OP_CARTESIAN = 58;

    public static final int STATUS_OK = 0;
    public static final int STATUS_STACK_OVERFLOW = 1;
    public static final int STATUS_STACK_UNDERFLOW = 2;
    public static final int STATUS_TYPE_ERROR = 3;
    public static final int STATUS_BAD_OPCODE = 4;
    public static final int STATUS_APPLY_FAILED = 5;
    public static final int STATUS_NO_RETURN = 6;
    public static final int STATUS_HEAP_OVERFLOW = 7;
    public static final int STATUS_CHOOSE_FAILED = 8;

    private final long[] program;
    private final long[] constantHeap;
    private final String[] variableNames;

    GpuTlaIR(final long[] program, final long[] constantHeap, final String[] variableNames) {
        this.program = program;
        this.constantHeap = constantHeap;
        this.variableNames = variableNames;

    }
    public long[] program() {
        return program;
    }

    public long[] constantHeap() {
        return constantHeap;
    }

    public String[] variableNames() {
        return variableNames;
    }

    public EvaluationInput prepare(final ValueEncoder.EncodedSemanticState current,
            final ValueEncoder.EncodedSemanticState next) {
        if (!Arrays.equals(current.variableNames, variableNames) || !Arrays.equals(next.variableNames, variableNames)) {
            throw new IllegalArgumentException("IR variable order does not match encoded state variable order");
        }
        final int currentBase = 0;
        final int nextBase = current.heap.length;
        final int constBase = current.heap.length + next.heap.length;

        final long[] heap = new long[current.heap.length + next.heap.length + constantHeap.length];
        ValueEncoder.copySemanticHeap(current.heap, heap, currentBase);
        ValueEncoder.copySemanticHeap(next.heap, heap, nextBase);
        ValueEncoder.copySemanticHeap(constantHeap, heap, constBase);

        final long[] currentRoots = new long[current.roots.length];
        final long[] nextRoots = new long[next.roots.length];
        for (int i = 0; i < current.roots.length; i++) {
            currentRoots[i] = currentBase + current.roots[i];
            nextRoots[i] = nextBase + next.roots[i];
        }

        final long[] rebasedProgram = new long[program.length];
        for (int i = 0; i < program.length; i += 2) {
            final long op = program[i];
            long operand = program[i + 1];
            if (op == OP_LOAD_CONST) {
                operand += constBase;
            }
            rebasedProgram[i] = op;
            rebasedProgram[i + 1] = operand;
        }
        return new EvaluationInput(rebasedProgram, heap, currentRoots, nextRoots);
    }

    public static final class EvaluationInput {
        public final long[] program;
        public final long[] heap;
        public final long[] currentRoots;
        public final long[] nextRoots;

        private EvaluationInput(final long[] program, final long[] heap, final long[] currentRoots, final long[] nextRoots) {
            this.program = program;
            this.heap = heap;
            this.currentRoots = currentRoots;
            this.nextRoots = nextRoots;
        }
    }

    static final class Builder {
        private final ArrayList<Long> code = new ArrayList<Long>();
        private final ArrayList<Long> constants = new ArrayList<Long>();
        private final String[] variableNames;
        private final LocalSlots locals;

        Builder(final String[] variableNames) {
            this(variableNames, new LocalSlots());
        }

        Builder(final String[] variableNames, final LocalSlots locals) {
            this.variableNames = variableNames;
            this.locals = locals;
        }

        LocalSlots locals() { return locals; }

        int currentOffset() {
            return code.size();
        }

        void emit(final int op) {
            emit(op, 0L);
        }

        int emitJump(final int op) {
            final int offset = currentOffset();
            emit(op, 0L);
            return offset;
        }

        void patchOperand(final int offset, final long operand) {
            code.set(offset + 1, Long.valueOf(operand));
        }

        void emit(final int op, final long operand) {
            code.add(Long.valueOf(op));
            code.add(Long.valueOf(operand));
        }

        int addConstant(final tlc2.value.IValue value) {
            final ValueEncoder.EncodedSemanticValue encoded = ValueEncoder.encodeSemanticValue(value);
            return addEncodedConstant(encoded);
        }

        int addSyntheticOpaque(final String namespace, final long identity) {
            return addEncodedConstant(ValueEncoder.encodeSyntheticOpaque(namespace, identity));
        }

        private int addEncodedConstant(final ValueEncoder.EncodedSemanticValue encoded) {
            final int base = constants.size();
            final long[] rebased = ValueEncoder.rebasedSemanticHeap(encoded.heap, base);
            for (int i = 0; i < rebased.length; i++) {
                constants.add(Long.valueOf(rebased[i]));
            }
            return base + encoded.root;
        }

        GpuTlaIR build() {
            final long[] program = new long[code.size()];
            for (int i = 0; i < code.size(); i++) program[i] = code.get(i).longValue();
            final long[] heap = new long[constants.size()];
            for (int i = 0; i < constants.size(); i++) heap[i] = constants.get(i).longValue();
            return new GpuTlaIR(program, heap, variableNames);
        }
    }
    /** Identity-based slots for runtime-bound TLA+ formals and EXCEPT at. */
    static final class LocalSlots {
        private static final int MAX_SLOTS = 64;
        private final IdentityHashMap<Object, Integer> slots = new IdentityHashMap<Object, Integer>();
        int slot(final Object symbol) {
            Integer existing = slots.get(symbol);
            if (existing != null) return existing.intValue();
            if (slots.size() >= MAX_SLOTS) {
                throw new UnsupportedOperationException("GPU TLA+ IR supports at most 64 runtime local bindings");
            }
            final int result = slots.size();
            slots.put(symbol, Integer.valueOf(result));
            return result;
        }
        boolean contains(final Object symbol) { return slots.containsKey(symbol); }
        int size() { return slots.size(); }
    }
}
