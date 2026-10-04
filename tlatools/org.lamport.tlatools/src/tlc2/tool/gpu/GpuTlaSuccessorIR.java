package tlc2.tool.gpu;

import java.util.ArrayList;
import java.util.Arrays;

public final class GpuTlaSuccessorIR {
    public static final int OP_ASSIGN = 1;
    public static final int OP_ENUM = 2;
    public static final int OP_FILTER = 3;
    public static final int OP_ENUM_LOCAL = 4;

    public static final int STATUS_OK = 0;
    public static final int STATUS_CAPACITY_EXCEEDED = 1;
    public static final int STATUS_BAD_PROGRAM = 2;
    public static final int STATUS_UNSUPPORTED_ENUM = 3;
    public static final int STATUS_EVAL_FAILED = 4;
    public static final int STATUS_HEAP_OVERFLOW = 5;
    public static final int STATUS_SHARE_OVERFLOW = 6;

    private final long[] branchTable;
    private final long[] opTable;
    private final long[] exprProgram;
    private final long[] exprStarts;
    private final long[] exprLens;
    private final long[] constantHeap;
    private final String[] variableNames;

    GpuTlaSuccessorIR(final long[] branchTable, final long[] opTable, final long[] exprProgram,
            final long[] exprStarts, final long[] exprLens, final long[] constantHeap, final String[] variableNames) {
        this.branchTable = branchTable;
        this.opTable = opTable;
        this.exprProgram = exprProgram;
        this.exprStarts = exprStarts;
        this.exprLens = exprLens;
        this.constantHeap = constantHeap;
        this.variableNames = variableNames;
    }

    public long[] branchTable() {
        return branchTable;
    }

    public long[] opTable() {
        return opTable;
    }

    public long[] exprProgram() {
        return exprProgram;
    }

    public long[] exprStarts() {
        return exprStarts;
    }

    public long[] exprLens() {
        return exprLens;
    }

    public long[] constantHeap() {
        return constantHeap;
    }

    public ExpansionInput prepare(final ValueEncoder.EncodedSemanticState current, final int maxSuccessors) {
        if (!Arrays.equals(current.variableNames, variableNames)) {
            throw new IllegalArgumentException("successor IR variable order does not match encoded state variable order");
        }
        final int constBase = current.heap.length;
        final long[] heap = new long[current.heap.length + constantHeap.length];
        ValueEncoder.copySemanticHeap(current.heap, heap, 0);
        ValueEncoder.copySemanticHeap(constantHeap, heap, constBase);

        final long[] roots = new long[current.roots.length];
        for (int i = 0; i < current.roots.length; i++) {
            roots[i] = current.roots[i];
        }

        final long[] rebasedExprProgram = new long[exprProgram.length];
        for (int i = 0; i < exprProgram.length; i += 2) {
            final long op = exprProgram[i];
            long operand = exprProgram[i + 1];
            if (op == GpuTlaIR.OP_LOAD_CONST) {
                operand += constBase;
            }
            rebasedExprProgram[i] = op;
            rebasedExprProgram[i + 1] = operand;
        }
        return new ExpansionInput(branchTable, opTable, rebasedExprProgram, exprStarts, exprLens, heap, roots, maxSuccessors);
    }

    public static final class ExpansionInput {
        public final long[] branchTable;
        public final long[] opTable;
        public final long[] exprProgram;
        public final long[] exprStarts;
        public final long[] exprLens;
        public final long[] heap;
        public final long[] currentRoots;
        public final int maxSuccessors;

        private ExpansionInput(final long[] branchTable, final long[] opTable, final long[] exprProgram,
                final long[] exprStarts, final long[] exprLens, final long[] heap, final long[] currentRoots,
                final int maxSuccessors) {
            this.branchTable = branchTable;
            this.opTable = opTable;
            this.exprProgram = exprProgram;
            this.exprStarts = exprStarts;
            this.exprLens = exprLens;
            this.heap = heap;
            this.currentRoots = currentRoots;
            this.maxSuccessors = maxSuccessors;
        }
    }

    static final class Builder {
        private final ArrayList<Long> branchTable = new ArrayList<Long>();
        private final ArrayList<Long> opTable = new ArrayList<Long>();
        private final ArrayList<Long> exprProgram = new ArrayList<Long>();
        private final ArrayList<Long> exprStarts = new ArrayList<Long>();
        private final ArrayList<Long> exprLens = new ArrayList<Long>();
        private final ArrayList<Long> constantHeap = new ArrayList<Long>();
        private final String[] variableNames;
        private final GpuTlaIR.LocalSlots locals;

        Builder(final String[] variableNames) {
            this(variableNames, new GpuTlaIR.LocalSlots());
        }

        Builder(final String[] variableNames, final GpuTlaIR.LocalSlots locals) {
            this.variableNames = variableNames;
            this.locals = locals;
        }

        GpuTlaIR.LocalSlots locals() { return locals; }

        int beginBranch() {
            final int start = opTable.size() / 3;
            branchTable.add(Long.valueOf(start));
            branchTable.add(Long.valueOf(0L));
            return branchTable.size() - 2;
        }

        void endBranch(final int branchIndex) {
            final int start = branchTable.get(branchIndex).intValue();
            final int end = opTable.size() / 3;
            branchTable.set(branchIndex + 1, Long.valueOf(end - start));
        }

        void addOp(final int op, final int varIndex, final int exprIndex) {
            opTable.add(Long.valueOf(op));
            opTable.add(Long.valueOf(varIndex));
            opTable.add(Long.valueOf(exprIndex));
        }

        int addExpr(final GpuTlaIR ir) {
            final int constBase = constantHeap.size();
            final long[] rebasedConstants = ValueEncoder.rebasedSemanticHeap(ir.constantHeap(), constBase);
            for (long word : rebasedConstants) {
                constantHeap.add(Long.valueOf(word));
            }
            final int start = exprProgram.size();
            final long[] program = ir.program();
            for (int i = 0; i < program.length; i += 2) {
                exprProgram.add(Long.valueOf(program[i]));
                long operand = program[i + 1];
                if (program[i] == GpuTlaIR.OP_LOAD_CONST) {
                    operand += constBase;
                }
                exprProgram.add(Long.valueOf(operand));
            }
            exprStarts.add(Long.valueOf(start));
            exprLens.add(Long.valueOf(program.length));
            return exprStarts.size() - 1;
        }

        GpuTlaSuccessorIR build() {
            return new GpuTlaSuccessorIR(toArray(branchTable), toArray(opTable), toArray(exprProgram),
                    toArray(exprStarts), toArray(exprLens), toArray(constantHeap), variableNames);
        }

        private static long[] toArray(final ArrayList<Long> values) {
            final long[] out = new long[values.size()];
            for (int i = 0; i < values.size(); i++) out[i] = values.get(i).longValue();
            return out;
        }
    }
}
