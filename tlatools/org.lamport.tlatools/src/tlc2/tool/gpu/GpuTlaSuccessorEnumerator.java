package tlc2.tool.gpu;

public final class GpuTlaSuccessorEnumerator {
    private GpuTlaSuccessorEnumerator() {
    }

    public static Result expand(final GpuTlaSuccessorIR ir, final ValueEncoder.EncodedSemanticState current,
            final int maxSuccessors) {
        return expand(ir.prepare(current, maxSuccessors));
    }

    public static Result expand(final GpuTlaSuccessorIR.ExpansionInput input) {
        final long[] raw = GPUKernel.expandSuccessors(input.branchTable, input.opTable, input.exprProgram,
                input.exprStarts, input.exprLens, input.heap, input.currentRoots, input.maxSuccessors);
        if (raw == null || raw.length < 4) {
            throw new IllegalStateException("GPU successor enumerator returned an invalid result packet");
        }
        long[] returnedHeap = new long[0];
        final int status = (int) raw[0];
        final int varCount = (int) raw[1];
        final int count = (int) raw[2];
        final boolean overflow = raw[3] != 0L;
        final long fixedWords = 4L + count + (long) count * varCount;
        if (count < 0 || varCount < 0 || fixedWords > raw.length) {
            throw new IllegalStateException("GPU successor enumerator returned invalid dimensions");
        }
        final long[] fingerprints = new long[count];
        final long[][] roots = new long[count][varCount];
        int cursor = 4;
        for (int i = 0; i < count; i++) fingerprints[i] = raw[cursor++];
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < varCount; j++) roots[i][j] = raw[cursor++];
        }
        if (raw.length > cursor) {
            final int heapWords = (int) raw[cursor++];
            if (heapWords < 0 || raw.length - cursor < heapWords) {
                throw new IllegalStateException("GPU successor enumerator returned an invalid heap packet");
            }
            returnedHeap = new long[heapWords];
            System.arraycopy(raw, cursor, returnedHeap, 0, heapWords);
        }
        return new Result(status, overflow, fingerprints, roots, returnedHeap);
    }

    public static final class Result {
        public final int status;
        public final boolean overflow;
        public final long[] fingerprints;
        public final long[][] roots;
        public final long[] heap;

        private Result(final int status, final boolean overflow, final long[] fingerprints,
                final long[][] roots, final long[] heap) {
            this.status = status;
            this.overflow = overflow;
            this.fingerprints = fingerprints;
            this.roots = roots;
            this.heap = heap;
        }

        public boolean ok() {
            return status == GpuTlaSuccessorIR.STATUS_OK;
        }
    }
}
