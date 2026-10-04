package tlc2.tool.gpu;

public final class GpuTlaEvaluator {
    private GpuTlaEvaluator() {
    }

    public static Result evaluate(final GpuTlaIR ir,
            final ValueEncoder.EncodedSemanticState current,
            final ValueEncoder.EncodedSemanticState next) {
        return evaluate(ir.prepare(current, next));
    }

    public static Result evaluate(final GpuTlaIR.EvaluationInput input) {
        final long[] raw = GPUKernel.evalIR(input.program, input.heap, input.currentRoots, input.nextRoots);
        if (raw == null || raw.length < 4) {
            throw new IllegalStateException("GPU IR evaluator returned an invalid result packet");
        }
        return new Result((int) raw[0], raw[1] != 0L, (int) raw[2], raw[3]);
    }

    public static final class Result {
        public final int status;
        public final boolean boolValue;
        public final int resultOffset;
        public final long fingerprint;

        private Result(final int status, final boolean boolValue, final int resultOffset, final long fingerprint) {
            this.status = status;
            this.boolValue = boolValue;
            this.resultOffset = resultOffset;
            this.fingerprint = fingerprint;
        }

        public boolean ok() {
            return status == GpuTlaIR.STATUS_OK;
        }
    }
}
