package tlc2.tool.gpu;

final class GpuTlaBatchDeduplicator {
    private GpuTlaBatchDeduplicator() {
    }

    static long fingerprint(final ValueEncoder.EncodedSemanticState state) {
        final long[] roots = new long[state.roots.length];
        for (int i = 0; i < roots.length; i++) roots[i] = state.roots[i];
        final long[] raw = GPUKernel.fingerprintState(state.heap, roots);
        if (raw == null || raw.length != 1) {
            throw new IllegalStateException("GPU state fingerprint returned an invalid result");
        }
        return raw[0];
    }

    static boolean[] firstOccurrences(final long[] fingerprints) {
        final long[] raw = GPUKernel.deduplicateFingerprints(fingerprints);
        if (raw == null || raw.length != fingerprints.length) {
            throw new IllegalStateException("GPU batch deduplicator returned an invalid result");
        }
        final boolean[] keep = new boolean[raw.length];
        for (int i = 0; i < raw.length; i++) keep[i] = raw[i] != 0L;
        return keep;
    }
}
