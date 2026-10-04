package tlc2.tool.gpu;

final class GPUKernel {
    static {
        System.loadLibrary("tlc_gpu");
    }

    private GPUKernel() {
    }

    static native long[] evalIR(long[] program, long[] heap, long[] currentRoots, long[] nextRoots);
    static native long[] fingerprintState(long[] heap, long[] roots);
    static native long[] deduplicateFingerprints(long[] fingerprints);
    static native long[] expandSuccessors(long[] branchTable, long[] opTable, long[] exprProgram,
            long[] exprStarts, long[] exprLens, long[] heap, long[] currentRoots, int maxSuccessors);
    static native void genericFrontierInit(int maxStates, int varCount, int heapWords, int hashCapacity,
            int stackBytes, boolean gcEnabled, int gcThresholdPercent, int gcMaxSkippedLayers,
            int batchStates, int frontierMode, int gpuOutputHeapWords, int gpuOutputMaxStates,
            int gpuShareCapacity, int gpuJavaBatchBytes);
    static native void genericFrontierLoadInitial(long[] heap, long[] roots);
    static native void genericFrontierLoadSegment(long[] heap, long[] roots, int fixedPrefixWords);
    static native int genericFrontierStep(long[] branchTable, long[] opTable, long[] exprProgram,
            long[] exprStarts, long[] exprLens, long[] constantHeap);
    static native long[] genericFrontierRunSegment(long[] branchTable, long[] opTable, long[] exprProgram,
            long[] exprStarts, long[] exprLens, long[] constantHeap);
    static native Object[] genericFrontierRunGpuSegment(long[] dynamicHeap, int[] absoluteRoots,
            int[] segmentDescriptors, int fixedPrefixWords, long[] branchTable, long[] opTable,
            long[] exprProgram, long[] exprStarts, long[] exprLens, long[] constantHeap);
    static native void genericFrontierCommitLayer(int nextFrontierSize);
    static native int genericFrontierDrain(long[] branchTable, long[] opTable, long[] exprProgram,
            long[] exprStarts, long[] exprLens, long[] constantHeap, int maxDepth);
    static native long[] genericFrontierSnapshot();
    static native long[] genericFrontierStats();
    static native long[] genericFrontierTransferStats();
}
