package tlc2.tool.gpu;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

public final class GpuTlaGenericFrontier {
    private static int maxStates = Integer.getInteger("tlc.gpu.generic.max.states", 1000000);
    private static int hashCapacity = Integer.getInteger("tlc.gpu.generic.hash.capacity", 2097152);
    private static int baseHeapWords = 0;
    private static GpuTlaSuccessorIR cachedIR = null;
    private static long[] cachedExprProgram = null;
    private static long[] cachedConstantHeap = null;
    private static boolean segmented = false;
    private static boolean gpuSegmented = false;
    private static int segmentMaxStates = 0;
    private static int segmentBatchStates = 32768;
    private static Deque<HostSegment> currentSegments = new ArrayDeque<HostSegment>();
    private static Deque<HostSegment> nextSegments = new ArrayDeque<HostSegment>();
    private static Deque<GpuHostBatch> gpuCurrentSegments = new ArrayDeque<GpuHostBatch>();
    private static Deque<GpuHostBatch> gpuNextSegments = new ArrayDeque<GpuHostBatch>();
    private static long currentFrontierSize = 0L;
    private static int frontierDepth = 0;
    private static String[] variableNames = new String[0];
    private static ValueEncoder.EncodedSemanticState semanticTemplate = null;

    private GpuTlaGenericFrontier() {
    }

    public static void init(final int maxStatesHint, final int varCount, final int heapWords) {
        cachedIR = null;
        cachedExprProgram = null;
        cachedConstantHeap = null;
        final String frontierMode = System.getProperty("tlc.gpu.generic.frontier", "resident");
        segmented = "host-segmented".equalsIgnoreCase(frontierMode);
        gpuSegmented = "gpu-segmented".equalsIgnoreCase(frontierMode);
        if (!segmented && !gpuSegmented && !"resident".equalsIgnoreCase(frontierMode)) {
            throw new IllegalArgumentException("unknown generic GPU frontier mode: " + frontierMode);
        }
        currentSegments.clear();
        nextSegments.clear();
        currentFrontierSize = 0L;
        gpuCurrentSegments.clear();
        gpuNextSegments.clear();
        frontierDepth = 0;
        variableNames = new String[0];
        maxStates = Integer.getInteger("tlc.gpu.generic.max.states", maxStatesHint > 0 ? maxStatesHint : maxStates);
        semanticTemplate = null;
        hashCapacity = Integer.getInteger("tlc.gpu.generic.hash.capacity", hashCapacity);
        final int extraHeapWords = Integer.getInteger("tlc.gpu.generic.heap.extra.words", 1048576);
        // The generic IR now carries local bindings and bounded-loop frames. The
        // former 32 KiB default can overflow on valid TwoPhase successor kernels.
        final int stackBytes = Integer.getInteger("tlc.gpu.generic.stack.bytes", 65536);
        final boolean gcEnabled = Boolean.getBoolean("tlc.gpu.generic.gc.enabled");
        final int gcThresholdPercent = Integer.getInteger("tlc.gpu.generic.gc.threshold.percent", 70);
        final int gcMaxSkippedLayers = Integer.getInteger("tlc.gpu.generic.gc.max.skip.layers", 4);
        final int batchStates = Integer.getInteger("tlc.gpu.generic.batch.states", 32768);
        segmentBatchStates = Math.max(1024, batchStates);
        segmentMaxStates = Integer.getInteger("tlc.gpu.generic.frontier.segment.max.states",
                Math.max(262144, Math.min(maxStates, 1048576)));
        final long defaultAcceptedStates = Math.min(4194304L,
                Math.max((long) segmentMaxStates, (long) segmentMaxStates * 4L));
        final int gpuAcceptedMaxStates = Integer.getInteger(
                "tlc.gpu.generic.gpu.segment.accepted.max.states", (int) defaultAcceptedStates);
        if (gpuAcceptedMaxStates < segmentMaxStates) {
            throw new IllegalArgumentException("gpu-segmented accepted staging capacity must be at least "
                    + "the input segment capacity (accepted=" + gpuAcceptedMaxStates
                    + ", segment=" + segmentMaxStates + ")");
        }
        final int nativeMaxStates = segmented ? segmentMaxStates
                : (gpuSegmented ? Math.max(segmentMaxStates, gpuAcceptedMaxStates) : maxStates);
        final int requestedHeapWords = heapWords + Math.max(0, extraHeapWords);
        final int gpuOutputHeapWords = Math.max(1024, Integer.getInteger(
                "tlc.gpu.generic.gpu.segment.output.heap.words",
                Math.max(1048576, Math.min(requestedHeapWords, 16777216))));
        final int gpuOutputMaxStates = Math.max(1, Integer.getInteger(
                "tlc.gpu.generic.gpu.segment.output.max.states",
                Math.min(segmentMaxStates, Math.max(262144, batchStates))));
        if (gpuOutputMaxStates > segmentMaxStates) {
            throw new IllegalArgumentException("gpu-segmented output capacity must not exceed "
                    + "the logical segment capacity (output=" + gpuOutputMaxStates
                    + ", segment=" + segmentMaxStates + ")");
        }
        final int gpuShareCapacity = Math.max(1024, Integer.getInteger(
                "tlc.gpu.generic.gpu.segment.share.capacity",
                Math.min(4194304, Math.max(1048576, gpuOutputMaxStates * 4))));
        final int gpuJavaBatchBytes = Math.max(1048576, Integer.getInteger(
                "tlc.gpu.generic.gpu.segment.java.batch.bytes", 67108864));
        final int nativeMode = gpuSegmented ? 2 : (segmented ? 1 : 0);
        GPUKernel.genericFrontierInit(nativeMaxStates, varCount, requestedHeapWords, hashCapacity,
                Math.max(4096, stackBytes), gcEnabled, gcThresholdPercent, gcMaxSkippedLayers,
                Math.max(1024, batchStates), nativeMode, gpuOutputHeapWords,
                Math.min(nativeMaxStates, gpuOutputMaxStates), gpuShareCapacity, gpuJavaBatchBytes);
    }

    public static void loadInitial(final ValueEncoder.EncodedSemanticState initial) {
        baseHeapWords = initial.heap.length;
        variableNames = initial.variableNames.clone();
        semanticTemplate = initial;
        GPUKernel.genericFrontierLoadInitial(initial.heap, toLongRoots(initial.roots));
        if (GpuStateSetRecorder.enabled()) {
            GpuStateSetRecorder.recordGpu(0, variableNames, initial.heap, toLongRoots(initial.roots));
        }
        if (segmented) {
            currentSegments.addLast(HostSegment.fromEncoded(initial));
            currentFrontierSize = 1L;
        } else if (gpuSegmented) {
            gpuCurrentSegments.addLast(GpuHostBatch.initial(initial));
            currentFrontierSize = 1L;
        }
    }

    public static int step(final GpuTlaSuccessorIR ir) {
        if (segmented) {
            return stepSegmented(ir);
        }
        cacheProgram(ir);
        if (gpuSegmented) {
            return stepGpuSegmented(ir);
        }
        final int frontier = GPUKernel.genericFrontierStep(ir.branchTable(), ir.opTable(), cachedExprProgram,
                ir.exprStarts(), ir.exprLens(), cachedConstantHeap);
        if (GpuStateSetRecorder.enabled() && frontier > 0) {
            recordResidentSnapshot();
        }
        return frontier;
    }

    public static int drain(final GpuTlaSuccessorIR ir) {
        if (segmented) {
            final int maxDepth = Integer.getInteger("tlc.gpu.generic.max.depth", 1000000);
            int frontier = currentFrontierSize > Integer.MAX_VALUE ? Integer.MAX_VALUE
                    : (int) currentFrontierSize;
            int depth = 0;
            while (frontier > 0 && depth < maxDepth) {
                frontier = stepSegmented(ir);
                depth++;
            }
            return frontier;
        }
        if (gpuSegmented) {
            final int maxDepth = Integer.getInteger("tlc.gpu.generic.max.depth", 1000000);
            int frontier = currentFrontierSize > Integer.MAX_VALUE ? Integer.MAX_VALUE
                    : (int) currentFrontierSize;
            int depth = 0;
            while (frontier > 0 && depth < maxDepth) {
                frontier = stepGpuSegmented(ir);
                depth++;
            }
            return frontier;
        }
        cacheProgram(ir);
        if (GpuStateSetRecorder.enabled()) {
            int frontier = currentFrontierSize > Integer.MAX_VALUE ? Integer.MAX_VALUE
                    : (int) currentFrontierSize;
            int depth = 0;
            final int maxDepth = Integer.getInteger("tlc.gpu.generic.max.depth", 1000000);
            while (frontier > 0 && depth < maxDepth) {
                frontier = step(ir);
                depth++;
            }
            return frontier;
        }
        final int maxDepth = Integer.getInteger("tlc.gpu.generic.max.depth", 1000000);
        return GPUKernel.genericFrontierDrain(ir.branchTable(), ir.opTable(), cachedExprProgram,
                ir.exprStarts(), ir.exprLens(), cachedConstantHeap, maxDepth);
    }

    private static void recordResidentSnapshot() {
        final long[] packet = GPUKernel.genericFrontierSnapshot();
        final HostSegment snapshot = HostSegment.fromPacket(packet, GpuTlaSuccessorIR.STATUS_OK);
        if (snapshot == null || snapshot.stateCount() == 0) {
            throw new IllegalStateException("generic GPU resident validation snapshot is empty");
        }
        final long[] stats = GPUKernel.genericFrontierStats();
        final long expected = stats.length >= 3 ? stats[2] : -1L;
        if (expected != snapshot.stateCount()) {
            throw new IllegalStateException("generic GPU resident validation snapshot contains "
                    + snapshot.stateCount() + " states, expected " + expected);
        }
        final long depth = stats.length >= 4 ? stats[3] : -1L;
        if (depth < 0 || depth > Integer.MAX_VALUE) {
            throw new IllegalStateException("invalid generic GPU resident validation depth: " + depth);
        }
        GpuStateSetRecorder.recordGpu((int) depth, variableNames, snapshot.heap, snapshot.roots);
    }

    private static int stepSegmented(final GpuTlaSuccessorIR ir) {
        if (currentSegments.isEmpty()) {
            currentFrontierSize = 0L;
            GPUKernel.genericFrontierCommitLayer(0);
            return 0;
        }
        cacheProgram(ir);
        nextSegments.clear();
        final int nextDepth = frontierDepth + 1;
        long nextSize = 0L;
        while (!currentSegments.isEmpty()) {
            final HostSegment input = currentSegments.removeFirst();
            final int vars = GPUFrontier.variableCount();
            final int inputStates = input.stateCount();
            for (int first = 0; first < inputStates; first += segmentBatchStates) {
                final int count = Math.min(segmentBatchStates, inputStates - first);
                final long[] inputRoots = Arrays.copyOfRange(input.roots, first * vars, (first + count) * vars);
                final HostSegment inputBatch = new HostSegment(input.heap, inputRoots);
                final int prefix = fixedPrefixWords();
                final long[] gpuHeap = ValueEncoder.rebasedSemanticHeap(inputBatch.heap, prefix);
                final long[] gpuRoots = rebaseRoots(inputBatch.roots, prefix);
                GPUKernel.genericFrontierLoadSegment(gpuHeap, gpuRoots, prefix);
                final long[] packet = GPUKernel.genericFrontierRunSegment(ir.branchTable(), ir.opTable(),
                        cachedExprProgram, ir.exprStarts(), ir.exprLens(), cachedConstantHeap);
                final HostSegment output = HostSegment.fromPacket(packet, GpuTlaSuccessorIR.STATUS_OK);
                if (output != null && output.stateCount() > 0) {
                    GpuStateSetRecorder.recordGpu(nextDepth, variableNames, output.heap, output.roots);
                    final int states = output.stateCount();
                    for (int outputFirst = 0; outputFirst < states; outputFirst += segmentMaxStates) {
                        final int outputCount = Math.min(segmentMaxStates, states - outputFirst);
                        final long[] roots = Arrays.copyOfRange(output.roots,
                                outputFirst * vars, (outputFirst + outputCount) * vars);
                        // The heap is immutable for this output packet; root-only splits avoid
                        // copying the complete semantic value graph in CPU RAM.
                        nextSegments.addLast(new HostSegment(output.heap, roots));
                    }
                    nextSize += states;
                }
            }
        }
        currentSegments = nextSegments;
        nextSegments = new ArrayDeque<HostSegment>();
        currentFrontierSize = nextSize;
        if (nextSize > Integer.MAX_VALUE) {
            throw new IllegalStateException("CPU frontier size exceeds the current Java progress counter range");
        }
        frontierDepth = nextDepth;
        GPUKernel.genericFrontierCommitLayer((int) nextSize);
        return (int) nextSize;
    }

    private static int stepGpuSegmented(final GpuTlaSuccessorIR ir) {
        if (gpuCurrentSegments.isEmpty()) {
            currentFrontierSize = 0L;
            GPUKernel.genericFrontierCommitLayer(0);
            return 0;
        }
        cacheProgram(ir);
        gpuNextSegments.clear();
        final int nextDepth = frontierDepth + 1;
        long nextSize = 0L;
        while (!gpuCurrentSegments.isEmpty()) {
            final GpuHostBatch input = gpuCurrentSegments.removeFirst();
            final Object[] output = GPUKernel.genericFrontierRunGpuSegment(input.dynamicHeap,
                    input.absoluteRoots, input.segmentDescriptors, fixedPrefixWords(),
                    ir.branchTable(), ir.opTable(), cachedExprProgram, ir.exprStarts(),
                    ir.exprLens(), cachedConstantHeap);
            if (output == null || output.length % 3 != 0) {
                throw new IllegalStateException("generic GPU device-segmented output has invalid shape");
            }
            for (int item = 0; item < output.length; item += 3) {
                if (!(output[item] instanceof long[]) || !(output[item + 1] instanceof int[])
                        || !(output[item + 2] instanceof int[])) {
                    throw new IllegalStateException("generic GPU device-segmented output has invalid types");
                }
                final GpuHostBatch segment = new GpuHostBatch((long[]) output[item],
                        (int[]) output[item + 1], (int[]) output[item + 2]);
                if (GpuStateSetRecorder.enabled()) {
                    recordGpuBatch(nextDepth, segment);
                }
                final int states = segment.stateCount();
                gpuNextSegments.addLast(segment);
                nextSize += states;
            }
        }
        gpuCurrentSegments = gpuNextSegments;
        gpuNextSegments = new ArrayDeque<GpuHostBatch>();
        currentFrontierSize = nextSize;
        if (nextSize > Integer.MAX_VALUE) {
            throw new IllegalStateException("CPU frontier size exceeds the current Java progress counter range");
        }
        frontierDepth = nextDepth;
        GPUKernel.genericFrontierCommitLayer((int) nextSize);
        return (int) nextSize;
    }

    private static void recordGpuBatch(final int depth, final GpuHostBatch batch) {
        final long[] prefix = fixedPrefixHeap();
        for (int descriptor = 0; descriptor < batch.segmentDescriptors.length; descriptor += 4) {
            final int heapOffset = batch.segmentDescriptors[descriptor];
            final int heapLength = batch.segmentDescriptors[descriptor + 1];
            final int rootsOffset = batch.segmentDescriptors[descriptor + 2];
            final int rootsLength = batch.segmentDescriptors[descriptor + 3];
            final long[] heap = Arrays.copyOf(prefix, prefix.length + heapLength);
            System.arraycopy(batch.dynamicHeap, heapOffset, heap, prefix.length, heapLength);
            final int[] roots = Arrays.copyOfRange(batch.absoluteRoots,
                    rootsOffset, rootsOffset + rootsLength);
            GpuStateSetRecorder.recordGpu(depth, variableNames, heap, toLongRoots(roots));
        }
    }

    private static long[] fixedPrefixHeap() {
        if (semanticTemplate == null || cachedConstantHeap == null) {
            throw new IllegalStateException("generic GPU fixed prefix is not initialized");
        }
        final long[] prefix = Arrays.copyOf(semanticTemplate.heap,
                semanticTemplate.heap.length + cachedConstantHeap.length);
        System.arraycopy(cachedConstantHeap, 0, prefix, semanticTemplate.heap.length,
                cachedConstantHeap.length);
        return prefix;
    }

    private static int fixedPrefixWords() {
        if (cachedConstantHeap == null) {
            throw new IllegalStateException("generic GPU IR is not cached before loading a frontier segment");
        }
        return baseHeapWords + cachedConstantHeap.length;
    }

    private static long[] rebaseRoots(final long[] roots, final int base) {
        final long[] out = roots.clone();
        for (int i = 0; i < out.length; i++) out[i] += base;
        return out;
    }

    private static void cacheProgram(final GpuTlaSuccessorIR ir) {
        if (cachedIR == ir) {
            return;
        }
        cachedIR = ir;
        cachedExprProgram = rebaseConstants(ir.exprProgram());
        cachedConstantHeap = ValueEncoder.rebasedSemanticHeap(ir.constantHeap(), baseHeapWords);
    }

    private static long[] rebaseConstants(final long[] program) {
        final long[] out = new long[program.length];
        for (int i = 0; i < program.length; i += 2) {
            out[i] = program[i];
            out[i + 1] = program[i] == GpuTlaIR.OP_LOAD_CONST ? program[i + 1] + baseHeapWords : program[i + 1];
        }
        return out;
    }

    public static long[] stats() {
        return GPUKernel.genericFrontierStats();
    }

    public static long[] transferStats() {
        return GPUKernel.genericFrontierTransferStats();
    }

    public static String frontierMode() {
        return gpuSegmented ? "gpu-segmented" : (segmented ? "host-segmented" : "resident");
    }

    private static long[] toLongRoots(final int[] roots) {
        final long[] out = new long[roots.length];
        for (int i = 0; i < roots.length; i++) out[i] = roots[i];
        return out;
    }

    private static final class GpuHostBatch {
        private final long[] dynamicHeap;
        private final int[] absoluteRoots;
        private final int[] segmentDescriptors;

        private GpuHostBatch(final long[] dynamicHeap, final int[] absoluteRoots,
                final int[] segmentDescriptors) {
            this.dynamicHeap = dynamicHeap;
            this.absoluteRoots = absoluteRoots;
            this.segmentDescriptors = segmentDescriptors;
            validate();
        }

        private static GpuHostBatch initial(final ValueEncoder.EncodedSemanticState state) {
            return new GpuHostBatch(new long[0], state.roots.clone(),
                    new int[] { 0, 0, 0, state.roots.length });
        }

        private int stateCount() {
            final int vars = GPUFrontier.variableCount();
            long states = 0L;
            for (int descriptor = 0; descriptor < segmentDescriptors.length; descriptor += 4) {
                states += segmentDescriptors[descriptor + 3] / vars;
            }
            if (states > Integer.MAX_VALUE) {
                throw new IllegalStateException("CPU GPU-segmented batch exceeds Java state count range");
            }
            return (int) states;
        }

        private void validate() {
            final int vars = GPUFrontier.variableCount();
            if (vars <= 0 || segmentDescriptors.length == 0
                    || segmentDescriptors.length % 4 != 0) {
                throw new IllegalStateException("CPU GPU-segmented batch has invalid descriptors");
            }
            for (int descriptor = 0; descriptor < segmentDescriptors.length; descriptor += 4) {
                final int heapOffset = segmentDescriptors[descriptor];
                final int heapLength = segmentDescriptors[descriptor + 1];
                final int rootsOffset = segmentDescriptors[descriptor + 2];
                final int rootsLength = segmentDescriptors[descriptor + 3];
                if (heapOffset < 0 || heapLength < 0
                        || heapOffset > dynamicHeap.length - heapLength
                        || rootsOffset < 0 || rootsLength <= 0
                        || rootsOffset > absoluteRoots.length - rootsLength
                        || rootsLength % vars != 0
                        || rootsLength / vars > segmentMaxStates) {
                    throw new IllegalStateException("CPU GPU-segmented batch contains an invalid segment");
                }
            }
        }
    }

    private static final class HostSegment {
        private final long[] heap;
        private final long[] roots;

        private HostSegment(final long[] heap, final long[] roots) {
            this.heap = heap;
            this.roots = roots;
        }

        private static HostSegment fromEncoded(final ValueEncoder.EncodedSemanticState state) {
            return new HostSegment(state.heap.clone(), toLongRoots(state.roots));
        }

        private int stateCount() {
            final int vars = GPUFrontier.variableCount();
            if (vars <= 0 || roots.length % vars != 0) {
                throw new IllegalStateException("CPU frontier segment has invalid root count");
            }
            return roots.length / vars;
        }

        private static HostSegment fromPacket(final long[] packet, final int expectedStatus) {
            if (packet == null || packet.length < 5) {
                throw new IllegalStateException("generic GPU segmented frontier returned an invalid packet");
            }
            final int status = (int) packet[0];
            final int count = (int) packet[1];
            final int heapWords = (int) packet[2];
            final int vars = (int) packet[3];
            if (status != expectedStatus) {
                if (status == GpuTlaSuccessorIR.STATUS_HEAP_OVERFLOW) {
                    throw new IllegalStateException("generic GPU segmented heap capacity exceeded; increase -Dtlc.gpu.generic.heap.extra.words");
                }
                throw new IllegalStateException("generic GPU segmented frontier failed with status " + status);
            }
            if (count <= 0) return null;
            if (vars != GPUFrontier.variableCount() || heapWords < 0
                    || packet.length != 5L + heapWords + (long) count * vars) {
                throw new IllegalStateException("generic GPU segmented frontier packet shape is invalid");
            }
            final long[] heap = Arrays.copyOfRange(packet, 5, 5 + heapWords);
            final long[] roots = Arrays.copyOfRange(packet, 5 + heapWords, packet.length);
            return new HostSegment(heap, roots);
        }
    }
}
