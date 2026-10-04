package tlc2.tool.gpu;

import tlc2.tool.Action;
import tlc2.tool.ITool;
import tlc2.tool.StateVec;
import tlc2.tool.TLCState;

public final class GPUFrontier {
    private enum Backend {
        NONE,
        GENERIC,
    }

    private static int maxStates = Integer.getInteger("tlc.gpu.generic.max.states", 1000000);
    private static long[] lastStats = new long[] {0L, 0L, 0L, 0L, 0L};
    private static Backend backend = Backend.NONE;
    private static GpuTlaSuccessorIR genericIR = null;
    private static int variableCount = 0;

    private GPUFrontier() {
    }

    public static final class StateEncoder {
        private StateEncoder() {
        }
    }

    public static final class NextStateExpand {
        private NextStateExpand() {
        }
    }

    public static final class Fingerprint {
        private Fingerprint() {
        }
    }

    public static final class DuplicateRemoval {
        private DuplicateRemoval() {
        }
    }

    public static final class StateDecode {
        private StateDecode() {
        }
    }


    public static void init(final int maxStatesHint, final int varCount) {
        maxStates = maxStatesHint > 0 ? maxStatesHint : maxStates;
        variableCount = varCount;
        lastStats = new long[] {0L, 0L, 0L, 0L, 0L};
        backend = Backend.NONE;
        genericIR = null;
    }

    public static void loadInitialGeneric(final ITool tool, final StateVec states) {
        if (states == null || states.size() == 0) {
            throw new IllegalArgumentException("generic GPU frontier requires at least one initial state");
        }
        if (states.size() != 1) {
            throw new UnsupportedOperationException("generic GPU resident frontier currently supports exactly one initial state, got " + states.size());
        }
        final Action[] actions = tool.getActions();
        if (Boolean.getBoolean("tlc.gpu.generic.debug.actions")) {
            System.out.println("GPU generic actions: count=" + actions.length);
            for (int i = 0; i < actions.length; i++) {
                System.out.println("  [" + i + "] " + actions[i]);
            }
        }
        if (actions.length == 0) {
            throw new UnsupportedOperationException("generic GPU frontier requires at least one Next action");
        }
        final TLCState initial = states.first();
        final ValueEncoder.EncodedSemanticState encoded = ValueEncoder.encodeSemanticState(initial);
        genericIR = GpuTlaSuccessorCompiler.compileActions(tool, initial, actions);
        GpuTlaGenericFrontier.init(maxStates, encoded.roots.length, encoded.heap.length + genericIR.constantHeap().length);
        GpuTlaGenericFrontier.loadInitial(encoded);
        lastStats = GpuTlaGenericFrontier.stats();
        backend = Backend.GENERIC;
    }

    public static int step() {
        final int frontierSize;
        if (backend == Backend.GENERIC) {
            if (genericIR == null) throw new IllegalStateException("generic GPU frontier is not initialized");
            final long previousDepth = lastStats.length >= 4 ? lastStats[3] : -1L;
            frontierSize = GpuTlaGenericFrontier.step(genericIR);
            lastStats = GpuTlaGenericFrontier.stats();
            checkGenericStatus();
            if (lastStats.length >= 4 && lastStats[3] <= previousDepth) {
                throw new RuntimeException("generic GPU frontier step did not commit a BFS layer; "
                        + "the CUDA kernel may have failed (check thread stack size and available GPU memory)");
            }
            if (lastStats.length >= 3 && lastStats[2] != frontierSize) {
                throw new RuntimeException("generic GPU frontier step returned " + frontierSize
                        + " states but committed " + lastStats[2]);
            }

            return frontierSize;
        }
        throw new IllegalStateException("GPU frontier is not initialized");
    }

    public static int drain() {
        if (backend == Backend.GENERIC) {
            if (genericIR == null) throw new IllegalStateException("generic GPU frontier is not initialized");
            final int frontierSize = GpuTlaGenericFrontier.drain(genericIR);
            lastStats = GpuTlaGenericFrontier.stats();
            checkGenericStatus();
            return frontierSize;
        }
        throw new IllegalStateException("GPU frontier is not initialized");
    }

    private static void checkGenericStatus() {
        if (lastStats.length >= 5 && lastStats[4] != 0L) {
            throw new RuntimeException("generic GPU frontier capacity exceeded; increase -Dtlc.gpu.generic.max.states");
        }
        if (lastStats.length >= 6 && lastStats[5] != 0L) {
            if (lastStats[5] == GpuTlaSuccessorIR.STATUS_HEAP_OVERFLOW) {
                throw new RuntimeException("generic GPU value heap capacity exceeded; increase -Dtlc.gpu.generic.heap.extra.words");
            }
            if (lastStats[5] == GpuTlaSuccessorIR.STATUS_SHARE_OVERFLOW) {
                throw new RuntimeException("generic GPU source-sharing table capacity exceeded; increase -Dtlc.gpu.generic.gpu.segment.share.capacity");
            }
            throw new RuntimeException("generic GPU frontier status error: " + lastStats[5]);
        }
    }

    public static String backendName() {
        return backend.name();
    }

    public static long[] stats() {
        return lastStats.clone();
    }

    public static long[] transferStats() {
        return backend == Backend.GENERIC ? GpuTlaGenericFrontier.transferStats() : new long[12];
    }

    public static String frontierMode() {
        return backend == Backend.GENERIC ? GpuTlaGenericFrontier.frontierMode() : "none";
    }

    static int variableCount() {
        return variableCount;
    }
}
