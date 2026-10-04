package tlc2.tool.gpu;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import tlc2.tool.Action;
import tlc2.tool.ITool;
import tlc2.tool.StateVec;
import tlc2.tool.TLCState;

public final class GpuTlaOutOfCoreFrontier implements AutoCloseable {
    private final GpuTlaSuccessorIR ir;
    private final ValueEncoder.EncodedSemanticState template;
    private final Path directory;
    private final GpuTlaNvmeFingerprintSet visited;
    private final int maxSuccessors;
    private final int successorBatchSize;

    private Path currentFrontier;
    private long generated;
    private long distinct;
    private long frontierSize;
    private long stateBytes;
    private int depth;
    private int status;
    private boolean overflow;

    private GpuTlaOutOfCoreFrontier(final ITool tool, final StateVec initialStates) throws IOException {
        if (initialStates == null || initialStates.size() == 0) {
            throw new IllegalArgumentException("GPU NVMe frontier requires at least one initial state");
        }

        final Action[] actions = tool.getActions();
        if (actions == null || actions.length == 0) {
            throw new UnsupportedOperationException("GPU NVMe frontier requires at least one Next action");
        }
        final TLCState initial = initialStates.first();
        template = ValueEncoder.encodeSemanticState(initial);
        ir = GpuTlaSuccessorCompiler.compileActions(tool, initial, actions);
        directory = Paths.get(System.getProperty("tlc.gpu.generic.nvme.dir",
                System.getProperty("java.io.tmpdir") + "/tlc-gpu-generic-nvme"));
        final long hashCapacity = Long.getLong("tlc.gpu.generic.nvme.hash.capacity", 268435456L);
        final int hashShards = Integer.getInteger("tlc.gpu.generic.nvme.hash.shards", 256);
        maxSuccessors = Integer.getInteger("tlc.gpu.generic.nvme.max.successors", 4096);
        successorBatchSize = Integer.getInteger("tlc.gpu.generic.nvme.batch.successors", 8192);
        if (maxSuccessors <= 0 || successorBatchSize <= 0) {
            throw new IllegalArgumentException("GPU NVMe successor capacities must be positive");
        }

        visited = new GpuTlaNvmeFingerprintSet(directory.resolve("visited"), hashCapacity, hashShards);
        currentFrontier = frontierPath(0);
        try (GpuTlaNvmeStateFrontier.Writer writer =
                GpuTlaNvmeStateFrontier.writer(currentFrontier, template.roots.length)) {
            for (int i = 0; i < initialStates.size(); i++) {
                final ValueEncoder.EncodedSemanticState encoded =
                        ValueEncoder.encodeSemanticState(initialStates.elementAt(i));
                final long fingerprint = GpuTlaBatchDeduplicator.fingerprint(encoded);
                if (!visited.putIfAbsent(fingerprint)) continue;
                writer.write(encoded);
                distinct++;
            }
            frontierSize = distinct;
            stateBytes += writer.bytes();
        }
    }

    public static GpuTlaOutOfCoreFrontier create(final ITool tool, final StateVec initialStates)
            throws IOException {
        return new GpuTlaOutOfCoreFrontier(tool, initialStates);
    }

    public int step() {
        if (frontierSize <= 0L) return 0;
        if (frontierSize > Integer.MAX_VALUE) {
            throw new IllegalStateException("a single BFS layer exceeds the current Java progress counter range");
        }

        final Path nextFrontier = frontierPath(depth + 1);
        final ArrayList<Candidate> candidates = new ArrayList<Candidate>(successorBatchSize);
        long nextCount = 0L;
        try (GpuTlaNvmeStateFrontier.Reader reader =
                    GpuTlaNvmeStateFrontier.reader(currentFrontier, template);
                GpuTlaNvmeStateFrontier.Writer writer =
                    GpuTlaNvmeStateFrontier.writer(nextFrontier, template.roots.length)) {
            ValueEncoder.EncodedSemanticState current;
            while ((current = reader.next()) != null) {
                final GpuTlaSuccessorEnumerator.Result result =
                        GpuTlaSuccessorEnumerator.expand(ir, current, maxSuccessors);
                if (Boolean.getBoolean("tlc.gpu.generic.nvme.debug") && depth == 0) {
                    System.out.println("GPU NVMe debug: heapWords=" + current.heap.length
                            + ", roots=" + java.util.Arrays.toString(current.roots)
                            + ", branches=" + java.util.Arrays.toString(ir.branchTable())
                            + ", ops=" + java.util.Arrays.toString(ir.opTable())
                            + ", status=" + result.status + ", successors=" + result.fingerprints.length
                            + ", returnedHeapWords=" + result.heap.length);
                }
                generated += result.fingerprints.length;
                if (result.overflow) {
                    overflow = true;
                    status = result.status;
                    throw new IllegalStateException("generic GPU NVMe successor capacity exceeded; increase "
                            + "-Dtlc.gpu.generic.nvme.max.successors");
                }
                if (!result.ok()) {
                    status = result.status;
                    throw new IllegalStateException("generic GPU NVMe successor evaluation failed with status "
                            + result.status);
                }
                if (result.fingerprints.length > 0 && result.heap.length == 0) {
                    throw new IllegalStateException("generic GPU NVMe successor packet did not include a value heap");
                }
                for (int i = 0; i < result.fingerprints.length; i++) {
                    candidates.add(new Candidate(result.fingerprints[i], result.heap, result.roots[i]));
                    if (candidates.size() >= successorBatchSize) {
                        nextCount += flush(candidates, writer);
                    }
                }
            }
            nextCount += flush(candidates, writer);
            stateBytes += writer.bytes();
        } catch (IOException e) {
            throw new RuntimeException("GPU NVMe frontier I/O failed in " + directory, e);
        }

        depth++;
        currentFrontier = nextFrontier;
        frontierSize = nextCount;
        visited.force();
        return nextCount > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) nextCount;
    }

    public long[] stats() {
        return new long[] {
                generated,
                distinct,
                frontierSize,
                depth,
                overflow ? 1L : 0L,
                status,
                stateBytes,
                visited.capacity()
        };
    }

    public Path directory() {
        return directory;
    }

    @Override
    public void close() throws IOException {
        visited.close();
    }

    private long flush(final ArrayList<Candidate> candidates,
            final GpuTlaNvmeStateFrontier.Writer writer) throws IOException {
        if (candidates.isEmpty()) return 0L;
        final long[] fingerprints = new long[candidates.size()];
        for (int i = 0; i < fingerprints.length; i++) fingerprints[i] = candidates.get(i).fingerprint;
        final boolean[] localKeep = GpuTlaBatchDeduplicator.firstOccurrences(fingerprints);
        long accepted = 0L;
        for (int i = 0; i < candidates.size(); i++) {
            if (!localKeep[i]) continue;
            final Candidate candidate = candidates.get(i);
            if (!visited.putIfAbsent(candidate.fingerprint)) continue;
            final ValueEncoder.EncodedSemanticState compact =
                    ValueEncoder.compactSemanticState(template, candidate.heap, candidate.roots);
            writer.write(compact);
            distinct++;
            accepted++;
        }
        candidates.clear();
        return accepted;
    }

    private Path frontierPath(final int level) {
        return directory.resolve(String.format("frontier-%06d.bin", level));
    }

    private static final class Candidate {
        private final long fingerprint;
        private final long[] heap;
        private final long[] roots;

        private Candidate(final long fingerprint, final long[] heap, final long[] roots) {
            this.fingerprint = fingerprint;
            this.heap = heap;
            this.roots = roots;
        }
    }
}
