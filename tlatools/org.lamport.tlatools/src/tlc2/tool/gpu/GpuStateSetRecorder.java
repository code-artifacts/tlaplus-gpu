package tlc2.tool.gpu;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;

import tlc2.TLCGlobals;
import tlc2.tool.TLCState;

/**
 * Optional validation output for comparing complete BFS layers.
 * Disabled unless tlc.validation.dump.dir is set.
 */
public final class GpuStateSetRecorder {
    private static final Object LOCK = new Object();
    private static final Map<Integer, BufferedWriter> WRITERS = new HashMap<Integer, BufferedWriter>();
    private static final Map<Integer, Long> COUNTS = new HashMap<Integer, Long>();
    private static boolean manifestWritten;
    private static boolean closed;
    private static String[] variableNames;

    private GpuStateSetRecorder() {
    }

    public static boolean enabled() {
        final String dir = System.getProperty("tlc.validation.dump.dir");
        return dir != null && !dir.trim().isEmpty();
    }

    public static void recordCpu(final TLCState state) {
        if (state == null || !enabled() || TLCGlobals.useGPU) return;
        final ValueEncoder.EncodedSemanticState encoded = ValueEncoder.encodeSemanticState(state);
        recordEncoded(state.getLevel() - TLCState.INIT_LEVEL, encoded.variableNames, encoded.heap,
                toLongRoots(encoded.roots));
    }

    public static void recordCpuInitial(final TLCState state) {
        if (state == null || !enabled() || TLCGlobals.useGPU) return;
        final ValueEncoder.EncodedSemanticState encoded = ValueEncoder.encodeSemanticState(state);
        recordEncoded(0, encoded.variableNames, encoded.heap, toLongRoots(encoded.roots));
    }

    public static void recordGpu(final int depth, final String[] names, final long[] heap,
            final long[] roots) {
        if (!enabled()) return;
        if (roots == null || roots.length == 0) return;
        final int vars = names == null ? 0 : names.length;
        if (vars <= 0 || roots.length % vars != 0) {
            throw new IllegalArgumentException("invalid GPU validation roots");
        }
        for (int offset = 0; offset < roots.length; offset += vars) {
            final long[] stateRoots = new long[vars];
            System.arraycopy(roots, offset, stateRoots, 0, vars);
            recordEncoded(depth, names, heap, stateRoots);
        }
    }

    private static void recordEncoded(final int depth, final String[] names, final long[] heap,
            final long[] roots) {
        if (depth < 0) throw new IllegalArgumentException("negative validation layer");
        final String canonical = ValueEncoder.canonicalSemanticState(names, heap, roots);
        synchronized (LOCK) {
            if (closed) throw new IllegalStateException("validation recorder is already closed");
            try {
                ensureManifest(names);
                BufferedWriter writer = WRITERS.get(Integer.valueOf(depth));
                if (writer == null) {
                    final Path path = layerPath(depth);
                    writer = Files.newBufferedWriter(path, StandardCharsets.UTF_8,
                            StandardOpenOption.CREATE, StandardOpenOption.APPEND);
                    WRITERS.put(Integer.valueOf(depth), writer);
                }
                writer.write(canonical);
                writer.newLine();
                final Integer key = Integer.valueOf(depth);
                final Long old = COUNTS.get(key);
                COUNTS.put(key, Long.valueOf(old == null ? 1L : old.longValue() + 1L));
            } catch (IOException e) {
                throw new IllegalStateException("unable to write validation layer " + depth, e);
            }
        }
    }

    private static void ensureManifest(final String[] names) throws IOException {
        if (manifestWritten) return;
        variableNames = names == null ? new String[0] : names.clone();
        final Path dir = validationDir();
        Files.createDirectories(dir);
        final Path manifest = dir.resolve("manifest.txt");
        try (BufferedWriter writer = Files.newBufferedWriter(manifest, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)) {
            writer.write("format=gpu-tla-layer-v1");
            writer.newLine();
            writer.write("mode=" + System.getProperty("tlc.validation.mode", "unknown"));
            writer.newLine();
            writer.write("variables=");
            for (int i = 0; i < variableNames.length; i++) {
                if (i > 0) writer.write(",");
                writer.write(variableNames[i]);
            }
            writer.newLine();
        }
        manifestWritten = true;
    }

    private static Path validationDir() {
        return Paths.get(System.getProperty("tlc.validation.dump.dir")).toAbsolutePath();
    }

    private static Path layerPath(final int depth) {
        return validationDir().resolve(String.format("layer-%06d.states", Integer.valueOf(depth)));
    }

    public static void close() {
        synchronized (LOCK) {
            if (closed) return;
            closed = true;
            for (BufferedWriter writer : WRITERS.values()) {
                try {
                    writer.close();
                } catch (IOException e) {
                    throw new IllegalStateException("unable to close validation output", e);
                }
            }
            WRITERS.clear();
        }
    }

    private static long[] toLongRoots(final int[] roots) {
        final long[] result = new long[roots.length];
        for (int i = 0; i < roots.length; i++) result[i] = roots[i];
        return result;
    }
}
