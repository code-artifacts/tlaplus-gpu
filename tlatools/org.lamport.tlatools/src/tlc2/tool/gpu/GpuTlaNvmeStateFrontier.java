package tlc2.tool.gpu;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

final class GpuTlaNvmeStateFrontier {
    private static final int MAGIC = 0x47544631;
    private static final int VERSION = 1;

    private GpuTlaNvmeStateFrontier() {
    }

    static Writer writer(final Path path, final int varCount) throws IOException {
        return new Writer(path, varCount);
    }

    static Reader reader(final Path path, final ValueEncoder.EncodedSemanticState template) throws IOException {
        return new Reader(path, template);
    }

    static final class Writer implements AutoCloseable {
        private final Path path;
        private final Path partial;
        private final int varCount;
        private final DataOutputStream out;
        private long count;
        private long bytes;
        private boolean closed;

        private Writer(final Path path, final int varCount) throws IOException {
            this.path = path;
            this.partial = path.resolveSibling(path.getFileName().toString() + ".part");
            this.varCount = varCount;
            Files.createDirectories(path.getParent());
            out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(partial), 1 << 20));
            out.writeInt(MAGIC);
            out.writeInt(VERSION);
            out.writeInt(varCount);
            bytes = 3L * Integer.BYTES;
        }

        void write(final ValueEncoder.EncodedSemanticState state) throws IOException {
            if (state.roots.length != varCount) {
                throw new IllegalArgumentException("frontier state variable count changed");
            }
            out.writeInt(state.heap.length);
            for (int root : state.roots) out.writeInt(root);
            for (long word : state.heap) out.writeLong(word);
            count++;
            bytes += Integer.BYTES + (long) varCount * Integer.BYTES + (long) state.heap.length * Long.BYTES;
        }

        long count() {
            return count;
        }

        long bytes() {
            return bytes;
        }

        @Override
        public void close() throws IOException {
            if (closed) return;
            closed = true;
            out.close();
            try {
                Files.move(partial, path, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(partial, path, StandardCopyOption.REPLACE_EXISTING);
            }
        }
    }

    static final class Reader implements AutoCloseable {
        private final DataInputStream in;
        private final ValueEncoder.EncodedSemanticState template;
        private final int varCount;

        private Reader(final Path path, final ValueEncoder.EncodedSemanticState template) throws IOException {
            this.template = template;
            in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path), 1 << 20));
            if (in.readInt() != MAGIC || in.readInt() != VERSION) {
                throw new IOException("invalid GPU NVMe frontier file: " + path);
            }
            varCount = in.readInt();
            if (varCount != template.roots.length) {
                throw new IOException("GPU NVMe frontier variable count does not match the compiled IR");
            }
        }

        ValueEncoder.EncodedSemanticState next() throws IOException {
            final int heapWords;
            try {
                heapWords = in.readInt();
            } catch (EOFException e) {
                return null;
            }
            if (heapWords <= 0) throw new IOException("invalid semantic heap size in NVMe frontier");
            final int[] roots = new int[varCount];
            final long[] heap = new long[heapWords];
            for (int i = 0; i < varCount; i++) roots[i] = in.readInt();
            for (int i = 0; i < heapWords; i++) heap[i] = in.readLong();
            return new ValueEncoder.EncodedSemanticState(template.variableNames, template.variableNameIds,
                    roots, heap);
        }

        @Override
        public void close() throws IOException {
            in.close();
        }
    }
}
