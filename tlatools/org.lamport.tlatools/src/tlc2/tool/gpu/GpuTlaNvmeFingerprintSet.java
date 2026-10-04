package tlc2.tool.gpu;

import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;

final class GpuTlaNvmeFingerprintSet implements AutoCloseable {
    private final Path logPath;
    private final int shardCount;
    private final int shardMask;
    private final int slotsPerShard;
    private final long[][] tables;
    private final FileOutputStream logFile;
    private final DataOutputStream log;
    private long size;

    GpuTlaNvmeFingerprintSet(final Path directory, final long requestedCapacity, final int requestedShards)
            throws IOException {
        shardCount = nextPowerOfTwo(Math.max(1, requestedShards));
        shardMask = shardCount - 1;
        final long perShard = Math.max(2L, (requestedCapacity + shardCount - 1L) / shardCount);
        slotsPerShard = nextPowerOfTwo(perShard > Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) perShard);
        if (slotsPerShard <= 0) {
            throw new IllegalArgumentException("in-memory fingerprint shard capacity is invalid");
        }
        tables = new long[shardCount][];
        Files.createDirectories(directory);
        try (DirectoryStream<Path> oldTables = Files.newDirectoryStream(directory, "fp-*.bin")) {
            for (Path oldTable : oldTables) Files.deleteIfExists(oldTable);
        }
        logPath = directory.resolve("fingerprints.bin");
        logFile = new FileOutputStream(logPath.toFile(), false);
        log = new DataOutputStream(new BufferedOutputStream(logFile, 8 << 20));
    }

    boolean putIfAbsent(final long fingerprint) throws IOException {
        final long key = encodeKey(fingerprint);
        final long mixed = mix64(key);
        final int shard = (int) (mixed >>> 32) & shardMask;
        long[] table = tables[shard];
        if (table == null) {
            table = new long[slotsPerShard];
            tables[shard] = table;
        }
        int slot = (int) mixed & (slotsPerShard - 1);
        for (int probe = 0; probe < slotsPerShard; probe++) {
            final long current = table[slot];
            if (current == key) return false;
            if (current == 0L) {
                table[slot] = key;
                log.writeLong(fingerprint);
                size++;
                return true;
            }
            slot = (slot + 1) & (slotsPerShard - 1);
        }
        throw new IllegalStateException("fingerprint shard is full; increase "
                + "-Dtlc.gpu.generic.nvme.hash.capacity");
    }

    long size() {
        return size;
    }

    long capacity() {
        return (long) shardCount * slotsPerShard;
    }

    long allocatedBytes() throws IOException {
        log.flush();
        return Files.size(logPath);
    }

    void force() {
        try {
            log.flush();
            if (Boolean.getBoolean("tlc.gpu.generic.nvme.sync.each.level")) {
                logFile.getFD().sync();
            }
        } catch (IOException e) {
            throw new RuntimeException("failed to flush GPU NVMe fingerprint log", e);
        }
    }

    @Override
    public void close() throws IOException {
        log.flush();
        logFile.getFD().sync();
        log.close();
    }

    private static long encodeKey(final long fingerprint) {
        final long key = fingerprint ^ 0x9e3779b97f4a7c15L;
        return key == 0L ? 1L : key;
    }

    private static long mix64(long value) {
        value ^= value >>> 30;
        value *= 0xbf58476d1ce4e5b9L;
        value ^= value >>> 27;
        value *= 0x94d049bb133111ebL;
        value ^= value >>> 31;
        return value;
    }

    private static int nextPowerOfTwo(final int value) {
        if (value <= 1) return 1;
        final int highest = Integer.highestOneBit(value - 1);
        if (highest > (1 << 29)) return 1 << 30;
        return highest << 1;
    }
}
