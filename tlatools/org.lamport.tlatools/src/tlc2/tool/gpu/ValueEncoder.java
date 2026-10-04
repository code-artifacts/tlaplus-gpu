package tlc2.tool.gpu;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;

import tla2sany.semantic.OpDeclNode;
import tlc2.tool.TLCState;
import tlc2.value.IValue;
import tlc2.value.impl.BoolValue;
import tlc2.value.impl.FcnRcdValue;
import tlc2.value.impl.IntValue;
import tlc2.value.impl.IntervalValue;
import tlc2.value.impl.ModelValue;
import tlc2.value.impl.RecordValue;
import tlc2.value.impl.SetEnumValue;
import tlc2.value.impl.StringValue;
import tlc2.value.impl.TupleValue;
import tlc2.value.impl.Value;
import tlc2.value.impl.ValueVec;
import util.UniqueString;

public final class ValueEncoder {
    public static final int TAG_NULL = 0;
    public static final int TAG_BOOL = 1;
    public static final int TAG_INT = 2;
    public static final int TAG_MODEL = 3;
    public static final int TAG_TUPLE = 4;
    public static final int TAG_SET_ENUM = 5;
    public static final int TAG_FCN_RCD = 6;
    public static final int TAG_RECORD = 7;
    public static final int TAG_INTERVAL = 8;
    public static final int TAG_STRING = 9;
    public static final int TAG_OPAQUE = 255;

    public static final int FCN_DOMAIN_EXPLICIT = 0;
    public static final int FCN_DOMAIN_INTERVAL = 1;

    private ValueEncoder() {
    }

    /**
     * Legacy scalar encoder used by early GPU experiments. Keep this stable so the
     * current generic resident path is not affected.
     */
    public static long encode(final IValue v) {
        if (v == null) {
            return 0L;
        }
        return v.hashCode();
    }

    public static EncodedSemanticValue encodeSemanticValue(final IValue value) {
        final SemanticEncoder encoder = new SemanticEncoder();
        final int root = encoder.encode(value);
        return new EncodedSemanticValue(encoder.toHeap(), root);
    }

    static EncodedSemanticValue encodeSyntheticOpaque(final String namespace, final long identity) {
        final long namespaceId = stableStringId(namespace);
        final long valueId = stableStringId(namespace + ":" + Long.toUnsignedString(identity));
        return new EncodedSemanticValue(new long[] {
                header(TAG_OPAQUE, 0, namespaceId), valueId
        }, 0);
    }

    public static EncodedSemanticState encodeSemanticState(final TLCState state) {
        final SemanticEncoder encoder = new SemanticEncoder();
        final OpDeclNode[] vars = state.getVars();
        final String[] names = new String[vars.length];
        final long[] nameIds = new long[vars.length];
        final int[] roots = new int[vars.length];
        for (int i = 0; i < vars.length; i++) {
            final UniqueString name = vars[i].getName();
            names[i] = name.toString();
            nameIds[i] = stableStringId(names[i]);
            roots[i] = encoder.encode(state.lookup(name));
        }
        return new EncodedSemanticState(names, nameIds, roots, encoder.toHeap());
    }

    /**
     * Returns a heap-offset-independent representation of one complete state.
     * Container order is normalized only where TLA+ semantics is order-independent.
     */
    public static String canonicalSemanticState(final String[] variableNames, final long[] heap,
            final long[] roots) {
        if (variableNames == null || heap == null || roots == null || variableNames.length != roots.length) {
            throw new IllegalArgumentException("invalid semantic state shape");
        }
        final List<String> variables = new ArrayList<String>(roots.length);
        for (int i = 0; i < roots.length; i++) {
            final String name = variableNames[i] == null ? "" : variableNames[i];
            variables.add(escapeCanonical(name) + "="
                    + canonicalSemanticValue(heap, roots[i], new HashSet<Integer>()));
        }
        Collections.sort(variables);
        return joinCanonical("state", variables);
    }

    private static String canonicalSemanticValue(final long[] heap, final long offset,
            final HashSet<Integer> active) {
        final int start = checkedOffset(heap, offset);
        if (!active.add(Integer.valueOf(start))) {
            throw new IllegalArgumentException("cyclic semantic heap object at offset " + start);
        }
        try {
            final long header = heap[start];
            final int tag = tagOf(header);
            final int arity = arityOf(header);
            final int payload = checkedPayloadWords(tag, arity, auxOf(header));
            final int end = checkedObjectEnd(heap, start, payload);
            switch (tag) {
            case TAG_NULL:
                return "null";
            case TAG_BOOL:
                return "bool:" + auxOf(header);
            case TAG_INT:
                return "int:" + signed32(auxOf(header));
            case TAG_MODEL:
                return "model:" + auxOf(header) + ":" + heap[start + 1] + ":" + heap[start + 2];
            case TAG_STRING:
                return "string:" + heap[start + 1];
            case TAG_INTERVAL:
                return "interval:" + heap[start + 1] + ":" + heap[start + 2];
            case TAG_TUPLE: {
                final List<String> elements = new ArrayList<String>(arity);
                for (int i = 0; i < arity; i++) {
                    elements.add(canonicalSemanticValue(heap, heap[start + 1 + i], active));
                }
                return joinCanonical("tuple", elements);
            }
            case TAG_SET_ENUM: {
                final HashSet<String> unique = new HashSet<String>();
                for (int i = 0; i < arity; i++) {
                    unique.add(canonicalSemanticValue(heap, heap[start + 1 + i], active));
                }
                final List<String> elements = new ArrayList<String>(unique);
                Collections.sort(elements);
                return joinCanonical("set", elements);
            }
            case TAG_FCN_RCD: {
                final List<String> pairs = new ArrayList<String>(arity);
                if (auxOf(header) == FCN_DOMAIN_INTERVAL) {
                    final long low = heap[start + 1];
                    final long high = heap[start + 2];
                    for (int i = 0; i < arity; i++) {
                        pairs.add(canonicalSemanticValue(heap, heap[start + 3 + i], active));
                    }
                    return "fcn-interval:" + low + ":" + high + ":" + joinCanonical("values", pairs);
                }
                if (auxOf(header) != FCN_DOMAIN_EXPLICIT) {
                    throw new IllegalArgumentException("unknown function domain kind at offset " + start);
                }
                for (int i = 0; i < arity; i++) {
                    final String domain = canonicalSemanticValue(heap, heap[start + 1 + i], active);
                    final String value = canonicalSemanticValue(heap, heap[start + 1 + arity + i], active);
                    pairs.add(joinCanonical("pair", asList(domain, value)));
                }
                Collections.sort(pairs);
                return joinCanonical("fcn", pairs);
            }
            case TAG_RECORD: {
                final List<String> fields = new ArrayList<String>(arity);
                for (int i = 0; i < arity; i++) {
                    final String field = Long.toUnsignedString(heap[start + 1 + i]);
                    final String value = canonicalSemanticValue(heap, heap[start + 1 + arity + i], active);
                    fields.add(joinCanonical("field", asList(field, value)));
                }
                Collections.sort(fields);
                return joinCanonical("record", fields);
            }
            case TAG_OPAQUE:
                return "opaque:" + auxOf(header) + ":" + heap[start + 1];
            default:
                throw new IllegalArgumentException("unknown semantic heap tag " + tag + " at offset " + start);
            }
        } finally {
            active.remove(Integer.valueOf(start));
        }
    }

    private static int checkedOffset(final long[] heap, final long offset) {
        if (offset < 0L || offset > Integer.MAX_VALUE || offset >= heap.length) {
            throw new IllegalArgumentException("semantic heap reference is out of bounds: " + offset);
        }
        return (int) offset;
    }

    private static int checkedObjectEnd(final long[] heap, final int start, final int payload) {
        final long end = (long) start + 1L + payload;
        if (end > heap.length) {
            throw new IllegalArgumentException("truncated semantic heap object at offset " + start);
        }
        return (int) end;
    }

    private static int checkedPayloadWords(final int tag, final int arity, final long aux) {
        if (arity < 0) {
            throw new IllegalArgumentException("negative semantic heap arity");
        }
        switch (tag) {
        case TAG_NULL:
        case TAG_BOOL:
        case TAG_INT:
            return 0;
        case TAG_MODEL:
        case TAG_INTERVAL:
            return 2;
        case TAG_STRING:
        case TAG_OPAQUE:
            return 1;
        case TAG_TUPLE:
        case TAG_SET_ENUM:
            return arity;
        case TAG_FCN_RCD:
            if (aux != FCN_DOMAIN_EXPLICIT && aux != FCN_DOMAIN_INTERVAL) {
                throw new IllegalArgumentException("unknown function domain kind");
            }
            return aux == FCN_DOMAIN_INTERVAL ? 2 + arity : 2 * arity;
        case TAG_RECORD:
            return 2 * arity;
        default:
            throw new IllegalArgumentException("unknown semantic heap tag " + tag);
        }
    }

    private static long signed32(final long value) {
        return value >= 0x80000000L ? value - 0x100000000L : value;
    }

    private static List<String> asList(final String first, final String second) {
        final List<String> values = new ArrayList<String>(2);
        values.add(first);
        values.add(second);
        return values;
    }

    private static String joinCanonical(final String kind, final List<String> parts) {
        final StringBuilder result = new StringBuilder(kind).append('[').append(parts.size()).append(']');
        for (String part : parts) {
            result.append(part.length()).append(':').append(part);
        }
        return result.toString();
    }

    private static String escapeCanonical(final String value) {
        final StringBuilder result = new StringBuilder(value.length());
        for (int i = 0; i < value.length(); i++) {
            final char c = value.charAt(i);
            if (Character.isLetterOrDigit(c) || c == '_' || c == '.' || c == '-') {
                result.append(c);
            } else {
                result.append('\\').append(Integer.toHexString(c)).append(';');
            }
        }
        return result.toString();
    }

    private static long header(final int tag, final int arity, final long aux) {
        return ((long) (tag & 0xff) << 56)
                | ((long) (arity & 0x00ffffff) << 32)
                | (aux & 0xffffffffL);
    }

    public static int tagOf(final long header) {
        return (int) ((header >>> 56) & 0xffL);
    }

    public static int arityOf(final long header) {
        return (int) ((header >>> 32) & 0x00ffffffL);
    }

    public static long auxOf(final long header) {
        return header & 0xffffffffL;
    }

    public static long[] rebasedSemanticHeap(final long[] source, final int offsetBase) {
        final long[] out = new long[source.length];
        System.arraycopy(source, 0, out, 0, source.length);
        rebaseSemanticHeapRange(out, 0, source.length, offsetBase);
        return out;
    }

    public static void copySemanticHeap(final long[] source, final long[] target, final int targetBase) {
        System.arraycopy(source, 0, target, targetBase, source.length);
        rebaseSemanticHeapRange(target, targetBase, source.length, targetBase);
    }

    private static void rebaseSemanticHeapRange(final long[] heap, final int objectBase, final int words,
            final int offsetBase) {
        int off = 0;
        while (off < words) {
            final int start = objectBase + off;
            final long h = heap[start];
            final int tag = tagOf(h);
            final int arity = arityOf(h);
            final int payloadWords = payloadWords(tag, arity, auxOf(h));
            rebasePayloadRefs(heap, start, tag, arity, offsetBase);
            off += 1 + payloadWords;
        }
    }

    private static int payloadWords(final int tag, final int arity, final long aux) {
        switch (tag) {
        case TAG_NULL:
        case TAG_BOOL:
        case TAG_INT:
            return 0;
        case TAG_MODEL:
        case TAG_INTERVAL:
            return 2;
        case TAG_STRING:
            return 1;
        case TAG_TUPLE:
        case TAG_SET_ENUM:
            return arity;
        case TAG_FCN_RCD:
            return aux == FCN_DOMAIN_INTERVAL ? 2 + arity : 2 * arity;
        case TAG_RECORD:
            return 2 * arity;
        case TAG_OPAQUE:
            return 1;
        default:
            return 0;
        }
    }

    private static void rebasePayloadRefs(final long[] heap, final int start, final int tag, final int arity,
            final int offsetBase) {
        if (offsetBase == 0) return;
        switch (tag) {
        case TAG_TUPLE:
        case TAG_SET_ENUM:
            for (int i = 0; i < arity; i++) heap[start + 1 + i] += offsetBase;
            return;
        case TAG_FCN_RCD:
            if (auxOf(heap[start]) == FCN_DOMAIN_INTERVAL) {
                for (int i = 0; i < arity; i++) heap[start + 3 + i] += offsetBase;
            } else {
                for (int i = 0; i < 2 * arity; i++) heap[start + 1 + i] += offsetBase;
            }
            return;
        case TAG_RECORD:
            for (int i = 0; i < arity; i++) heap[start + 1 + arity + i] += offsetBase;
            return;
        default:
            return;
        }
    }

    public static EncodedSemanticState compactSemanticState(final EncodedSemanticState template,
            final long[] sourceHeap, final long[] sourceRoots) {
        if (template == null || sourceHeap == null || sourceRoots == null
                || sourceRoots.length != template.roots.length) {
            throw new IllegalArgumentException("invalid semantic state compaction input");
        }
        final SemanticHeapCompactor compactor = new SemanticHeapCompactor(sourceHeap);
        final int[] compactRoots = new int[sourceRoots.length];
        for (int i = 0; i < sourceRoots.length; i++) {
            if (sourceRoots[i] < 0L || sourceRoots[i] > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("semantic root offset is outside the 32-bit heap range");
            }
            compactRoots[i] = compactor.copy((int) sourceRoots[i]);
        }
        return new EncodedSemanticState(template.variableNames.clone(), template.variableNameIds.clone(),
                compactRoots, compactor.toHeap());
    }

    private static final class SemanticHeapCompactor {
        private final long[] source;
        private final ArrayList<Long> target = new ArrayList<Long>();
        private final HashMap<Integer, Integer> offsets = new HashMap<Integer, Integer>();

        private SemanticHeapCompactor(final long[] source) {
            this.source = source;
        }

        private int copy(final int sourceOffset) {
            if (sourceOffset < 0 || sourceOffset >= source.length) {
                throw new IllegalArgumentException("semantic heap reference is out of bounds: " + sourceOffset);
            }
            final Integer existing = offsets.get(Integer.valueOf(sourceOffset));
            if (existing != null) return existing.intValue();

            final long h = source[sourceOffset];
            final int tag = tagOf(h);
            final int arity = arityOf(h);
            final int words = payloadWords(tag, arity, auxOf(h));
            if (words < 0 || sourceOffset + words >= source.length) {
                throw new IllegalArgumentException("truncated semantic heap object at offset " + sourceOffset);
            }
            final int targetOffset = target.size();
            offsets.put(Integer.valueOf(sourceOffset), Integer.valueOf(targetOffset));
            target.add(Long.valueOf(h));
            for (int i = 0; i < words; i++) target.add(Long.valueOf(source[sourceOffset + 1 + i]));

            switch (tag) {
            case TAG_TUPLE:
            case TAG_SET_ENUM:
                for (int i = 0; i < arity; i++) remap(targetOffset + 1 + i);
                break;
            case TAG_FCN_RCD:
                if (auxOf(h) == FCN_DOMAIN_INTERVAL) {
                    for (int i = 0; i < arity; i++) remap(targetOffset + 3 + i);
                } else {
                    for (int i = 0; i < 2 * arity; i++) remap(targetOffset + 1 + i);
                }
                break;
            case TAG_RECORD:
                for (int i = 0; i < arity; i++) remap(targetOffset + 1 + arity + i);
                break;
            default:
                break;
            }
            return targetOffset;
        }

        private void remap(final int targetWord) {
            final long sourceReference = target.get(targetWord).longValue();
            if (sourceReference < 0L || sourceReference > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("semantic child offset is outside the 32-bit heap range");
            }
            target.set(targetWord, Long.valueOf(copy((int) sourceReference)));
        }

        private long[] toHeap() {
            final long[] out = new long[target.size()];
            for (int i = 0; i < out.length; i++) out[i] = target.get(i).longValue();
            return out;
        }
    }
    public static long stableStringId(final String s) {
        long h = 0xcbf29ce484222325L;
        for (int i = 0; i < s.length(); i++) {
            h ^= s.charAt(i);
            h *= 0x100000001b3L;
        }
        return h == 0L ? 1L : h;
    }

    public static final class EncodedSemanticValue {
        public final long[] heap;
        public final int root;

        private EncodedSemanticValue(final long[] heap, final int root) {
            this.heap = heap;
            this.root = root;
        }
    }

    public static final class EncodedSemanticState {
        public final String[] variableNames;
        public final long[] variableNameIds;
        public final int[] roots;
        public final long[] heap;

        EncodedSemanticState(final String[] variableNames, final long[] variableNameIds,
                final int[] roots, final long[] heap) {
            this.variableNames = variableNames;
            this.variableNameIds = variableNameIds;
            this.roots = roots;
            this.heap = heap;
        }
    }

    private static final class SemanticEncoder {
        private final ArrayList<Long> heap = new ArrayList<Long>();
        private final IdentityHashMap<IValue, Integer> seen = new IdentityHashMap<IValue, Integer>();

        int encode(final IValue value) {
            if (value == null) {
                return append(TAG_NULL, 0, 0L);
            }
            final Integer existing = seen.get(value);
            if (existing != null) {
                return existing.intValue();
            }
            if (!(value instanceof Value)) {
                return appendOpaque(value);
            }
            final Value v = (Value) value;
            if (v instanceof BoolValue) {
                return append(TAG_BOOL, 0, ((BoolValue) v).val ? 1L : 0L);
            }
            if (v instanceof IntValue) {
                return append(TAG_INT, 0, ((IntValue) v).val);
            }
            if (v instanceof ModelValue) {
                final ModelValue mv = (ModelValue) v;
                return append(TAG_MODEL, 0, stableStringId(mv.val.toString()), mv.index, mv.type);
            }
            if (v instanceof StringValue) {
                final StringValue sv = (StringValue) v;
                final long id = stableStringId(sv.getVal().toString());
                return append(TAG_STRING, 0, id, id);
            }
            if (v instanceof IntervalValue) {
                final IntervalValue iv = (IntervalValue) v;
                return append(TAG_INTERVAL, 0, 0L, iv.low, iv.high);
            }
            if (v instanceof TupleValue) {
                return appendTuple((TupleValue) v);
            }
            if (v instanceof SetEnumValue) {
                return appendSet((SetEnumValue) v);
            }
            if (v instanceof FcnRcdValue) {
                return appendFcn((FcnRcdValue) v);
            }
            if (v instanceof RecordValue) {
                return appendRecord((RecordValue) v);
            }
            return appendOpaque(v);
        }

        long[] toHeap() {
            final long[] out = new long[heap.size()];
            for (int i = 0; i < heap.size(); i++) {
                out[i] = heap.get(i).longValue();
            }
            return out;
        }

        private int appendTuple(final TupleValue tuple) {
            final Value[] elems = tuple.elems;
            final int off = reserve(TAG_TUPLE, elems.length, 0L, elems.length);
            seen.put(tuple, off);
            for (int i = 0; i < elems.length; i++) {
                heap.set(off + 1 + i, Long.valueOf(encode(elems[i])));
            }
            return off;
        }

        private int appendSet(final SetEnumValue set) {
            if (set.elems == null) {
                return appendOpaque(set);
            }
            final int size = set.elems.size();
            final int off = reserve(TAG_SET_ENUM, size, 0L, size);
            seen.put(set, off);
            final ValueVec elems = set.elems;
            for (int i = 0; i < size; i++) {
                heap.set(off + 1 + i, Long.valueOf(encode(elems.elementAt(i))));
            }
            return off;
        }

        private int appendFcn(final FcnRcdValue fcn) {
            final int size = fcn.values.length;
            if (fcn.intv != null) {
                final int off = reserve(TAG_FCN_RCD, size, FCN_DOMAIN_INTERVAL, 2 + size);
                seen.put(fcn, off);
                heap.set(off + 1, Long.valueOf(fcn.intv.low));
                heap.set(off + 2, Long.valueOf(fcn.intv.high));
                for (int i = 0; i < size; i++) {
                    heap.set(off + 3 + i, Long.valueOf(encode(fcn.values[i])));
                }
                return off;
            }
            final int off = reserve(TAG_FCN_RCD, size, FCN_DOMAIN_EXPLICIT, 2 * size);
            seen.put(fcn, off);
            for (int i = 0; i < size; i++) {
                heap.set(off + 1 + i, Long.valueOf(encode(fcn.domain[i])));
                heap.set(off + 1 + size + i, Long.valueOf(encode(fcn.values[i])));
            }
            return off;
        }

        private int appendRecord(final RecordValue record) {
            final int size = record.values.length;
            final int off = reserve(TAG_RECORD, size, 0L, 2 * size);
            seen.put(record, off);
            for (int i = 0; i < size; i++) {
                heap.set(off + 1 + i, Long.valueOf(stableStringId(record.names[i].toString())));
                heap.set(off + 1 + size + i, Long.valueOf(encode(record.values[i])));
            }
            return off;
        }

        private int appendOpaque(final IValue value) {
            return append(TAG_OPAQUE, 0, value.hashCode(), stableStringId(value.getClass().getName()));
        }

        private int append(final int tag, final int arity, final long aux, final long... payload) {
            final int off = heap.size();
            heap.add(Long.valueOf(header(tag, arity, aux)));
            for (int i = 0; i < payload.length; i++) {
                heap.add(Long.valueOf(payload[i]));
            }
            return off;
        }

        private int reserve(final int tag, final int arity, final long aux, final int payloadWords) {
            final int off = heap.size();
            heap.add(Long.valueOf(header(tag, arity, aux)));
            for (int i = 0; i < payloadWords; i++) {
                heap.add(Long.valueOf(0L));
            }
            return off;
        }
    }
}
