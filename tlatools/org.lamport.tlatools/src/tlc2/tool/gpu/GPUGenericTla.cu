#include <cuda_runtime.h>
#include <jni.h>
#include <stdio.h>
#include <cstring>
#include <chrono>
#include <stdlib.h>
#include <vector>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>


// Generic TLA+ semantic value ABI. Java side emits a flat long heap where each
// object starts with a header word: tag[63:56], arity[55:32], aux[31:0].
// Compound payload words store child heap offsets; primitive payload words store
// immediate values or stable symbol ids. This is the base layer for the
// generic GPU IR evaluator and resident frontier.
static const int GPU_TLA_TAG_NULL = 0;
static const int GPU_TLA_TAG_BOOL = 1;
static const int GPU_TLA_TAG_INT = 2;
static const int GPU_TLA_TAG_MODEL = 3;
static const int GPU_TLA_TAG_TUPLE = 4;
static const int GPU_TLA_TAG_SET_ENUM = 5;
static const int GPU_TLA_TAG_FCN_RCD = 6;
static const int GPU_TLA_TAG_RECORD = 7;
static const int GPU_TLA_TAG_INTERVAL = 8;
static const int GPU_TLA_TAG_STRING = 9;
static const int GPU_TLA_TAG_OPAQUE = 255;

static const int GPU_TLA_FCN_DOMAIN_EXPLICIT = 0;
static const int GPU_TLA_FCN_DOMAIN_INTERVAL = 1;
static const size_t GPU_TLA_DEFAULT_STACK_BYTES = 65536;

struct GpuTlaValueRef {
    const long long* heap;
    int offset;
};

__host__ __device__ __forceinline__ int gpuTlaTag(long long header) {
    return (int) ((((unsigned long long) header) >> 56) & 0xffULL);
}

__host__ __device__ __forceinline__ int gpuTlaArity(long long header) {
    return (int) ((((unsigned long long) header) >> 32) & 0x00ffffffULL);
}

__host__ __device__ __forceinline__ unsigned int gpuTlaAux(long long header) {
    return (unsigned int) (((unsigned long long) header) & 0xffffffffULL);
}

__host__ __device__ __forceinline__ long long gpuTlaMakeHeader(int tag, int arity, unsigned int aux) {
    return (long long) ((((unsigned long long) (tag & 0xff)) << 56)
            | (((unsigned long long) (arity & 0x00ffffff)) << 32)
            | ((unsigned long long) aux));
}

__host__ __device__ __forceinline__ long long gpuTlaHeader(const GpuTlaValueRef v) {
    return v.heap[v.offset];
}

__host__ __device__ __forceinline__ GpuTlaValueRef gpuTlaChild(const GpuTlaValueRef v, int index) {
    GpuTlaValueRef child;
    child.heap = v.heap;
    child.offset = (int) v.heap[v.offset + 1 + index];
    return child;
}


__host__ __device__ bool gpuTlaEquals(const GpuTlaValueRef a, const GpuTlaValueRef b);
__host__ __device__ bool gpuTlaMember(const GpuTlaValueRef elem, const GpuTlaValueRef set);
__host__ __device__ bool gpuTlaFcnApply(const GpuTlaValueRef fcn, const GpuTlaValueRef arg, GpuTlaValueRef* out);
__host__ __device__ unsigned long long gpuTlaFingerprint(const GpuTlaValueRef v);

__host__ __device__ __forceinline__ long long gpuTlaPayload(const GpuTlaValueRef v, int index) {
    return v.heap[v.offset + 1 + index];
}

__host__ __device__ __forceinline__ int gpuTlaSignedAux(long long header) {
    return (int) gpuTlaAux(header);
}

__host__ __device__ __forceinline__ bool gpuTlaRecordFieldMatches(long long fieldId, const GpuTlaValueRef key) {
    const long long kh = gpuTlaHeader(key);
    const int tag = gpuTlaTag(kh);
    if (tag == GPU_TLA_TAG_STRING) return fieldId == gpuTlaPayload(key, 0);
    if (tag == GPU_TLA_TAG_OPAQUE) return ((unsigned int) fieldId) == gpuTlaAux(kh);
    return false;
}

__host__ __device__ __forceinline__ unsigned long long gpuTlaMix64(unsigned long long x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__host__ __device__ __forceinline__ unsigned long long gpuTlaTagSalt(int tag) {
    return 0x9e3779b97f4a7c15ULL ^ ((unsigned long long) tag * 0x100000001b3ULL);
}

__host__ __device__ __forceinline__ unsigned long long gpuTlaFoldCommutative(unsigned long long seed,
        unsigned long long sum, unsigned long long xors, int arity) {
    const unsigned long long rotated = (xors << 17) | (xors >> 47);
    return gpuTlaMix64(seed ^ gpuTlaMix64(sum + 0x9e3779b97f4a7c15ULL)
            ^ rotated ^ ((unsigned long long) arity << 48));
}

__host__ __device__ __forceinline__ int gpuTlaIntervalSize(const GpuTlaValueRef interval) {
    const int low = (int) gpuTlaPayload(interval, 0);
    const int high = (int) gpuTlaPayload(interval, 1);
    return high < low ? 0 : high - low + 1;
}

__host__ __device__ bool gpuTlaTupleEqualsFcnInterval(const GpuTlaValueRef tuple, const GpuTlaValueRef fcn) {
    const int tupleArity = gpuTlaArity(gpuTlaHeader(tuple));
    const int fcnArity = gpuTlaArity(gpuTlaHeader(fcn));
    if (gpuTlaAux(gpuTlaHeader(fcn)) != GPU_TLA_FCN_DOMAIN_INTERVAL || tupleArity != fcnArity) return false;
    const int low = (int) gpuTlaPayload(fcn, 0);
    const int high = (int) gpuTlaPayload(fcn, 1);
    if (low != 1 || high != tupleArity) return false;
    for (int i = 0; i < tupleArity; i++) {
        GpuTlaValueRef fcnValue;
        fcnValue.heap = fcn.heap;
        fcnValue.offset = (int) gpuTlaPayload(fcn, 2 + i);
        if (!gpuTlaEquals(gpuTlaChild(tuple, i), fcnValue)) return false;
    }
    return true;
}

__host__ __device__ bool gpuTlaIntervalEqualsSet(const GpuTlaValueRef interval, const GpuTlaValueRef set) {
    const int setArity = gpuTlaArity(gpuTlaHeader(set));
    if (gpuTlaIntervalSize(interval) != setArity) return false;
    for (int i = 0; i < setArity; i++) {
        if (!gpuTlaMember(gpuTlaChild(set, i), interval)) return false;
    }
    return true;
}

__host__ __device__ bool gpuTlaEquals(const GpuTlaValueRef a, const GpuTlaValueRef b) {
    const long long ah = gpuTlaHeader(a);
    const long long bh = gpuTlaHeader(b);
    const int at = gpuTlaTag(ah);
    const int bt = gpuTlaTag(bh);

    if (at != bt) {
        if (at == GPU_TLA_TAG_INTERVAL && bt == GPU_TLA_TAG_SET_ENUM) return gpuTlaIntervalEqualsSet(a, b);
        if (at == GPU_TLA_TAG_SET_ENUM && bt == GPU_TLA_TAG_INTERVAL) return gpuTlaIntervalEqualsSet(b, a);
        if (at == GPU_TLA_TAG_TUPLE && bt == GPU_TLA_TAG_FCN_RCD) return gpuTlaTupleEqualsFcnInterval(a, b);
        if (at == GPU_TLA_TAG_FCN_RCD && bt == GPU_TLA_TAG_TUPLE) return gpuTlaTupleEqualsFcnInterval(b, a);
        return false;
    }

    const int arity = gpuTlaArity(ah);
    if (arity != gpuTlaArity(bh)) return false;

    switch (at) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
        return ah == bh;
    case GPU_TLA_TAG_STRING:
        return gpuTlaPayload(a, 0) == gpuTlaPayload(b, 0);
    case GPU_TLA_TAG_MODEL:
        return ah == bh && gpuTlaPayload(a, 0) == gpuTlaPayload(b, 0) && gpuTlaPayload(a, 1) == gpuTlaPayload(b, 1);
    case GPU_TLA_TAG_INTERVAL:
        return gpuTlaPayload(a, 0) == gpuTlaPayload(b, 0) && gpuTlaPayload(a, 1) == gpuTlaPayload(b, 1);
    case GPU_TLA_TAG_TUPLE:
        for (int i = 0; i < arity; i++) {
            if (!gpuTlaEquals(gpuTlaChild(a, i), gpuTlaChild(b, i))) return false;
        }
        return true;
    case GPU_TLA_TAG_SET_ENUM:
        for (int i = 0; i < arity; i++) {
            if (!gpuTlaMember(gpuTlaChild(a, i), b)) return false;
        }
        return true;
    case GPU_TLA_TAG_FCN_RCD:
        if (gpuTlaAux(ah) != gpuTlaAux(bh)) return false;
        if (gpuTlaAux(ah) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            if (gpuTlaPayload(a, 0) != gpuTlaPayload(b, 0) || gpuTlaPayload(a, 1) != gpuTlaPayload(b, 1)) return false;
            for (int i = 0; i < arity; i++) {
                GpuTlaValueRef av;
                av.heap = a.heap;
                av.offset = (int) gpuTlaPayload(a, 2 + i);
                GpuTlaValueRef bv;
                bv.heap = b.heap;
                bv.offset = (int) gpuTlaPayload(b, 2 + i);
                if (!gpuTlaEquals(av, bv)) return false;
            }
            return true;
        }
        if (gpuTlaAux(ah) != GPU_TLA_FCN_DOMAIN_EXPLICIT) return false;
        for (int i = 0; i < arity; i++) {
            GpuTlaValueRef ad;
            ad.heap = a.heap;
            ad.offset = (int) gpuTlaPayload(a, i);
            GpuTlaValueRef av;
            av.heap = a.heap;
            av.offset = (int) gpuTlaPayload(a, arity + i);
            bool found = false;
            for (int j = 0; j < arity; j++) {
                GpuTlaValueRef bd;
                bd.heap = b.heap;
                bd.offset = (int) gpuTlaPayload(b, j);
                if (!gpuTlaEquals(ad, bd)) continue;
                GpuTlaValueRef bv;
                bv.heap = b.heap;
                bv.offset = (int) gpuTlaPayload(b, arity + j);
                if (!gpuTlaEquals(av, bv)) return false;
                found = true;
                break;
            }
            if (!found) return false;
        }
        return true;
    case GPU_TLA_TAG_RECORD:
        for (int i = 0; i < arity; i++) {
            const long long field = gpuTlaPayload(a, i);
            GpuTlaValueRef av;
            av.heap = a.heap;
            av.offset = (int) gpuTlaPayload(a, arity + i);
            bool found = false;
            for (int j = 0; j < arity; j++) {
                if (field != gpuTlaPayload(b, j)) continue;
                GpuTlaValueRef bv;
                bv.heap = b.heap;
                bv.offset = (int) gpuTlaPayload(b, arity + j);
                if (!gpuTlaEquals(av, bv)) return false;
                found = true;
                break;
            }
            if (!found) return false;
        }
        return true;
    case GPU_TLA_TAG_OPAQUE:
        return ah == bh && gpuTlaPayload(a, 0) == gpuTlaPayload(b, 0);
    default:
        return false;
    }
}

__host__ __device__ bool gpuTlaMember(const GpuTlaValueRef elem, const GpuTlaValueRef set) {
    const long long sh = gpuTlaHeader(set);
    const int st = gpuTlaTag(sh);
    if (st == GPU_TLA_TAG_SET_ENUM) {
        const int arity = gpuTlaArity(sh);
        for (int i = 0; i < arity; i++) {
            if (gpuTlaEquals(elem, gpuTlaChild(set, i))) return true;
        }
        return false;
    }
    if (st == GPU_TLA_TAG_INTERVAL) {
        if (gpuTlaTag(gpuTlaHeader(elem)) != GPU_TLA_TAG_INT) return false;
        const int x = gpuTlaSignedAux(gpuTlaHeader(elem));
        const int low = (int) gpuTlaPayload(set, 0);
        const int high = (int) gpuTlaPayload(set, 1);
        return x >= low && x <= high;
    }
    return false;
}

__host__ __device__ bool gpuTlaFcnApply(const GpuTlaValueRef fcn, const GpuTlaValueRef arg, GpuTlaValueRef* out) {
    const long long fh = gpuTlaHeader(fcn);
    const int ft = gpuTlaTag(fh);
    const int arity = gpuTlaArity(fh);
    out->heap = fcn.heap;
    out->offset = -1;

    if (ft == GPU_TLA_TAG_TUPLE) {
        if (gpuTlaTag(gpuTlaHeader(arg)) != GPU_TLA_TAG_INT) return false;
        const int index = gpuTlaSignedAux(gpuTlaHeader(arg));
        if (index < 1 || index > arity) return false;
        *out = gpuTlaChild(fcn, index - 1);
        return true;
    }

    if (ft == GPU_TLA_TAG_RECORD) {
        for (int i = 0; i < arity; i++) {
            if (!gpuTlaRecordFieldMatches(gpuTlaPayload(fcn, i), arg)) continue;
            out->offset = (int) gpuTlaPayload(fcn, arity + i);
            return true;
        }
        return false;
    }

    if (ft != GPU_TLA_TAG_FCN_RCD) return false;

    if (gpuTlaAux(fh) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
        if (gpuTlaTag(gpuTlaHeader(arg)) != GPU_TLA_TAG_INT) return false;
        const int x = gpuTlaSignedAux(gpuTlaHeader(arg));
        const int low = (int) gpuTlaPayload(fcn, 0);
        const int high = (int) gpuTlaPayload(fcn, 1);
        if (x < low || x > high) return false;
        out->offset = (int) gpuTlaPayload(fcn, 2 + (x - low));
        return true;
    }

    if (gpuTlaAux(fh) != GPU_TLA_FCN_DOMAIN_EXPLICIT) return false;

    for (int i = 0; i < arity; i++) {
        GpuTlaValueRef domain;
        domain.heap = fcn.heap;
        domain.offset = (int) gpuTlaPayload(fcn, i);
        if (!gpuTlaEquals(domain, arg)) continue;
        out->offset = (int) gpuTlaPayload(fcn, arity + i);
        return true;
    }
    return false;
}

__host__ __device__ unsigned long long gpuTlaFingerprint(const GpuTlaValueRef v) {
    const long long h = gpuTlaHeader(v);
    const int tag = gpuTlaTag(h);
    const int arity = gpuTlaArity(h);
    unsigned long long fp = gpuTlaMix64(gpuTlaTagSalt(tag) ^ (unsigned long long) h);

    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
        return fp;
    case GPU_TLA_TAG_STRING:
        return gpuTlaMix64(fp ^ gpuTlaMix64((unsigned long long) gpuTlaPayload(v, 0)));
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL:
        for (int i = 0; i < 2; i++) fp ^= gpuTlaMix64((unsigned long long) gpuTlaPayload(v, i) + ((unsigned long long) i << 32));
        return gpuTlaMix64(fp);
    case GPU_TLA_TAG_OPAQUE:
        fp ^= gpuTlaMix64((unsigned long long) gpuTlaPayload(v, 0));
        return gpuTlaMix64(fp);
    case GPU_TLA_TAG_TUPLE:
        for (int i = 0; i < arity; i++) {
            fp ^= gpuTlaMix64(gpuTlaFingerprint(gpuTlaChild(v, i)) + 0x9e3779b97f4a7c15ULL + ((unsigned long long) i << 32));
        }
        return gpuTlaMix64(fp);
    case GPU_TLA_TAG_SET_ENUM: {
        unsigned long long sum = 0ULL;
        unsigned long long xors = 0ULL;
        for (int i = 0; i < arity; i++) {
            const unsigned long long child = gpuTlaFingerprint(gpuTlaChild(v, i));
            sum += child;
            xors ^= gpuTlaMix64(child + 0x517cc1b727220a95ULL);
        }
        return gpuTlaFoldCommutative(fp, sum, xors, arity);
    }
    case GPU_TLA_TAG_FCN_RCD: {
        if (gpuTlaAux(h) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            fp ^= gpuTlaMix64((unsigned long long) gpuTlaPayload(v, 0));
            fp ^= gpuTlaMix64((unsigned long long) gpuTlaPayload(v, 1) + 1ULL);
            for (int i = 0; i < arity; i++) {
                GpuTlaValueRef val;
                val.heap = v.heap;
                val.offset = (int) gpuTlaPayload(v, 2 + i);
                fp ^= gpuTlaMix64(gpuTlaFingerprint(val) + ((unsigned long long) i << 32));
            }
            return gpuTlaMix64(fp);
        }
        if (gpuTlaAux(h) != GPU_TLA_FCN_DOMAIN_EXPLICIT) return fp;
        unsigned long long sum = 0ULL;
        unsigned long long xors = 0ULL;
        for (int i = 0; i < arity; i++) {
            GpuTlaValueRef domain;
            domain.heap = v.heap;
            domain.offset = (int) gpuTlaPayload(v, i);
            GpuTlaValueRef val;
            val.heap = v.heap;
            val.offset = (int) gpuTlaPayload(v, arity + i);
            const unsigned long long pair = gpuTlaMix64(gpuTlaFingerprint(domain) ^ gpuTlaMix64(gpuTlaFingerprint(val) + 0x94d049bb133111ebULL));
            sum += pair;
            xors ^= pair;
        }
        return gpuTlaFoldCommutative(fp, sum, xors, arity);
    }
    case GPU_TLA_TAG_RECORD: {
        unsigned long long sum = 0ULL;
        unsigned long long xors = 0ULL;
        for (int i = 0; i < arity; i++) {
            GpuTlaValueRef val;
            val.heap = v.heap;
            val.offset = (int) gpuTlaPayload(v, arity + i);
            const unsigned long long pair = gpuTlaMix64((unsigned long long) gpuTlaPayload(v, i) ^ gpuTlaFingerprint(val));
            sum += pair;
            xors ^= pair;
        }
        return gpuTlaFoldCommutative(fp, sum, xors, arity);
    }
    default:
        return fp;
    }
}



static const int GPU_TLA_IR_LOAD_CONST = 1;
static const int GPU_TLA_IR_LOAD_VAR = 2;
static const int GPU_TLA_IR_LOAD_NEXT_VAR = 3;
static const int GPU_TLA_IR_EQ = 4;
static const int GPU_TLA_IR_MEMBER = 5;
static const int GPU_TLA_IR_FCN_APPLY = 6;
static const int GPU_TLA_IR_AND = 7;
static const int GPU_TLA_IR_OR = 8;
static const int GPU_TLA_IR_NOT = 9;
static const int GPU_TLA_IR_NEQ = 10;
static const int GPU_TLA_IR_NOT_MEMBER = 11;
static const int GPU_TLA_IR_IMPLIES = 12;
static const int GPU_TLA_IR_EQUIV = 13;
static const int GPU_TLA_IR_SUBSETEQ = 14;
static const int GPU_TLA_IR_JUMP_IF_FALSE = 15;
static const int GPU_TLA_IR_JUMP = 16;
static const int GPU_TLA_IR_RETURN = 17;
static const int GPU_TLA_IR_ADD = 18;
static const int GPU_TLA_IR_SUB = 19;
static const int GPU_TLA_IR_MUL = 20;
static const int GPU_TLA_IR_LT = 21;
static const int GPU_TLA_IR_LE = 22;
static const int GPU_TLA_IR_GT = 23;
static const int GPU_TLA_IR_GE = 24;
static const int GPU_TLA_IR_INTERVAL = 25;
static const int GPU_TLA_IR_NEG = 26;
static const int GPU_TLA_IR_DOMAIN = 27;
static const int GPU_TLA_IR_UNION = 28;
static const int GPU_TLA_IR_SUBSET = 29;
static const int GPU_TLA_IR_SET_UNION = 30;
static const int GPU_TLA_IR_SET_INTERSECT = 31;
static const int GPU_TLA_IR_SET_DIFF = 32;
static const int GPU_TLA_IR_SET_ENUM = 33;
static const int GPU_TLA_IR_RECORD_ONE = 34;
static const int GPU_TLA_IR_RECORD_APPEND = 35;
static const int GPU_TLA_IR_RECORD_SELECT = 36;
static const int GPU_TLA_IR_EXCEPT_UPDATE = 37;
static const int GPU_TLA_IR_RECORD_SET_ONE = 38;
static const int GPU_TLA_IR_RECORD_SET_APPEND = 39;
static const int GPU_TLA_IR_SET_OF_FCNS = 40;
static const int GPU_TLA_IR_LOAD_LOCAL = 41;
static const int GPU_TLA_IR_TUPLE = 42;
static const int GPU_TLA_IR_EXISTS_BEGIN = 43;
static const int GPU_TLA_IR_EXISTS_NEXT = 44;
static const int GPU_TLA_IR_FORALL_BEGIN = 45;
static const int GPU_TLA_IR_FORALL_NEXT = 46;
static const int GPU_TLA_IR_FILTER_BEGIN = 47;
static const int GPU_TLA_IR_FILTER_NEXT = 48;
static const int GPU_TLA_IR_MAP_BEGIN = 49;
static const int GPU_TLA_IR_MAP_NEXT = 50;
static const int GPU_TLA_IR_EXCEPT_SAVE_AT = 51;
static const int GPU_TLA_IR_SAVE_LOCAL = 52;
static const int GPU_TLA_IR_CHOOSE_CHECK = 53;
static const int GPU_TLA_IR_CHOOSE_BEGIN = 54;
static const int GPU_TLA_IR_CHOOSE_NEXT = 55;
static const int GPU_TLA_IR_FCN_BEGIN = 56;
static const int GPU_TLA_IR_FCN_NEXT = 57;
static const int GPU_TLA_IR_CARTESIAN = 58;

static const int GPU_TLA_IR_STATUS_OK = 0;
static const int GPU_TLA_IR_STATUS_STACK_OVERFLOW = 1;
static const int GPU_TLA_IR_STATUS_STACK_UNDERFLOW = 2;
static const int GPU_TLA_IR_STATUS_TYPE_ERROR = 3;
static const int GPU_TLA_IR_STATUS_BAD_OPCODE = 4;
static const int GPU_TLA_IR_STATUS_APPLY_FAILED = 5;
static const int GPU_TLA_IR_STATUS_NO_RETURN = 6;
static const int GPU_TLA_IR_STATUS_HEAP_OVERFLOW = 7;
static const int GPU_TLA_IR_STATUS_CHOOSE_FAILED = 8;

struct GpuTlaEvalSlot {
    int kind; // 0=value ref, 1=bool
    GpuTlaValueRef ref;
    bool boolean;
};

__device__ bool gpuTlaSlotToBool(const GpuTlaEvalSlot slot, bool* out) {
    if (slot.kind == 1) {
        *out = slot.boolean;
        return true;
    }
    const long long header = gpuTlaHeader(slot.ref);
    if (gpuTlaTag(header) != GPU_TLA_TAG_BOOL) return false;
    *out = gpuTlaAux(header) != 0U;
    return true;
}

__device__ bool gpuTlaPop(GpuTlaEvalSlot* stack, int* sp, GpuTlaEvalSlot* out) {
    if (*sp <= 0) return false;
    *out = stack[--(*sp)];
    return true;
}

__device__ bool gpuTlaPush(GpuTlaEvalSlot* stack, int* sp, const GpuTlaEvalSlot slot) {
    if (*sp >= 256) return false;
    stack[(*sp)++] = slot;
    return true;
}

__device__ GpuTlaEvalSlot gpuTlaRefSlot(const long long* heap, int offset) {
    GpuTlaEvalSlot slot;
    slot.kind = 0;
    slot.ref.heap = heap;
    slot.ref.offset = offset;
    slot.boolean = false;
    return slot;
}

__device__ GpuTlaEvalSlot gpuTlaBoolSlot(bool value) {
    GpuTlaEvalSlot slot;
    slot.kind = 1;
    slot.ref.heap = NULL;
    slot.ref.offset = -1;
    slot.boolean = value;
    return slot;
}

__device__ bool gpuTlaSlotToInt(const GpuTlaEvalSlot slot, int* out) {
    if (slot.kind != 0) return false;
    const long long header = gpuTlaHeader(slot.ref);
    if (gpuTlaTag(header) != GPU_TLA_TAG_INT) return false;
    *out = gpuTlaSignedAux(header);
    return true;
}

__device__ bool gpuTlaAllocWords(long long* heap, int* heapTop, int heapWords, int words, int* outOffset) {
    if (heap == NULL || heapTop == NULL || heapWords <= 0 || words <= 0) return false;
    const int off = atomicAdd(heapTop, words);
    if (off < 0 || off + words > heapWords) return false;
    *outOffset = off;
    return true;
}

__device__ bool gpuTlaAllocInt(long long* heap, int* heapTop, int heapWords, int value, GpuTlaValueRef* out) {
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_INT, 0, (unsigned int) value);
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ bool gpuTlaAllocInterval(long long* heap, int* heapTop, int heapWords, int low, int high, GpuTlaValueRef* out) {
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 3, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_INTERVAL, 0, 0U);
    heap[off + 1] = (long long) low;
    heap[off + 2] = (long long) high;
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ bool gpuTlaAllocSetEnum(long long* heap, int* heapTop, int heapWords, const int* elems, int count,
        GpuTlaValueRef* out) {
    if (count < 0) return false;
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + count, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, count, 0U);
    for (int i = 0; i < count; i++) heap[off + 1 + i] = (long long) elems[i];
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ bool gpuTlaAllocTuple(long long* heap, int* heapTop, int heapWords, const int* elems, int count,
        GpuTlaValueRef* out) {
    if (count < 0) return false;
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + count, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_TUPLE, count, 0U);
    for (int i = 0; i < count; i++) heap[off + 1 + i] = (long long) elems[i];
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ bool gpuTlaAppendUniqueOffset(const long long* heap, int* elems, int* count, int maxCount, int candidate) {
    GpuTlaValueRef cand;
    cand.heap = heap;
    cand.offset = candidate;
    for (int i = 0; i < *count; i++) {
        GpuTlaValueRef existing;
        existing.heap = heap;
        existing.offset = elems[i];
        if (gpuTlaEquals(existing, cand)) return true;
    }
    if (*count >= maxCount) return false;
    elems[(*count)++] = candidate;
    return true;
}

__device__ bool gpuTlaCollectSetElements(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef set,
        int* elems, int* count, int maxCount) {
    const int tag = gpuTlaTag(gpuTlaHeader(set));
    if (tag == GPU_TLA_TAG_SET_ENUM) {
        const int arity = gpuTlaArity(gpuTlaHeader(set));
        for (int i = 0; i < arity; i++) {
            if (!gpuTlaAppendUniqueOffset(heap, elems, count, maxCount, gpuTlaChild(set, i).offset)) return false;
        }
        return true;
    }
    if (tag == GPU_TLA_TAG_INTERVAL) {
        const int low = (int) gpuTlaPayload(set, 0);
        const int size = gpuTlaIntervalSize(set);
        for (int i = 0; i < size; i++) {
            GpuTlaValueRef iv;
            if (!gpuTlaAllocInt(heap, heapTop, heapWords, low + i, &iv)) return false;
            if (!gpuTlaAppendUniqueOffset(heap, elems, count, maxCount, iv.offset)) return false;
        }
        return true;
    }
    return false;
}

__device__ bool gpuTlaDomain(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef value, GpuTlaValueRef* out) {
    const long long h = gpuTlaHeader(value);
    const int tag = gpuTlaTag(h);
    const int arity = gpuTlaArity(h);
    if (tag == GPU_TLA_TAG_TUPLE) {
        return gpuTlaAllocInterval(heap, heapTop, heapWords, 1, arity, out);
    }
    if (tag == GPU_TLA_TAG_FCN_RCD) {
        if (gpuTlaAux(h) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            return gpuTlaAllocInterval(heap, heapTop, heapWords, (int) gpuTlaPayload(value, 0), (int) gpuTlaPayload(value, 1), out);
        }
        if (arity > 256) return false;
        int elems[256];
        int count = 0;
        for (int i = 0; i < arity; i++) {
            if (!gpuTlaAppendUniqueOffset(heap, elems, &count, 256, (int) gpuTlaPayload(value, i))) return false;
        }
        return gpuTlaAllocSetEnum(heap, heapTop, heapWords, elems, count, out);
    }
    return false;
}

__device__ bool gpuTlaUnion(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef setOfSets, GpuTlaValueRef* out) {
    if (gpuTlaTag(gpuTlaHeader(setOfSets)) != GPU_TLA_TAG_SET_ENUM) return false;
    const int outer = gpuTlaArity(gpuTlaHeader(setOfSets));
    int elems[512];
    int count = 0;
    for (int i = 0; i < outer; i++) {
        GpuTlaValueRef child = gpuTlaChild(setOfSets, i);
        const int tag = gpuTlaTag(gpuTlaHeader(child));
        if (tag == GPU_TLA_TAG_SET_ENUM) {
            const int arity = gpuTlaArity(gpuTlaHeader(child));
            for (int j = 0; j < arity; j++) {
                if (!gpuTlaAppendUniqueOffset(heap, elems, &count, 512, gpuTlaChild(child, j).offset)) return false;
            }
        } else if (tag == GPU_TLA_TAG_INTERVAL) {
            const int low = (int) gpuTlaPayload(child, 0);
            const int size = gpuTlaIntervalSize(child);
            for (int j = 0; j < size; j++) {
                GpuTlaValueRef iv;
                if (!gpuTlaAllocInt(heap, heapTop, heapWords, low + j, &iv)) return false;
                if (!gpuTlaAppendUniqueOffset(heap, elems, &count, 512, iv.offset)) return false;
            }
        } else {
            return false;
        }
    }
    return gpuTlaAllocSetEnum(heap, heapTop, heapWords, elems, count, out);
}

__device__ bool gpuTlaAllocRecord(long long* heap, int* heapTop, int heapWords, const long long* fields,
        const int* values, int count, GpuTlaValueRef* out) {
    if (count < 0) return false;
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + 2 * count, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_RECORD, count, 0U);
    for (int i = 0; i < count; i++) {
        heap[off + 1 + i] = fields[i];
        heap[off + 1 + count + i] = (long long) values[i];
    }
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ bool gpuTlaAllocFcnRcdExplicit(long long* heap, int* heapTop, int heapWords, const long long* domains,
        const int* values, int count, GpuTlaValueRef* out) {
    if (count < 0) return false;
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + 2 * count, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_FCN_RCD, count, GPU_TLA_FCN_DOMAIN_EXPLICIT);
    for (int i = 0; i < count; i++) {
        heap[off + 1 + i] = domains[i];
        heap[off + 1 + count + i] = (long long) values[i];
    }
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ __noinline__ bool gpuTlaAllocRecordSetOne(long long* heap, int* heapTop, int heapWords, long long fieldId,
        const GpuTlaValueRef valuesSet, GpuTlaValueRef* out) {
    int elems[512];
    int count = 0;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, valuesSet, elems, &count, 512)) return false;
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + count, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, count, 0U);
    for (int i = 0; i < count; i++) {
        long long fields[1];
        int values[1];
        fields[0] = fieldId;
        values[0] = elems[i];
        GpuTlaValueRef rec;
        if (!gpuTlaAllocRecord(heap, heapTop, heapWords, fields, values, 1, &rec)) return false;
        heap[off + 1 + i] = (long long) rec.offset;
    }
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ bool gpuTlaRecordAppend(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef record,
        long long fieldId, const GpuTlaValueRef value, GpuTlaValueRef* out);

__device__ __noinline__ bool gpuTlaAllocRecordSetAppend(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef recordSet,
        long long fieldId, const GpuTlaValueRef valuesSet, GpuTlaValueRef* out) {
    int records[512];
    int recordCount = 0;
    int elems[512];
    int valueCount = 0;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, recordSet, records, &recordCount, 512)) return false;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, valuesSet, elems, &valueCount, 512)) return false;
    const long long total64 = (long long) recordCount * (long long) valueCount;
    if (total64 < 0 || total64 > 512) return false;
    const int total = (int) total64;
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + total, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, total, 0U);
    int pos = 0;
    for (int i = 0; i < recordCount; i++) {
        GpuTlaValueRef record;
        record.heap = heap;
        record.offset = records[i];
        for (int j = 0; j < valueCount; j++) {
            GpuTlaValueRef value;
            value.heap = heap;
            value.offset = elems[j];
            GpuTlaValueRef rec;
            if (!gpuTlaRecordAppend(heap, heapTop, heapWords, record, fieldId, value, &rec)) return false;
            heap[off + 1 + pos++] = (long long) rec.offset;
        }
    }
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ __noinline__ bool gpuTlaAllocSetOfFcns(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef domainSet,
        const GpuTlaValueRef rangeSet, GpuTlaValueRef* out) {
    int domainOffsets[256];
    long long domains[256];
    int domainCount = 0;
    int elems[256];
    int valueCount = 0;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, domainSet, domainOffsets, &domainCount, 256)) return false;
    for (int i = 0; i < domainCount; i++) domains[i] = (long long) domainOffsets[i];
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, rangeSet, elems, &valueCount, 256)) return false;
    long long total64 = 1LL;
    for (int i = 0; i < domainCount; i++) {
        total64 *= (long long) valueCount;
        if (total64 < 0 || total64 > 512) return false;
    }
    const int total = (int) total64;
    int off = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + total, &off)) return false;
    heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, total, 0U);
    int values[256];
    for (int combo = 0; combo < total; combo++) {
        long long q = combo;
        for (int i = 0; i < domainCount; i++) {
            const int choice = (int) (q % valueCount);
            q /= valueCount;
            values[i] = elems[choice];
        }
        GpuTlaValueRef fcn;
        if (!gpuTlaAllocFcnRcdExplicit(heap, heapTop, heapWords, domains, values, domainCount, &fcn)) return false;
        heap[off + 1 + combo] = (long long) fcn.offset;
    }
    out->heap = heap;
    out->offset = off;
    return true;
}

__device__ bool gpuTlaRecordAppend(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef record,
        long long fieldId, const GpuTlaValueRef value, GpuTlaValueRef* out) {
    if (gpuTlaTag(gpuTlaHeader(record)) != GPU_TLA_TAG_RECORD) return false;
    const int arity = gpuTlaArity(gpuTlaHeader(record));
    if (arity >= 256) return false;
    long long fields[257];
    int values[257];
    for (int i = 0; i < arity; i++) {
        fields[i] = gpuTlaPayload(record, i);
        values[i] = (int) gpuTlaPayload(record, arity + i);
    }
    fields[arity] = fieldId;
    values[arity] = value.offset;
    return gpuTlaAllocRecord(heap, heapTop, heapWords, fields, values, arity + 1, out);
}

__device__ bool gpuTlaRecordSelectByField(const GpuTlaValueRef record, long long fieldId, GpuTlaValueRef* out) {
    if (gpuTlaTag(gpuTlaHeader(record)) != GPU_TLA_TAG_RECORD) return false;
    const int arity = gpuTlaArity(gpuTlaHeader(record));
    for (int i = 0; i < arity; i++) {
        if (gpuTlaPayload(record, i) == fieldId) {
            out->heap = record.heap;
            out->offset = (int) gpuTlaPayload(record, arity + i);
            return true;
        }
    }
    return false;
}

__device__ bool gpuTlaExceptUpdate(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef base,
        const int* path, int pathLen, const GpuTlaValueRef rhs, GpuTlaValueRef* out) {
    if (pathLen <= 0) {
        *out = rhs;
        return true;
    }
    GpuTlaValueRef key;
    key.heap = heap;
    key.offset = path[0];
    const int tag = gpuTlaTag(gpuTlaHeader(base));
    const int arity = gpuTlaArity(gpuTlaHeader(base));

    if (tag == GPU_TLA_TAG_FCN_RCD) {
        if (arity > 512) return false;
        int hit = -1;
        if (gpuTlaAux(gpuTlaHeader(base)) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            if (gpuTlaTag(gpuTlaHeader(key)) != GPU_TLA_TAG_INT) return false;
            const int x = gpuTlaSignedAux(gpuTlaHeader(key));
            const int low = (int) gpuTlaPayload(base, 0);
            const int high = (int) gpuTlaPayload(base, 1);
            if (x < low || x > high) return false;
            hit = x - low;
            GpuTlaValueRef oldVal;
            oldVal.heap = heap;
            oldVal.offset = (int) gpuTlaPayload(base, 2 + hit);
            GpuTlaValueRef newVal;
            if (!gpuTlaExceptUpdate(heap, heapTop, heapWords, oldVal, path + 1, pathLen - 1, rhs, &newVal)) return false;
            int off = -1;
            if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + 2 + arity, &off)) return false;
            heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_FCN_RCD, arity, GPU_TLA_FCN_DOMAIN_INTERVAL);
            heap[off + 1] = gpuTlaPayload(base, 0);
            heap[off + 2] = gpuTlaPayload(base, 1);
            for (int i = 0; i < arity; i++) heap[off + 3 + i] = gpuTlaPayload(base, 2 + i);
            heap[off + 3 + hit] = (long long) newVal.offset;
            out->heap = heap;
            out->offset = off;
            return true;
        }
        if (gpuTlaAux(gpuTlaHeader(base)) != GPU_TLA_FCN_DOMAIN_EXPLICIT) return false;
        long long domains[512];
        int values[512];
        for (int i = 0; i < arity; i++) {
            domains[i] = gpuTlaPayload(base, i);
            values[i] = (int) gpuTlaPayload(base, arity + i);
            GpuTlaValueRef domain;
            domain.heap = heap;
            domain.offset = (int) domains[i];
            if (gpuTlaEquals(domain, key)) hit = i;
        }
        if (hit < 0) return false;
        GpuTlaValueRef oldVal;
        oldVal.heap = heap;
        oldVal.offset = values[hit];
        GpuTlaValueRef newVal;
        if (!gpuTlaExceptUpdate(heap, heapTop, heapWords, oldVal, path + 1, pathLen - 1, rhs, &newVal)) return false;
        values[hit] = newVal.offset;
        return gpuTlaAllocFcnRcdExplicit(heap, heapTop, heapWords, domains, values, arity, out);
    }

    if (tag == GPU_TLA_TAG_TUPLE) {
        if (gpuTlaTag(gpuTlaHeader(key)) != GPU_TLA_TAG_INT) return false;
        const int index = gpuTlaSignedAux(gpuTlaHeader(key));
        if (index < 1 || index > arity || arity > 512) return false;
        int elems[512];
        for (int i = 0; i < arity; i++) elems[i] = gpuTlaChild(base, i).offset;
        GpuTlaValueRef oldVal = gpuTlaChild(base, index - 1);
        GpuTlaValueRef newVal;
        if (!gpuTlaExceptUpdate(heap, heapTop, heapWords, oldVal, path + 1, pathLen - 1, rhs, &newVal)) return false;
        elems[index - 1] = newVal.offset;
        int off = -1;
        if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + arity, &off)) return false;
        heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_TUPLE, arity, 0U);
        for (int i = 0; i < arity; i++) heap[off + 1 + i] = (long long) elems[i];
        out->heap = heap;
        out->offset = off;
        return true;
    }

    if (tag == GPU_TLA_TAG_RECORD) {
        if (arity > 512) return false;
        int hit = -1;
        long long fields[512];
        int values[512];
        for (int i = 0; i < arity; i++) {
            fields[i] = gpuTlaPayload(base, i);
            values[i] = (int) gpuTlaPayload(base, arity + i);
            if (gpuTlaRecordFieldMatches(fields[i], key)) hit = i;
        }
        if (hit < 0) return false;
        GpuTlaValueRef oldVal;
        oldVal.heap = heap;
        oldVal.offset = values[hit];
        GpuTlaValueRef newVal;
        if (!gpuTlaExceptUpdate(heap, heapTop, heapWords, oldVal, path + 1, pathLen - 1, rhs, &newVal)) return false;
        values[hit] = newVal.offset;
        return gpuTlaAllocRecord(heap, heapTop, heapWords, fields, values, arity, out);
    }

    return false;
}

__device__ bool gpuTlaSetUnion(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef left,
        const GpuTlaValueRef right, GpuTlaValueRef* out) {
    int elems[512];
    int count = 0;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, left, elems, &count, 512)) return false;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, right, elems, &count, 512)) return false;
    return gpuTlaAllocSetEnum(heap, heapTop, heapWords, elems, count, out);
}

__device__ bool gpuTlaSetIntersect(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef left,
        const GpuTlaValueRef right, GpuTlaValueRef* out) {
    int leftElems[512];
    int leftCount = 0;
    int elems[512];
    int count = 0;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, left, leftElems, &leftCount, 512)) return false;
    for (int i = 0; i < leftCount; i++) {
        GpuTlaValueRef elem;
        elem.heap = heap;
        elem.offset = leftElems[i];
        if (gpuTlaMember(elem, right)) {
            if (!gpuTlaAppendUniqueOffset(heap, elems, &count, 512, elem.offset)) return false;
        }
    }
    return gpuTlaAllocSetEnum(heap, heapTop, heapWords, elems, count, out);
}

__device__ bool gpuTlaSetDiff(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef left,
        const GpuTlaValueRef right, GpuTlaValueRef* out) {
    int leftElems[512];
    int leftCount = 0;
    int elems[512];
    int count = 0;
    if (!gpuTlaCollectSetElements(heap, heapTop, heapWords, left, leftElems, &leftCount, 512)) return false;
    for (int i = 0; i < leftCount; i++) {
        GpuTlaValueRef elem;
        elem.heap = heap;
        elem.offset = leftElems[i];
        if (!gpuTlaMember(elem, right)) {
            if (!gpuTlaAppendUniqueOffset(heap, elems, &count, 512, elem.offset)) return false;
        }
    }
    return gpuTlaAllocSetEnum(heap, heapTop, heapWords, elems, count, out);
}

__device__ bool gpuTlaSubset(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef baseSet, GpuTlaValueRef* out) {
    int elems[16];
    int count = 0;
    const int tag = gpuTlaTag(gpuTlaHeader(baseSet));
    if (tag == GPU_TLA_TAG_SET_ENUM) {
        const int arity = gpuTlaArity(gpuTlaHeader(baseSet));
        if (arity > 16) return false;
        for (int i = 0; i < arity; i++) {
            if (!gpuTlaAppendUniqueOffset(heap, elems, &count, 16, gpuTlaChild(baseSet, i).offset)) return false;
        }
    } else if (tag == GPU_TLA_TAG_INTERVAL) {
        const int size = gpuTlaIntervalSize(baseSet);
        if (size > 16) return false;
        const int low = (int) gpuTlaPayload(baseSet, 0);
        for (int i = 0; i < size; i++) {
            GpuTlaValueRef iv;
            if (!gpuTlaAllocInt(heap, heapTop, heapWords, low + i, &iv)) return false;
            elems[count++] = iv.offset;
        }
    } else {
        return false;
    }

    const int subsetCount = 1 << count;
    int outOff = -1;
    if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + subsetCount, &outOff)) return false;
    heap[outOff] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, subsetCount, 0U);
    out->heap = heap;
    out->offset = outOff;
    for (int mask = 0; mask < subsetCount; mask++) {
        int memberOffsets[16];
        int memberCount = 0;
        for (int i = 0; i < count; i++) {
            if (((mask >> i) & 1) != 0) memberOffsets[memberCount++] = elems[i];
        }
        GpuTlaValueRef subset;
        if (!gpuTlaAllocSetEnum(heap, heapTop, heapWords, memberOffsets, memberCount, &subset)) return false;
        heap[outOff + 1 + mask] = (long long) subset.offset;
    }
    return true;
}

__device__ bool gpuTlaPushIntResult(GpuTlaEvalSlot* stack, int* sp, long long value, long long* heap,
        int* heapTop, int heapWords) {
    if (value < -2147483648LL || value > 2147483647LL) return false;
    GpuTlaValueRef ref;
    if (!gpuTlaAllocInt(heap, heapTop, heapWords, (int) value, &ref)) return false;
    return gpuTlaPush(stack, sp, gpuTlaRefSlot(ref.heap, ref.offset));
}

__device__ bool gpuTlaEnumSize(const GpuTlaValueRef setRef, int* outSize) {
    const int tag = gpuTlaTag(gpuTlaHeader(setRef));
    if (tag == GPU_TLA_TAG_SET_ENUM) {
        *outSize = gpuTlaArity(gpuTlaHeader(setRef));
        return true;
    }
    if (tag == GPU_TLA_TAG_INTERVAL) {
        *outSize = gpuTlaIntervalSize(setRef);
        return true;
    }
    return false;
}

__device__ bool gpuTlaEnumChoiceOffset(long long* heap, int* heapTop, int heapWords, const GpuTlaValueRef setRef,
        int choice, int* outOffset) {
    const int tag = gpuTlaTag(gpuTlaHeader(setRef));
    if (tag == GPU_TLA_TAG_SET_ENUM) {
        if (choice < 0 || choice >= gpuTlaArity(gpuTlaHeader(setRef))) return false;
        *outOffset = gpuTlaChild(setRef, choice).offset;
        return true;
    }
    if (tag == GPU_TLA_TAG_INTERVAL) {
        const int low = (int) gpuTlaPayload(setRef, 0);
        const int size = gpuTlaIntervalSize(setRef);
        if (choice < 0 || choice >= size) return false;
        GpuTlaValueRef ref;
        if (!gpuTlaAllocInt(heap, heapTop, heapWords, low + choice, &ref)) return false;
        *outOffset = ref.offset;
        return true;
    }
    return false;
}

__device__ bool gpuTlaExceptLookup(const GpuTlaValueRef base, const GpuTlaValueRef key, GpuTlaValueRef* out) {
    const int tag = gpuTlaTag(gpuTlaHeader(base));
    const int arity = gpuTlaArity(gpuTlaHeader(base));
    if (tag == GPU_TLA_TAG_TUPLE) {
        if (gpuTlaTag(gpuTlaHeader(key)) != GPU_TLA_TAG_INT) return false;
        const int index = gpuTlaSignedAux(gpuTlaHeader(key));
        if (index < 1 || index > arity) return false;
        *out = gpuTlaChild(base, index - 1);
        return true;
    }
    if (tag == GPU_TLA_TAG_RECORD) {
        for (int i = 0; i < arity; i++) {
            if (gpuTlaRecordFieldMatches((long long) gpuTlaPayload(base, i), key)) {
                out->heap = base.heap;
                out->offset = (int) gpuTlaPayload(base, arity + i);
                return true;
            }
        }
        return false;
    }
    if (tag == GPU_TLA_TAG_FCN_RCD) {
        if (gpuTlaAux(gpuTlaHeader(base)) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            if (gpuTlaTag(gpuTlaHeader(key)) != GPU_TLA_TAG_INT) return false;
            const int x = gpuTlaSignedAux(gpuTlaHeader(key));
            const int low = (int) gpuTlaPayload(base, 0);
            const int high = (int) gpuTlaPayload(base, 1);
            if (x < low || x > high) return false;
            out->heap = base.heap;
            out->offset = (int) gpuTlaPayload(base, 2 + x - low);
            return true;
        }
        for (int i = 0; i < arity; i++) {
            GpuTlaValueRef domain;
            domain.heap = base.heap;
            domain.offset = (int) gpuTlaPayload(base, i);
            if (gpuTlaEquals(domain, key)) {
                out->heap = base.heap;
                out->offset = (int) gpuTlaPayload(base, arity + i);
                return true;
            }
        }
    }
    return false;
}

__device__ bool gpuTlaExceptSelect(const GpuTlaValueRef base, const int* path, int pathLen, GpuTlaValueRef* out) {
    GpuTlaValueRef cur = base;
    for (int i = 0; i < pathLen; i++) {
        GpuTlaValueRef key;
        key.heap = base.heap;
        key.offset = path[i];
        if (!gpuTlaExceptLookup(cur, key, &cur)) return false;
    }
    *out = cur;
    return true;
}

struct GpuTlaLoopFrame {
    int kind;
    int slot;
    int domainOffset;
    int size;
    int index;
    int baseSp;
    int bodyPc;
    int endPc;
    int outputOffset;
    int outputCount;
};

__device__ int gpuTlaEvalIrDevice(const long long* program, int programLen, long long* heap,
        const long long* currentRoots, const long long* nextRoots, int varCount,
        long long* localRoots, int localCount,
        long long* outBool, long long* outOffset, long long* outFingerprint, int* heapTop, int heapWords) {
    GpuTlaEvalSlot stack[256];
    int sp = 0;
    GpuTlaLoopFrame loops[64];
    int loopSp = 0;
    int pc = 0;

    while (pc + 1 < programLen) {
        const int op = (int) program[pc];
        const long long operand = program[pc + 1];
        GpuTlaEvalSlot a;
        GpuTlaEvalSlot b;
        bool ab;
        bool bb;

        switch (op) {
        case GPU_TLA_IR_LOAD_CONST:
            if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, (int) operand))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_LOAD_VAR:
            if (operand < 0 || operand >= varCount) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, (int) currentRoots[operand]))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_LOAD_NEXT_VAR:
            if (operand < 0 || operand >= varCount) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, (int) nextRoots[operand]))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_LOAD_LOCAL:
            if (operand < 0 || operand >= localCount || localRoots == NULL || localRoots[(int) operand] < 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, (int) localRoots[(int) operand]))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_SAVE_LOCAL:
            if (operand < 0 || operand >= localCount || localRoots == NULL) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            localRoots[(int) operand] = (long long) a.ref.offset;
            pc += 2;
            break;
        case GPU_TLA_IR_CHOOSE_CHECK:
            if (operand < 0 || operand >= localCount || localRoots == NULL
                    || localRoots[(int) operand] < 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (!gpuTlaSlotToBool(a, &ab)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!ab) return GPU_TLA_IR_STATUS_CHOOSE_FAILED;
            if (!gpuTlaPush(stack, &sp,
                    gpuTlaRefSlot(heap, (int) localRoots[(int) operand]))) {
                return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_EQ:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(gpuTlaEquals(a.ref, b.ref)))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_NEQ:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(!gpuTlaEquals(a.ref, b.ref)))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_MEMBER:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(gpuTlaMember(a.ref, b.ref)))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_NOT_MEMBER:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(!gpuTlaMember(a.ref, b.ref)))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_FCN_APPLY: {
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            GpuTlaValueRef result;
            if (!gpuTlaFcnApply(a.ref, b.ref, &result)) { printf("generic IR apply failed: pc=%d fcnOffset=%d fcnTag=%d argOffset=%d argTag=%d\n", pc, a.ref.offset, gpuTlaTag(gpuTlaHeader(a.ref)), b.ref.offset, gpuTlaTag(gpuTlaHeader(b.ref))); return GPU_TLA_IR_STATUS_APPLY_FAILED; }
            if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(result.heap, result.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        }
        case GPU_TLA_IR_AND:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (!gpuTlaSlotToBool(a, &ab) || !gpuTlaSlotToBool(b, &bb)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(ab && bb))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_OR:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (!gpuTlaSlotToBool(a, &ab) || !gpuTlaSlotToBool(b, &bb)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(ab || bb))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_IMPLIES:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (!gpuTlaSlotToBool(a, &ab) || !gpuTlaSlotToBool(b, &bb)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot((!ab) || bb))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_EQUIV:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (!gpuTlaSlotToBool(a, &ab) || !gpuTlaSlotToBool(b, &bb)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(ab == bb))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_SUBSETEQ:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (gpuTlaTag(gpuTlaHeader(a.ref)) != GPU_TLA_TAG_SET_ENUM) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                const int arity = gpuTlaArity(gpuTlaHeader(a.ref));
                bool subset = true;
                for (int i = 0; i < arity; i++) {
                    if (!gpuTlaMember(gpuTlaChild(a.ref, i), b.ref)) {
                        subset = false;
                        break;
                    }
                }
                if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(subset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_NOT:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (!gpuTlaSlotToBool(a, &ab)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(!ab))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            pc += 2;
            break;
        case GPU_TLA_IR_ADD:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            {
                int ai;
                int bi;
                if (!gpuTlaSlotToInt(a, &ai) || !gpuTlaSlotToInt(b, &bi)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPushIntResult(stack, &sp, (long long) ai + (long long) bi, heap, heapTop, heapWords)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_SUB:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            {
                int ai;
                int bi;
                if (!gpuTlaSlotToInt(a, &ai) || !gpuTlaSlotToInt(b, &bi)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPushIntResult(stack, &sp, (long long) ai - (long long) bi, heap, heapTop, heapWords)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_MUL:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            {
                int ai;
                int bi;
                if (!gpuTlaSlotToInt(a, &ai) || !gpuTlaSlotToInt(b, &bi)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPushIntResult(stack, &sp, (long long) ai * (long long) bi, heap, heapTop, heapWords)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_NEG:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            {
                int ai;
                if (!gpuTlaSlotToInt(a, &ai)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPushIntResult(stack, &sp, -((long long) ai), heap, heapTop, heapWords)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_LT:
        case GPU_TLA_IR_LE:
        case GPU_TLA_IR_GT:
        case GPU_TLA_IR_GE:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            {
                int ai;
                int bi;
                if (!gpuTlaSlotToInt(a, &ai) || !gpuTlaSlotToInt(b, &bi)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                bool cmp = false;
                if (op == GPU_TLA_IR_LT) cmp = ai < bi;
                else if (op == GPU_TLA_IR_LE) cmp = ai <= bi;
                else if (op == GPU_TLA_IR_GT) cmp = ai > bi;
                else cmp = ai >= bi;
                if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(cmp))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_INTERVAL:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            {
                int low;
                int high;
                if (!gpuTlaSlotToInt(a, &low) || !gpuTlaSlotToInt(b, &high)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                GpuTlaValueRef ref;
                if (!gpuTlaAllocInterval(heap, heapTop, heapWords, low, high, &ref)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_DOMAIN:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaDomain(heap, heapTop, heapWords, a.ref, &ref)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_UNION:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaUnion(heap, heapTop, heapWords, a.ref, &ref)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_SUBSET:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaSubset(heap, heapTop, heapWords, a.ref, &ref)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_SET_UNION:
        case GPU_TLA_IR_SET_INTERSECT:
        case GPU_TLA_IR_SET_DIFF:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                bool ok = false;
                if (op == GPU_TLA_IR_SET_UNION) ok = gpuTlaSetUnion(heap, heapTop, heapWords, a.ref, b.ref, &ref);
                else if (op == GPU_TLA_IR_SET_INTERSECT) ok = gpuTlaSetIntersect(heap, heapTop, heapWords, a.ref, b.ref, &ref);
                else ok = gpuTlaSetDiff(heap, heapTop, heapWords, a.ref, b.ref, &ref);
                if (!ok) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_SET_ENUM:
            if (operand < 0 || operand > 512) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                int elems[512];
                for (int i = (int) operand - 1; i >= 0; i--) {
                    if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
                    if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                    elems[i] = a.ref.offset;
                }
                GpuTlaValueRef ref;
                if (!gpuTlaAllocSetEnum(heap, heapTop, heapWords, elems, (int) operand, &ref)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_TUPLE:
            if (operand < 0 || operand > 512) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                int elems[512];
                for (int i = (int) operand - 1; i >= 0; i--) {
                    if (!gpuTlaPop(stack, &sp, &a) || a.kind != 0) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
                    elems[i] = a.ref.offset;
                }
                int off = -1;
                if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + (int) operand, &off)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                heap[off] = gpuTlaMakeHeader(GPU_TLA_TAG_TUPLE, (int) operand, 0U);
                for (int i = 0; i < (int) operand; i++) heap[off + 1 + i] = elems[i];
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, off))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_CARTESIAN: {
            if (operand <= 0 || operand > 32 || operand > sp) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            GpuTlaValueRef domains[32];
            int sizes[32];
            long long total = 1LL;
            for (int i = (int) operand - 1; i >= 0; i--) {
                if (!gpuTlaPop(stack, &sp, &a) || a.kind != 0) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
                domains[i] = a.ref;
                if (!gpuTlaEnumSize(domains[i], &sizes[i])) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (sizes[i] == 0) total = 0LL;
                if (total != 0LL) {
                    total *= (long long) sizes[i];
                    if (total > 1048576LL) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                }
            }
            int setOffset = -1;
            if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + (int) total, &setOffset)) {
                return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
            }
            heap[setOffset] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, (int) total, 0U);
            for (int combo = 0; combo < (int) total; combo++) {
                int elems[32];
                int q = combo;
                for (int i = (int) operand - 1; i >= 0; i--) {
                    const int choice = q % sizes[i];
                    q /= sizes[i];
                    if (!gpuTlaEnumChoiceOffset(heap, heapTop, heapWords, domains[i], choice, &elems[i])) {
                        return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                    }
                }
                GpuTlaValueRef tuple;
                if (!gpuTlaAllocTuple(heap, heapTop, heapWords, elems, (int) operand, &tuple)) {
                    return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                }
                heap[setOffset + 1 + combo] = (long long) tuple.offset;
            }
            if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, setOffset))) {
                return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        }
        case GPU_TLA_IR_RECORD_ONE:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                long long fields[1];
                int values[1];
                fields[0] = operand;
                values[0] = a.ref.offset;
                GpuTlaValueRef ref;
                if (!gpuTlaAllocRecord(heap, heapTop, heapWords, fields, values, 1, &ref)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_RECORD_APPEND:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaRecordAppend(heap, heapTop, heapWords, a.ref, operand, b.ref, &ref)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_RECORD_SELECT:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaRecordSelectByField(a.ref, operand, &ref)) { printf("generic IR record select failed: pc=%d offset=%d tag=%d arity=%d field=%lld first=%lld second=%lld\n", pc, a.ref.offset, gpuTlaTag(gpuTlaHeader(a.ref)), gpuTlaArity(gpuTlaHeader(a.ref)), operand, gpuTlaArity(gpuTlaHeader(a.ref)) > 0 ? gpuTlaPayload(a.ref, 0) : 0LL, gpuTlaArity(gpuTlaHeader(a.ref)) > 1 ? gpuTlaPayload(a.ref, 1) : 0LL); return GPU_TLA_IR_STATUS_APPLY_FAILED; }
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_RECORD_SET_ONE:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaAllocRecordSetOne(heap, heapTop, heapWords, operand, a.ref, &ref)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_RECORD_SET_APPEND:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaAllocRecordSetAppend(heap, heapTop, heapWords, a.ref, operand, b.ref, &ref)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_SET_OF_FCNS:
            if (!gpuTlaPop(stack, &sp, &b) || !gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind != 0 || b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                GpuTlaValueRef ref;
                if (!gpuTlaAllocSetOfFcns(heap, heapTop, heapWords, a.ref, b.ref, &ref)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_EXCEPT_UPDATE:
            if (operand <= 0 || operand > 32) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (!gpuTlaPop(stack, &sp, &b)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (b.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            {
                int path[32];
                for (int i = (int) operand - 1; i >= 0; i--) {
                    if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
                    if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                    path[i] = a.ref.offset;
                }
                if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
                if (a.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                GpuTlaValueRef ref;
                if (!gpuTlaExceptUpdate(heap, heapTop, heapWords, a.ref, path, (int) operand, b.ref, &ref)) {
                    // EXCEPT can fail either because its path is invalid or because
                    // a persistent copy-on-write update exhausted the value heap.
                    if (heapTop != NULL && atomicAdd(heapTop, 0) >= heapWords) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                    return GPU_TLA_IR_STATUS_TYPE_ERROR;
                }
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(ref.heap, ref.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            }
            pc += 2;
            break;
        case GPU_TLA_IR_EXISTS_BEGIN:
        case GPU_TLA_IR_FORALL_BEGIN:
        case GPU_TLA_IR_FILTER_BEGIN:
        case GPU_TLA_IR_MAP_BEGIN:
        case GPU_TLA_IR_CHOOSE_BEGIN:
        case GPU_TLA_IR_FCN_BEGIN: {
            if (loopSp >= 64 || localRoots == NULL || localCount <= 0) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
            if (!gpuTlaPop(stack, &sp, &a) || a.kind != 0) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            const int slot = (int) (((unsigned long long) operand) >> 32);
            const int endPc = (int) (unsigned int) operand;
            if (slot < 0 || slot >= localCount || endPc < 0 || endPc >= programLen) return GPU_TLA_IR_STATUS_BAD_OPCODE;
            GpuTlaValueRef domain = a.ref;
            int size = 0;
            if (!gpuTlaEnumSize(domain, &size)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (size == 0) {
                if (op == GPU_TLA_IR_CHOOSE_BEGIN) return GPU_TLA_IR_STATUS_CHOOSE_FAILED;
                if (op == GPU_TLA_IR_FCN_BEGIN) {
                    long long noDomains[1];
                    int noValues[1];
                    GpuTlaValueRef emptyFcn;
                    if (!gpuTlaAllocFcnRcdExplicit(heap, heapTop, heapWords,
                            noDomains, noValues, 0, &emptyFcn)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                    if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, emptyFcn.offset))) {
                        return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                    }
                } else if (op == GPU_TLA_IR_FILTER_BEGIN || op == GPU_TLA_IR_MAP_BEGIN) {
                    GpuTlaValueRef empty;
                    if (!gpuTlaAllocSetEnum(heap, heapTop, heapWords, NULL, 0, &empty)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                    if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, empty.offset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                } else if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(op == GPU_TLA_IR_FORALL_BEGIN))) {
                    return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                }
                pc = endPc;
                break;
            }
            int choiceOffset = -1;
            if (!gpuTlaEnumChoiceOffset(heap, heapTop, heapWords, domain, 0, &choiceOffset)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
            GpuTlaLoopFrame& frame = loops[loopSp++];
            frame.kind = op; frame.slot = slot; frame.domainOffset = a.ref.offset; frame.size = size;
            frame.index = 0; frame.baseSp = sp; frame.bodyPc = pc + 2; frame.endPc = endPc;
            frame.outputOffset = -1; frame.outputCount = 0;
            if (op == GPU_TLA_IR_FILTER_BEGIN || op == GPU_TLA_IR_MAP_BEGIN) {
                int outOffset = -1;
                if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + size, &outOffset)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                heap[outOffset] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, 0, 0U);
                frame.outputOffset = outOffset;
            } else if (op == GPU_TLA_IR_FCN_BEGIN) {
                int outOffset = -1;
                if (!gpuTlaAllocWords(heap, heapTop, heapWords, 1 + 2 * size, &outOffset)) {
                    return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
                }
                frame.outputOffset = outOffset;
            }
            localRoots[slot] = choiceOffset;
            pc += 2;
            break;
        }
        case GPU_TLA_IR_EXISTS_NEXT:
        case GPU_TLA_IR_FORALL_NEXT:
        case GPU_TLA_IR_FILTER_NEXT:
        case GPU_TLA_IR_MAP_NEXT:
        case GPU_TLA_IR_CHOOSE_NEXT:
        case GPU_TLA_IR_FCN_NEXT: {
            if (loopSp <= 0) return GPU_TLA_IR_STATUS_BAD_OPCODE;
            GpuTlaLoopFrame& frame = loops[loopSp - 1];
            const bool matchingLoop =
                    (op == GPU_TLA_IR_EXISTS_NEXT && frame.kind == GPU_TLA_IR_EXISTS_BEGIN)
                    || (op == GPU_TLA_IR_FORALL_NEXT && frame.kind == GPU_TLA_IR_FORALL_BEGIN)
                    || (op == GPU_TLA_IR_FILTER_NEXT && frame.kind == GPU_TLA_IR_FILTER_BEGIN)
                    || (op == GPU_TLA_IR_MAP_NEXT && frame.kind == GPU_TLA_IR_MAP_BEGIN)
                    || (op == GPU_TLA_IR_CHOOSE_NEXT && frame.kind == GPU_TLA_IR_CHOOSE_BEGIN)
                    || (op == GPU_TLA_IR_FCN_NEXT && frame.kind == GPU_TLA_IR_FCN_BEGIN);
            if (!matchingLoop) return GPU_TLA_IR_STATUS_BAD_OPCODE;
            GpuTlaEvalSlot result;
            if (!gpuTlaPop(stack, &sp, &result)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            bool keep = false;
            if (frame.kind == GPU_TLA_IR_MAP_BEGIN || frame.kind == GPU_TLA_IR_FCN_BEGIN) {
                if (result.kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                keep = true;
            } else {
                if (!gpuTlaSlotToBool(result, &keep)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            }
            if (frame.kind == GPU_TLA_IR_CHOOSE_BEGIN && keep) {
                const int chosen = (int) localRoots[frame.slot];
                const int endPc = frame.endPc;
                loopSp--;
                if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, chosen))) {
                    return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                }
                pc = endPc;
                break;
            }
            if ((frame.kind == GPU_TLA_IR_EXISTS_BEGIN && keep) || (frame.kind == GPU_TLA_IR_FORALL_BEGIN && !keep)) {
                loopSp--;
                if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(keep))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                pc = frame.endPc;
                break;
            }
            if (frame.kind == GPU_TLA_IR_FILTER_BEGIN && keep) {
                bool duplicate = false;
                for (int i = 0; i < frame.outputCount; i++) {
                    GpuTlaValueRef existing; existing.heap = heap; existing.offset = (int) heap[frame.outputOffset + 1 + i];
                    GpuTlaValueRef candidate; candidate.heap = heap; candidate.offset = (int) localRoots[frame.slot];
                    if (gpuTlaEquals(existing, candidate)) { duplicate = true; break; }
                }
                if (!duplicate) heap[frame.outputOffset + 1 + frame.outputCount++] = localRoots[frame.slot];
            } else if (frame.kind == GPU_TLA_IR_MAP_BEGIN && keep) {
                bool duplicate = false;
                for (int i = 0; i < frame.outputCount; i++) {
                    GpuTlaValueRef existing; existing.heap = heap; existing.offset = (int) heap[frame.outputOffset + 1 + i];
                    if (gpuTlaEquals(existing, result.ref)) { duplicate = true; break; }
                }
                if (!duplicate) heap[frame.outputOffset + 1 + frame.outputCount++] = result.ref.offset;
            } else if (frame.kind == GPU_TLA_IR_FCN_BEGIN) {
                heap[frame.outputOffset + 1 + frame.index] = localRoots[frame.slot];
                heap[frame.outputOffset + 1 + frame.size + frame.index] = result.ref.offset;
            }
            if (frame.index + 1 >= frame.size) {
                loopSp--;
                if (frame.kind == GPU_TLA_IR_FILTER_BEGIN || frame.kind == GPU_TLA_IR_MAP_BEGIN) {
                    heap[frame.outputOffset] = gpuTlaMakeHeader(GPU_TLA_TAG_SET_ENUM, frame.outputCount, 0U);
                    if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, frame.outputOffset))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                } else if (frame.kind == GPU_TLA_IR_FCN_BEGIN) {
                    heap[frame.outputOffset] = gpuTlaMakeHeader(
                            GPU_TLA_TAG_FCN_RCD, frame.size, GPU_TLA_FCN_DOMAIN_EXPLICIT);
                    if (!gpuTlaPush(stack, &sp, gpuTlaRefSlot(heap, frame.outputOffset))) {
                        return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                    }
                } else if (frame.kind == GPU_TLA_IR_CHOOSE_BEGIN) {
                    return GPU_TLA_IR_STATUS_CHOOSE_FAILED;
                } else {
                    const bool finalValue = frame.kind == GPU_TLA_IR_FORALL_BEGIN;
                    if (!gpuTlaPush(stack, &sp, gpuTlaBoolSlot(finalValue))) return GPU_TLA_IR_STATUS_STACK_OVERFLOW;
                }
                pc = frame.endPc;
                break;
            }
            frame.index++;
            int choiceOffset = -1;
            GpuTlaValueRef domain; domain.heap = heap; domain.offset = frame.domainOffset;
            if (!gpuTlaEnumChoiceOffset(heap, heapTop, heapWords, domain, frame.index, &choiceOffset)) return GPU_TLA_IR_STATUS_HEAP_OVERFLOW;
            localRoots[frame.slot] = choiceOffset;
            sp = frame.baseSp;
            pc = (int) operand;
            break;
        }
        case GPU_TLA_IR_EXCEPT_SAVE_AT: {
            const int slot = (int) (((unsigned long long) operand) >> 32);
            const int pathLen = (int) (unsigned int) operand;
            const int baseIndex = sp - pathLen - 1;
            if (slot < 0 || slot >= localCount || pathLen <= 0 || baseIndex < 0 || localRoots == NULL) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            if (stack[baseIndex].kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            int path[32];
            if (pathLen > 32) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            for (int i = 0; i < pathLen; i++) {
                if (stack[baseIndex + 1 + i].kind != 0) return GPU_TLA_IR_STATUS_TYPE_ERROR;
                path[i] = stack[baseIndex + 1 + i].ref.offset;
            }
            GpuTlaValueRef selected;
            if (!gpuTlaExceptSelect(stack[baseIndex].ref, path, pathLen, &selected)) return GPU_TLA_IR_STATUS_APPLY_FAILED;
            localRoots[slot] = selected.offset;
            pc += 2;
            break;
        }
        case GPU_TLA_IR_JUMP_IF_FALSE:
            if (operand < 0 || operand >= programLen || (operand % 2) != 0) return GPU_TLA_IR_STATUS_BAD_OPCODE;
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (!gpuTlaSlotToBool(a, &ab)) return GPU_TLA_IR_STATUS_TYPE_ERROR;
            pc = ab ? pc + 2 : (int) operand;
            break;
        case GPU_TLA_IR_JUMP:
            if (operand < 0 || operand >= programLen || (operand % 2) != 0) return GPU_TLA_IR_STATUS_BAD_OPCODE;
            pc = (int) operand;
            break;
        case GPU_TLA_IR_RETURN:
            if (!gpuTlaPop(stack, &sp, &a)) return GPU_TLA_IR_STATUS_STACK_UNDERFLOW;
            if (a.kind == 0) {
                *outOffset = a.ref.offset;
                *outFingerprint = (long long) gpuTlaFingerprint(a.ref);
                bool boolResult = false;
                if (gpuTlaSlotToBool(a, &boolResult)) *outBool = boolResult ? 1LL : 0LL;
                else *outBool = 0LL;
            } else {
                *outOffset = -1;
                *outFingerprint = gpuTlaMix64(a.boolean ? 1ULL : 0ULL);
                *outBool = a.boolean ? 1LL : 0LL;
            }
            return GPU_TLA_IR_STATUS_OK;
        default:
            return GPU_TLA_IR_STATUS_BAD_OPCODE;
        }
    }
    return GPU_TLA_IR_STATUS_NO_RETURN;
}

__global__ void gpuTlaEvalIrKernel(const long long* program, int programLen, long long* heap,
        const long long* currentRoots, const long long* nextRoots, int varCount,
        long long* out, int* heapTop, int heapWords) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    long long localRoots[64];
    for (int i = 0; i < 64; i++) localRoots[i] = -1LL;
    const int localCount = 64;
    long long boolResult = 0LL;
    long long offsetResult = -1LL;
    long long fingerprint = 0LL;
    const int status = gpuTlaEvalIrDevice(program, programLen, heap, currentRoots, nextRoots, varCount,
            localRoots, localCount,
            &boolResult, &offsetResult, &fingerprint, heapTop, heapWords);
    out[0] = (long long) status;
    out[1] = boolResult;
    out[2] = offsetResult;
    out[3] = fingerprint;
}


static const int GPU_TLA_SUCC_ASSIGN = 1;
static const int GPU_TLA_SUCC_ENUM = 2;
static const int GPU_TLA_SUCC_FILTER = 3;
static const int GPU_TLA_SUCC_ENUM_LOCAL = 4;

static const int GPU_TLA_SUCC_STATUS_OK = 0;
static const int GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED = 1;
static const int GPU_TLA_SUCC_STATUS_BAD_PROGRAM = 2;
static const int GPU_TLA_SUCC_STATUS_UNSUPPORTED_ENUM = 3;
static const int GPU_TLA_SUCC_STATUS_EVAL_FAILED = 4;
static const int GPU_TLA_SUCC_STATUS_SHARE_OVERFLOW = 6;
static const int GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW = 5;

__host__ __device__ int gpuTlaSuccStatusForEval(const int evalStatus) {
    return evalStatus == GPU_TLA_IR_STATUS_HEAP_OVERFLOW
            ? GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW
            : GPU_TLA_SUCC_STATUS_EVAL_FAILED;
}

__device__ int gpuTlaPublishEvalStatus(int* statusOut, const int evalStatus) {
    const int successorStatus = gpuTlaSuccStatusForEval(evalStatus);
    // Heap exhaustion is terminal for this resident arena.  It has priority
    // over an earlier ordinary evaluation failure from another GPU thread.
    return successorStatus == GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW
            ? atomicExch(statusOut, successorStatus)
            : atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, successorStatus);
}

__device__ unsigned long long gpuTlaStateFingerprint(const long long* heap, const long long* roots, int varCount) {
    unsigned long long fp = 0x6a09e667f3bcc909ULL ^ (unsigned long long) varCount;
    for (int i = 0; i < varCount; i++) {
        GpuTlaValueRef ref;
        ref.heap = heap;
        ref.offset = (int) roots[i];
        fp ^= gpuTlaMix64(gpuTlaFingerprint(ref) + 0x9e3779b97f4a7c15ULL + ((unsigned long long) i << 32));
        fp = (fp << 23) | (fp >> 41);
    }
    return gpuTlaMix64(fp);
}

__device__ unsigned long long gpuTlaStateFingerprint32(const long long* heap, const int* roots, int varCount) {
    unsigned long long fp = 0x6a09e667f3bcc909ULL ^ (unsigned long long) varCount;
    for (int i = 0; i < varCount; i++) {
        GpuTlaValueRef ref;
        ref.heap = heap;
        ref.offset = roots[i];
        fp ^= gpuTlaMix64(gpuTlaFingerprint(ref) + 0x9e3779b97f4a7c15ULL + ((unsigned long long) i << 32));
        fp = (fp << 23) | (fp >> 41);
    }
    return gpuTlaMix64(fp);
}

__device__ int gpuTlaEvalExprRef(const long long* exprProgram, const long long* exprStarts, const long long* exprLens,
        int exprIndex, long long* heap, const long long* currentRoots, const long long* nextRoots,
        int varCount, long long* localRoots, int localCount, long long* outOffset, long long* outBool, int* heapTop, int heapWords) {
    long long fingerprint = 0LL;
    const int start = (int) exprStarts[exprIndex];
    const int len = (int) exprLens[exprIndex];
    return gpuTlaEvalIrDevice(exprProgram + start, len, heap, currentRoots, nextRoots, varCount,
            localRoots, localCount,
            outBool, outOffset, &fingerprint, heapTop, heapWords);
}

__global__ void gpuTlaExpandSuccessorsKernel(const long long* branchTable, int branchWords,
        const long long* opTable, int opWords, const long long* exprProgram,
        const long long* exprStarts, const long long* exprLens, long long* heap,
        const long long* currentRoots, int varCount, int maxSuccessors, int* heapTop, int heapWords, long long* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (varCount <= 0 || varCount > 128 || branchWords % 2 != 0 || opWords % 3 != 0) {
        out[0] = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
        out[1] = varCount;
        out[2] = 0;
        out[3] = 0;
        return;
    }

    long long tempRoots[128];
    long long localRoots[64];
    long long enumSets[32];
    int enumVars[32];
    int enumLocals[32];
    int enumSizes[32];
    int emitted = 0;
    int overflow = 0;
    int status = GPU_TLA_SUCC_STATUS_OK;

    const int branchCount = branchWords / 2;
    for (int b = 0; b < branchCount && status == GPU_TLA_SUCC_STATUS_OK; b++) {
        const int opStart = (int) branchTable[2 * b];
        const int opCount = (int) branchTable[2 * b + 1];
        if (opStart < 0 || opCount < 0 || (opStart + opCount) * 3 > opWords) {
            status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
            break;
        }

        for (int i = 0; i < varCount; i++) tempRoots[i] = currentRoots[i];
        for (int i = 0; i < 64; i++) localRoots[i] = -1LL;
        int enumCount = 0;
        bool emptyBranch = false;

        for (int i = 0; i < opCount; i++) {
            const int rec = 3 * (opStart + i);
            const int op = (int) opTable[rec];
            const int var = (int) opTable[rec + 1];
            const int expr = (int) opTable[rec + 2];
            if (op != GPU_TLA_SUCC_ENUM && op != GPU_TLA_SUCC_ENUM_LOCAL) continue;
            if (enumCount >= 32 || var < 0 || (op == GPU_TLA_SUCC_ENUM ? var >= varCount : var >= 64)) {
                status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                break;
            }
            long long setOffset = -1LL;
            long long boolOut = 0LL;
            const int evalStatus = gpuTlaEvalExprRef(exprProgram, exprStarts, exprLens, expr, heap,
                    currentRoots, tempRoots, varCount, localRoots, 64, &setOffset, &boolOut, heapTop, heapWords);
            if (evalStatus != GPU_TLA_IR_STATUS_OK || setOffset < 0) {
                printf("generic expand eval failed: phase=enum branch=%d opIndex=%d succOp=%d var=%d expr=%d irStatus=%d offset=%lld bool=%lld\n",
                        b, i, op, var, expr, evalStatus, setOffset, boolOut);
                status = gpuTlaSuccStatusForEval(evalStatus);
                break;
            }
            GpuTlaValueRef setRef;
            setRef.heap = heap;
            setRef.offset = (int) setOffset;
            int size = 0;
            if (!gpuTlaEnumSize(setRef, &size)) {
                status = GPU_TLA_SUCC_STATUS_UNSUPPORTED_ENUM;
                break;
            }
            if (size == 0) {
                emptyBranch = true;
                break;
            }
            enumSets[enumCount] = setOffset;
            enumVars[enumCount] = var;
            enumLocals[enumCount] = op == GPU_TLA_SUCC_ENUM_LOCAL ? 1 : 0;
            enumSizes[enumCount] = size;
            enumCount++;
        }
        if (status != GPU_TLA_SUCC_STATUS_OK || emptyBranch) continue;

        long long combinations = 1LL;
        for (int i = 0; i < enumCount; i++) {
            combinations *= enumSizes[i];
            if (combinations < 0) {
                status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                break;
            }
        }
        if (status != GPU_TLA_SUCC_STATUS_OK) break;

        for (long long combo = 0; combo < combinations; combo++) {
            for (int i = 0; i < varCount; i++) tempRoots[i] = currentRoots[i];
        for (int i = 0; i < 64; i++) localRoots[i] = -1LL;
            long long q = combo;
            for (int e = 0; e < enumCount; e++) {
                const int choice = (int) (q % enumSizes[e]);
                q /= enumSizes[e];
                GpuTlaValueRef setRef;
                setRef.heap = heap;
                setRef.offset = (int) enumSets[e];
                int choiceOffset = -1;
                if (!gpuTlaEnumChoiceOffset(heap, heapTop, heapWords, setRef, choice, &choiceOffset)) {
                    status = GPU_TLA_SUCC_STATUS_EVAL_FAILED;
                    break;
                }
                if (enumLocals[e]) localRoots[enumVars[e]] = choiceOffset;
                else tempRoots[enumVars[e]] = choiceOffset;
            }
            if (status != GPU_TLA_SUCC_STATUS_OK) break;

            bool keep = true;
            for (int i = 0; i < opCount && keep; i++) {
                const int rec = 3 * (opStart + i);
                const int op = (int) opTable[rec];
                const int var = (int) opTable[rec + 1];
                const int expr = (int) opTable[rec + 2];
                if (op == GPU_TLA_SUCC_ENUM || op == GPU_TLA_SUCC_ENUM_LOCAL) continue;
                long long valueOffset = -1LL;
                long long boolOut = 0LL;
                const int evalStatus = gpuTlaEvalExprRef(exprProgram, exprStarts, exprLens, expr, heap,
                        currentRoots, tempRoots, varCount, localRoots, 64, &valueOffset, &boolOut, heapTop, heapWords);
                if (evalStatus != GPU_TLA_IR_STATUS_OK) {
                    printf("generic expand eval failed: phase=apply branch=%d opIndex=%d succOp=%d var=%d expr=%d irStatus=%d offset=%lld bool=%lld\n",
                            b, i, op, var, expr, evalStatus, valueOffset, boolOut);
                    status = gpuTlaSuccStatusForEval(evalStatus);
                    keep = false;
                    break;
                }
                if (op == GPU_TLA_SUCC_ASSIGN) {
                    if (var < 0 || var >= varCount || valueOffset < 0) {
                        status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                        keep = false;
                        break;
                    }
                    tempRoots[var] = valueOffset;
                } else if (op == GPU_TLA_SUCC_FILTER) {
                    keep = boolOut != 0LL;
                } else {
                    status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                    keep = false;
                    break;
                }
            }
            if (!keep || status != GPU_TLA_SUCC_STATUS_OK) continue;
            if (emitted >= maxSuccessors) {
                overflow = 1;
                status = GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED;
                break;
            }
            const int fpBase = 4;
            const int rootsBase = 4 + maxSuccessors;
            out[fpBase + emitted] = (long long) gpuTlaStateFingerprint(heap, tempRoots, varCount);
            for (int v = 0; v < varCount; v++) {
                out[rootsBase + emitted * varCount + v] = tempRoots[v];
            }
            emitted++;
        }
    }

    out[0] = status;
    out[1] = varCount;
    out[2] = emitted;
    out[3] = overflow;
}


static long long* gGenericHeap = NULL;
static long long* gGenericCompactHeap = NULL;
static int* gGenericStateRoots = NULL;
static int* gGenericNextStateRoots = NULL;
static int* gGenericCompactSizes = NULL;
static int* gGenericCompactOffsets = NULL;
static int* gGenericCurrent = NULL;
static int* gGenericNext = NULL;
static unsigned long long* gGenericVisited = NULL;
static int* gGenericCurrentSize = NULL;
static int* gGenericNextSize = NULL;
static int* gGenericStateCount = NULL;
static int* gGenericOverflow = NULL;
static int* gGenericStatus = NULL;
static int* gGenericHeapTop = NULL;
static int* gGenericCompactHeapTop = NULL;
static unsigned long long* gGenericGenerated = NULL;
static int gGenericMaxStates = 0;
static int gGenericVarCount = 0;
static int gGenericHeapWords = 0;
static int gGenericBaseHeapWords = 0;
static int gGenericHashCapacity = 0;
static int gGenericHostCurrentSize = 0;
static int gGenericDepth = 0;
static int gGenericPeakHeapWords = 0;
static long long* gGenericResidentBranch = NULL;
static long long* gGenericResidentOp = NULL;
static long long* gGenericResidentExprProgram = NULL;
static long long* gGenericResidentExprStarts = NULL;
static long long* gGenericResidentExprLens = NULL;
static int gGenericResidentBranchWords = 0;
static int gGenericResidentOpWords = 0;
static int gGenericResidentExprProgramWords = 0;
static int gGenericResidentExprCount = 0;
static int gGenericResidentConstantWords = 0;
static bool gGenericResidentProgramReady = false;
static bool gGenericGcEnabled = false;
static int gGenericGcThresholdPercent = 70;
static int gGenericGcMaxSkippedLayers = 4;
static int gGenericLayersSinceGc = 0;
static int gGenericLastLayerAllocatedWords = 0;
static int gGenericBatchStates = 32768;
static unsigned long long gGenericGcCount = 0ULL;
static unsigned long long gGenericGcSkippedCount = 0ULL;
static bool gGenericSegmented = false;
static int gGenericHostFrontierSize = 0;
static bool gGenericGpuSegmented = false;
static int* gGpuAcceptedRoots = NULL;
static int* gGpuAcceptedIndices = NULL;
static int* gGpuAcceptedCount = NULL;
static long long* gGpuOutputHeap[2] = { NULL, NULL };
static int* gGpuOutputRoots[2] = { NULL, NULL };
static long long* gGpuPinnedHeap[2] = { NULL, NULL };
static int* gGpuShareKeys = NULL;
static int* gGpuShareValues = NULL;
static int* gGpuShareSizes = NULL;
static int* gGpuShareOffsets = NULL;
static unsigned long long* gGpuShareHits = NULL;
static unsigned long long* gGpuShareMisses = NULL;
static unsigned long long* gGpuOutputShareHits[2] = { NULL, NULL };
static unsigned long long* gGpuOutputShareMisses[2] = { NULL, NULL };
static int* gGpuOutputControl[2] = { NULL, NULL };
static int* gGpuPinnedControl[2] = { NULL, NULL };
static cudaStream_t gGpuComputeStream = NULL;
static cudaStream_t gGpuCopyStream = NULL;
static cudaEvent_t gGpuComputeDone[2] = { NULL, NULL };
static cudaEvent_t gGpuCopyDone[2] = { NULL, NULL };
static int gGpuOutputHeapWords = 0;
static int gGpuOutputMaxStates = 0;
static int gGpuShareCapacity = 0;
static int gGpuJavaBatchBytes = 64 * 1024 * 1024;
static unsigned long long gGpuSegmentedDistinctCount = 0ULL;
static unsigned long long gTransferH2DBytes = 0ULL;
static unsigned long long gTransferD2HPayloadBytes = 0ULL;
static unsigned long long gTransferD2HControlBytes = 0ULL;
static unsigned long long gTransferH2DCalls = 0ULL;
static unsigned long long gTransferD2HCalls = 0ULL;
static unsigned long long gTransferH2DWaitNanos = 0ULL;
static unsigned long long gTransferD2HWaitNanos = 0ULL;
static unsigned long long gTransferSegmentHeaders = 0ULL;
static unsigned long long gTransferAcceptedStates = 0ULL;
static unsigned long long gTransferDynamicHeapWords = 0ULL;

static unsigned long long genericElapsedNanos(
        const std::chrono::steady_clock::time_point& begin) {
    return (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - begin).count();
}

static void resetGenericTransferStats() {
    gTransferH2DBytes = 0ULL;
    gTransferD2HPayloadBytes = 0ULL;
    gTransferD2HControlBytes = 0ULL;
    gTransferH2DCalls = 0ULL;
    gTransferD2HCalls = 0ULL;
    gTransferH2DWaitNanos = 0ULL;
    gTransferD2HWaitNanos = 0ULL;
    gTransferSegmentHeaders = 0ULL;
    gTransferAcceptedStates = 0ULL;
    gTransferDynamicHeapWords = 0ULL;
}
static int gGenericSegmentSourceTop = 0;


struct GenericResidentGraphControl {
    long long* heap;
    int* stateRoots;
    int* current;
    int* next;
    int* nextSize;
    int* stateCount;
    unsigned long long* visited;
    int* overflow;
    int* status;
    int* heapTop;
    unsigned long long* generated;
    int heapWords;
    int currentSize;
    int hashCapacity;
    int maxStates;
    int varCount;
    int depth;
    int maxDepth;
};

static bool gpuTlaConfigureRuntime(JNIEnv* env) {
    static bool configured = false;
    if (configured) return true;
    const cudaError_t err = cudaDeviceSetLimit(cudaLimitStackSize, GPU_TLA_DEFAULT_STACK_BYTES);
    if (err == cudaSuccess) {
        configured = true;
        return true;
    }
    char message[192];
    snprintf(message, sizeof(message), "failed to configure generic GPU thread stack: %s",
            cudaGetErrorString(err));
    jclass errorClass = env->FindClass("java/lang/RuntimeException");
    if (errorClass != NULL) env->ThrowNew(errorClass, message);
    return false;
}

static long long* gSuccBranch = NULL;
static long long* gSuccOp = NULL;
static long long* gSuccExprProgram = NULL;
static long long* gSuccExprStarts = NULL;
static long long* gSuccExprLens = NULL;
static long long* gSuccHeap = NULL;
static long long* gSuccRoots = NULL;
static long long* gSuccOut = NULL;
static int* gSuccHeapTop = NULL;
static int gSuccBranchWords = 0;
static int gSuccOpWords = 0;
static int gSuccExprProgramWords = 0;
static int gSuccExprCount = 0;
static int gSuccVarCount = 0;
static int gSuccMaxOut = 0;
static int gSuccHeapCapacity = 0;

static void freeGenericSuccessorCache() {
    if (gSuccBranch) cudaFree(gSuccBranch);
    if (gSuccOp) cudaFree(gSuccOp);
    if (gSuccExprProgram) cudaFree(gSuccExprProgram);
    if (gSuccExprStarts) cudaFree(gSuccExprStarts);
    if (gSuccExprLens) cudaFree(gSuccExprLens);
    if (gSuccHeap) cudaFree(gSuccHeap);
    if (gSuccRoots) cudaFree(gSuccRoots);
    if (gSuccOut) cudaFree(gSuccOut);
    if (gSuccHeapTop) cudaFree(gSuccHeapTop);
    gSuccBranch = NULL;
    gSuccOp = NULL;
    gSuccExprProgram = NULL;
    gSuccExprStarts = NULL;
    gSuccExprLens = NULL;
    gSuccHeap = NULL;
    gSuccRoots = NULL;
    gSuccOut = NULL;
    gSuccHeapTop = NULL;
    gSuccBranchWords = 0;
    gSuccOpWords = 0;
    gSuccExprProgramWords = 0;
    gSuccExprCount = 0;
    gSuccVarCount = 0;
    gSuccMaxOut = 0;
    gSuccHeapCapacity = 0;
}

static cudaError_t ensureGenericSuccessorCache(int branchWords, int opWords,
        int exprProgramWords, int exprCount, int varCount, int maxOut, int heapRequired) {
    const bool shapeChanged = gSuccBranch == NULL || gSuccBranchWords != branchWords
            || gSuccOpWords != opWords || gSuccExprProgramWords != exprProgramWords
            || gSuccExprCount != exprCount || gSuccVarCount != varCount || gSuccMaxOut != maxOut;
    if (shapeChanged) {
        freeGenericSuccessorCache();
        gSuccBranchWords = branchWords;
        gSuccOpWords = opWords;
        gSuccExprProgramWords = exprProgramWords;
        gSuccExprCount = exprCount;
        gSuccVarCount = varCount;
        gSuccMaxOut = maxOut;
        cudaError_t err = cudaMalloc(&gSuccBranch, (size_t) branchWords * sizeof(long long));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&gSuccOp, (size_t) (opWords == 0 ? 1 : opWords) * sizeof(long long));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&gSuccExprProgram, (size_t) exprProgramWords * sizeof(long long));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&gSuccExprStarts, (size_t) exprCount * sizeof(long long));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&gSuccExprLens, (size_t) exprCount * sizeof(long long));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&gSuccRoots, (size_t) varCount * sizeof(long long));
        if (err != cudaSuccess) return err;
        const int outWords = 4 + maxOut + maxOut * varCount;
        err = cudaMalloc(&gSuccOut, (size_t) outWords * sizeof(long long));
        if (err != cudaSuccess) return err;
        err = cudaMalloc(&gSuccHeapTop, sizeof(int));
        if (err != cudaSuccess) return err;
    }
    if (gSuccHeap == NULL || gSuccHeapCapacity < heapRequired) {
        int newCapacity = heapRequired;
        if (gSuccHeapCapacity > 0 && gSuccHeapCapacity <= 1073741823) {
            const int doubled = gSuccHeapCapacity * 2;
            if (doubled > newCapacity) newCapacity = doubled;
        }
        long long* newHeap = NULL;
        const cudaError_t err = cudaMalloc(&newHeap, (size_t) newCapacity * sizeof(long long));
        if (err != cudaSuccess) return err;
        if (gSuccHeap) cudaFree(gSuccHeap);
        gSuccHeap = newHeap;
        gSuccHeapCapacity = newCapacity;
    }
    return cudaSuccess;
}

static int nextPow2Int(int x) {
    int p = 1;
    while (p < x && p > 0) p <<= 1;
    return p > 0 ? p : x;
}

static size_t genericResidentAllocationBytes(size_t* heapBytes, size_t* rootsBytes,
        size_t* scanBytes, size_t* frontierBytes, size_t* visitedBytes) {
    const size_t maxStates = (size_t) gGenericMaxStates;
    const size_t varCount = (size_t) gGenericVarCount;
    *heapBytes = (size_t) gGenericHeapWords * sizeof(long long)
            * (gGenericGcEnabled ? 2U : 1U);
    *rootsBytes = maxStates * varCount * sizeof(int) * 2U;
    *scanBytes = maxStates * sizeof(int) * 2U;
    *frontierBytes = maxStates * sizeof(int) * 2U;
    *visitedBytes = (size_t) gGenericHashCapacity * sizeof(unsigned long long);
    return *heapBytes + *rootsBytes + *scanBytes + *frontierBytes + *visitedBytes;
}

static void genericResidentThrowPreflightError(JNIEnv* env, size_t requestedBytes,
        size_t freeBytes, size_t totalBytes, size_t heapBytes, size_t rootsBytes,
        size_t scanBytes, size_t frontierBytes, size_t visitedBytes) {
    char message[768];
    snprintf(message, sizeof(message),
            "generic GPU resident initialization requires at least %zuMiB but CUDA has "
            "%zuMiB free of %zuMiB (heap=%zuMiB%s, roots=%zuMiB, scan=%zuMiB, "
            "frontiers=%zuMiB, visited=%zuMiB; maxStates=%d, varCount=%d, "
            "heapWords=%d, hashCapacity=%d). Release GPU memory or lower "
            "-Dtlc.gpu.generic.heap.extra.words, -Dtlc.gpu.generic.max.states, and "
            "-Dtlc.gpu.generic.hash.capacity. This check excludes later IR/runtime "
            "allocations, so leave additional headroom.",
            requestedBytes / (1024 * 1024), freeBytes / (1024 * 1024),
            totalBytes / (1024 * 1024), heapBytes / (1024 * 1024),
            gGenericGcEnabled ? " (two heaps for GC)" : "", rootsBytes / (1024 * 1024),
            scanBytes / (1024 * 1024), frontierBytes / (1024 * 1024),
            visitedBytes / (1024 * 1024), gGenericMaxStates, gGenericVarCount,
            gGenericHeapWords, gGenericHashCapacity);
    jclass errorClass = env->FindClass("java/lang/RuntimeException");
    if (errorClass != NULL) env->ThrowNew(errorClass, message);
}

static void freeGenericResidentProgram() {
    if (gGenericResidentBranch) cudaFree(gGenericResidentBranch);
    if (gGenericResidentOp) cudaFree(gGenericResidentOp);
    if (gGenericResidentExprProgram) cudaFree(gGenericResidentExprProgram);
    if (gGenericResidentExprStarts) cudaFree(gGenericResidentExprStarts);
    if (gGenericResidentExprLens) cudaFree(gGenericResidentExprLens);
    gGenericResidentBranch = NULL;
    gGenericResidentOp = NULL;
    gGenericResidentExprProgram = NULL;
    gGenericResidentExprStarts = NULL;
    gGenericResidentExprLens = NULL;
    gGenericResidentBranchWords = 0;
    gGenericResidentOpWords = 0;
    gGenericResidentExprProgramWords = 0;
    gGenericResidentExprCount = 0;
    gGenericResidentConstantWords = 0;
    gGenericResidentProgramReady = false;
}

static void freeGpuSegmentedResources() {
    if (gGpuComputeStream) cudaStreamSynchronize(gGpuComputeStream);
    if (gGpuCopyStream) cudaStreamSynchronize(gGpuCopyStream);
    for (int i = 0; i < 2; i++) {
        if (gGpuComputeDone[i]) cudaEventDestroy(gGpuComputeDone[i]);
        if (gGpuCopyDone[i]) cudaEventDestroy(gGpuCopyDone[i]);
        if (gGpuOutputHeap[i]) cudaFree(gGpuOutputHeap[i]);
        if (gGpuOutputRoots[i]) cudaFree(gGpuOutputRoots[i]);
        if (gGpuOutputControl[i]) cudaFree(gGpuOutputControl[i]);
        if (gGpuPinnedHeap[i]) cudaFreeHost(gGpuPinnedHeap[i]);
        if (gGpuPinnedControl[i]) cudaFreeHost(gGpuPinnedControl[i]);
        if (gGpuOutputShareHits[i]) cudaFree(gGpuOutputShareHits[i]);
        if (gGpuOutputShareMisses[i]) cudaFree(gGpuOutputShareMisses[i]);
        gGpuComputeDone[i] = NULL;
        gGpuCopyDone[i] = NULL;
        gGpuOutputHeap[i] = NULL;
        gGpuOutputRoots[i] = NULL;
        gGpuOutputControl[i] = NULL;
        gGpuPinnedHeap[i] = NULL;
        gGpuPinnedControl[i] = NULL;
        gGpuOutputShareHits[i] = NULL;
        gGpuOutputShareMisses[i] = NULL;
    }
    if (gGpuComputeStream) cudaStreamDestroy(gGpuComputeStream);
    if (gGpuCopyStream) cudaStreamDestroy(gGpuCopyStream);
    if (gGpuAcceptedRoots) cudaFree(gGpuAcceptedRoots);
    if (gGpuAcceptedIndices) cudaFree(gGpuAcceptedIndices);
    if (gGpuAcceptedCount) cudaFree(gGpuAcceptedCount);
    if (gGpuShareKeys) cudaFree(gGpuShareKeys);
    if (gGpuShareValues) cudaFree(gGpuShareValues);
    if (gGpuShareSizes) cudaFree(gGpuShareSizes);
    if (gGpuShareOffsets) cudaFree(gGpuShareOffsets);
    if (gGpuShareHits) cudaFree(gGpuShareHits);
    if (gGpuShareMisses) cudaFree(gGpuShareMisses);
    gGpuComputeStream = NULL;
    gGpuCopyStream = NULL;
    gGpuAcceptedRoots = NULL;
    gGpuAcceptedIndices = NULL;
    gGpuAcceptedCount = NULL;
    gGpuShareKeys = NULL;
    gGpuShareValues = NULL;
    gGpuShareSizes = NULL;
    gGpuShareOffsets = NULL;
    gGpuShareHits = NULL;
    gGpuShareMisses = NULL;
    gGpuOutputHeapWords = 0;
    gGpuOutputMaxStates = 0;
    gGpuShareCapacity = 0;
    gGpuJavaBatchBytes = 64 * 1024 * 1024;
    gGpuSegmentedDistinctCount = 0ULL;
}

static void freeGenericResidentGpu() {
    freeGpuSegmentedResources();
    freeGenericResidentProgram();
    if (gGenericHeap) cudaFree(gGenericHeap);
    if (gGenericCompactHeap) cudaFree(gGenericCompactHeap);
    if (gGenericStateRoots) cudaFree(gGenericStateRoots);
    if (gGenericNextStateRoots) cudaFree(gGenericNextStateRoots);
    if (gGenericCompactSizes) cudaFree(gGenericCompactSizes);
    if (gGenericCompactOffsets) cudaFree(gGenericCompactOffsets);
    if (gGenericCurrent) cudaFree(gGenericCurrent);
    if (gGenericNext) cudaFree(gGenericNext);
    if (gGenericVisited) cudaFree(gGenericVisited);
    if (gGenericCurrentSize) cudaFree(gGenericCurrentSize);
    if (gGenericNextSize) cudaFree(gGenericNextSize);
    if (gGenericStateCount) cudaFree(gGenericStateCount);
    if (gGenericOverflow) cudaFree(gGenericOverflow);
    if (gGenericStatus) cudaFree(gGenericStatus);
    if (gGenericHeapTop) cudaFree(gGenericHeapTop);
    if (gGenericCompactHeapTop) cudaFree(gGenericCompactHeapTop);
    if (gGenericGenerated) cudaFree(gGenericGenerated);
    gGenericHeap = NULL;
    gGenericCompactHeap = NULL;
    gGenericStateRoots = NULL;
    gGenericNextStateRoots = NULL;
    gGenericCompactSizes = NULL;
    gGenericCompactOffsets = NULL;
    gGenericCurrent = NULL;
    gGenericNext = NULL;
    gGenericVisited = NULL;
    gGenericCurrentSize = NULL;
    gGenericNextSize = NULL;
    gGenericStateCount = NULL;
    gGenericOverflow = NULL;
    gGenericStatus = NULL;
    gGenericHeapTop = NULL;
    gGenericCompactHeapTop = NULL;
    gGenericGenerated = NULL;
    gGenericMaxStates = 0;
    gGenericVarCount = 0;
    gGenericHeapWords = 0;
    gGenericBaseHeapWords = 0;
    gGenericHashCapacity = 0;
    gGenericHostCurrentSize = 0;
    gGenericDepth = 0;
    gGenericPeakHeapWords = 0;
    gGenericLayersSinceGc = 0;
    gGenericLastLayerAllocatedWords = 0;
    gGenericBatchStates = 32768;
    gGenericGcCount = 0ULL;
    gGenericGcSkippedCount = 0ULL;
    gGenericSegmented = false;
    gGenericHostFrontierSize = 0;
    gGenericSegmentSourceTop = 0;
    gGenericGpuSegmented = false;
    resetGenericTransferStats();
}


static bool genericResidentEnsureProgram(JNIEnv* env,
        jlongArray branchArray, jlongArray opArray, jlongArray exprProgramArray,
        jlongArray exprStartsArray, jlongArray exprLensArray, jlongArray constantHeapArray,
        int branchLen, int opLen, int exprProgramLen, int exprCount, int constantLen) {
    if (gGenericResidentProgramReady) {
        const bool sameShape = branchLen == gGenericResidentBranchWords
                && opLen == gGenericResidentOpWords
                && exprProgramLen == gGenericResidentExprProgramWords
                && exprCount == gGenericResidentExprCount
                && constantLen == gGenericResidentConstantWords;
        if (!sameShape && gGenericStatus) {
            const int status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
            cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        }
        return sameShape;
    }

    jlong* hBranch = env->GetLongArrayElements(branchArray, NULL);
    jlong* hOp = env->GetLongArrayElements(opArray, NULL);
    jlong* hExprProgram = env->GetLongArrayElements(exprProgramArray, NULL);
    jlong* hExprStarts = env->GetLongArrayElements(exprStartsArray, NULL);
    jlong* hExprLens = env->GetLongArrayElements(exprLensArray, NULL);
    jlong* hConstant = env->GetLongArrayElements(constantHeapArray, NULL);
    cudaError_t err = cudaSuccess;
    if (hBranch == NULL || (opLen > 0 && hOp == NULL) || hExprProgram == NULL || hExprStarts == NULL
            || hExprLens == NULL || (constantLen > 0 && hConstant == NULL)) {
        err = cudaErrorMemoryAllocation;
    }
    if (err == cudaSuccess) err = cudaMalloc(&gGenericResidentBranch, (size_t) branchLen * sizeof(long long));
    if (err == cudaSuccess) err = cudaMalloc(&gGenericResidentOp, (size_t) (opLen == 0 ? 1 : opLen) * sizeof(long long));
    if (err == cudaSuccess) err = cudaMalloc(&gGenericResidentExprProgram, (size_t) exprProgramLen * sizeof(long long));
    if (err == cudaSuccess) err = cudaMalloc(&gGenericResidentExprStarts, (size_t) exprCount * sizeof(long long));
    if (err == cudaSuccess) err = cudaMalloc(&gGenericResidentExprLens, (size_t) exprCount * sizeof(long long));
    if (err == cudaSuccess) err = cudaMemcpy(gGenericResidentBranch, hBranch,
            (size_t) branchLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (err == cudaSuccess && opLen > 0) err = cudaMemcpy(gGenericResidentOp, hOp,
            (size_t) opLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(gGenericResidentExprProgram, hExprProgram,
            (size_t) exprProgramLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(gGenericResidentExprStarts, hExprStarts,
            (size_t) exprCount * sizeof(long long), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(gGenericResidentExprLens, hExprLens,
            (size_t) exprCount * sizeof(long long), cudaMemcpyHostToDevice);
    if (err == cudaSuccess && constantLen > 0) err = cudaMemcpy(
            gGenericHeap + gGenericBaseHeapWords, hConstant,
            (size_t) constantLen * sizeof(long long), cudaMemcpyHostToDevice);

    if (hBranch) env->ReleaseLongArrayElements(branchArray, hBranch, JNI_ABORT);
    if (hOp) env->ReleaseLongArrayElements(opArray, hOp, JNI_ABORT);
    if (hExprProgram) env->ReleaseLongArrayElements(exprProgramArray, hExprProgram, JNI_ABORT);
    if (hExprStarts) env->ReleaseLongArrayElements(exprStartsArray, hExprStarts, JNI_ABORT);
    if (hExprLens) env->ReleaseLongArrayElements(exprLensArray, hExprLens, JNI_ABORT);
    if (hConstant) env->ReleaseLongArrayElements(constantHeapArray, hConstant, JNI_ABORT);

    if (err != cudaSuccess) {
        freeGenericResidentProgram();
        if (gGenericStatus) {
            const int status = 1000 + (int) err;
            cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        }
        char message[256];
        snprintf(message, sizeof(message), "generic GPU resident IR upload failed: %s",
                cudaGetErrorString(err));
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        return false;
    }

    gGenericResidentBranchWords = branchLen;
    gGenericResidentOpWords = opLen;
    gGenericResidentExprProgramWords = exprProgramLen;
    gGenericResidentExprCount = exprCount;
    gGenericResidentConstantWords = constantLen;
    gGenericResidentProgramReady = true;
    const int fixedPrefixWords = gGenericBaseHeapWords + constantLen;
    cudaMemcpy(gGenericHeapTop, &fixedPrefixWords, sizeof(int), cudaMemcpyHostToDevice);
    return true;
}

__device__ __forceinline__ unsigned long long genericFingerprintKey(long fp) {
    unsigned long long key = ((unsigned long long) fp) ^ 0x9e3779b97f4a7c15ULL;
    return key == 0ULL ? 1ULL : key;
}

__device__ bool genericInsertFingerprint(unsigned long long* table, int capacity, long fp, int* overflow) {
    const unsigned long long key = genericFingerprintKey(fp);
    unsigned long long slotHash = gpuTlaMix64(key);
    for (int probe = 0; probe < capacity; probe++) {
        const int slot = (int) ((slotHash + (unsigned long long) probe) & (unsigned long long) (capacity - 1));
        const unsigned long long old = atomicCAS(table + slot, 0ULL, key);
        if (old == 0ULL) return true;
        if (old == key) return false;
    }
    atomicExch(overflow, 1);
    return false;
}
__global__ void gpuTlaFingerprintStateKernel(const long long* heap, const long long* roots,
        int varCount, unsigned long long* out) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out[0] = gpuTlaStateFingerprint(heap, roots, varCount);
    }
}

__global__ void gpuTlaDeduplicateFingerprintsKernel(const long long* fingerprints, int count,
        unsigned long long* table, int capacity, unsigned char* keep) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    unsigned long long key = ((unsigned long long) fingerprints[idx]) ^ 0x9e3779b97f4a7c15ULL;
    if (key == 0ULL) key = 1ULL;
    const unsigned long long start = gpuTlaMix64(key);
    keep[idx] = 0;
    for (int probe = 0; probe < capacity; probe++) {
        const int slot = (int) ((start + (unsigned long long) probe) & (unsigned long long) (capacity - 1));
        const unsigned long long old = atomicCAS(table + slot, 0ULL, key);
        if (old == 0ULL) {
            keep[idx] = 1;
            return;
        }
        if (old == key) return;
    }
}


__global__ void genericResidentLoadInitialKernel(const long long* heap, const long long* roots, int varCount,
        int* stateRoots, int* current, unsigned long long* visited, int* currentSize,
        int* stateCount, int* overflow, int hashCapacity, int maxStates) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (varCount <= 0 || varCount > 128 || maxStates <= 0) {
        atomicExch(overflow, 1);
        return;
    }
    const unsigned long long fp = gpuTlaStateFingerprint(heap, roots, varCount);
    if (!genericInsertFingerprint(visited, hashCapacity, (long) fp, overflow)) return;
    for (int i = 0; i < varCount; i++) stateRoots[i] = (int) roots[i];
    current[0] = 0;
    *currentSize = 1;
    *stateCount = 1;
}

__global__ void genericResidentSnapshotRootsKernel(const int* stateRoots, const int* current,
        int count, int varCount, int* snapshotRoots) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = count * varCount;
    if (index >= total) return;
    const int state = index / varCount;
    const int var = index % varCount;
    const int stateIndex = current[state];
    snapshotRoots[index] = stateRoots[(size_t) stateIndex * varCount + var];
}

__global__ void genericResidentStepKernel(const long long* branchTable, int branchWords,
        const long long* opTable, int opWords, const long long* exprProgram,
        const long long* exprStarts, const long long* exprLens, long long* heap,
        const int* stateRoots, int* allStateRoots, const int* current,
        int currentSize, int* next, int* nextSize, int* stateCount,
        unsigned long long* visited, int* overflow, int* statusOut, int* heapTop, int heapWords,
        unsigned long long* generated, int hashCapacity, int maxStates, int varCount) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= currentSize) return;
    if (varCount <= 0 || varCount > 128 || branchWords % 2 != 0 || opWords % 3 != 0) {
        atomicExch(statusOut, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }

    const int stateIndex = current[idx];
    if (stateIndex < 0 || stateIndex >= maxStates) {
        atomicExch(statusOut, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }

    long long currentRoots[128];
    long long tempRoots[128];
    long long localRoots[64];
    long long enumSets[32];
    int enumVars[32];
    int enumLocals[32];
    int enumSizes[32];
    for (int i = 0; i < varCount; i++) currentRoots[i] = stateRoots[(size_t) stateIndex * varCount + i];

    int status = GPU_TLA_SUCC_STATUS_OK;
    const int branchCount = branchWords / 2;
    for (int b = 0; b < branchCount && status == GPU_TLA_SUCC_STATUS_OK; b++) {
        const int opStart = (int) branchTable[2 * b];
        const int opCount = (int) branchTable[2 * b + 1];
        if (opStart < 0 || opCount < 0 || (opStart + opCount) * 3 > opWords) {
            status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
            break;
        }

        for (int i = 0; i < varCount; i++) tempRoots[i] = currentRoots[i];
        for (int i = 0; i < 64; i++) localRoots[i] = -1LL;
        int enumCount = 0;
        bool emptyBranch = false;

        for (int i = 0; i < opCount; i++) {
            const int rec = 3 * (opStart + i);
            const int op = (int) opTable[rec];
            const int var = (int) opTable[rec + 1];
            const int expr = (int) opTable[rec + 2];
            if (op != GPU_TLA_SUCC_ENUM && op != GPU_TLA_SUCC_ENUM_LOCAL) continue;
            if (enumCount >= 32 || var < 0 || (op == GPU_TLA_SUCC_ENUM ? var >= varCount : var >= 64) || expr < 0) {
                status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                break;
            }
            long long setOffset = -1LL;
            long long boolOut = 0LL;
            const int evalStatus = gpuTlaEvalExprRef(exprProgram, exprStarts, exprLens, expr, heap,
                    currentRoots, tempRoots, varCount, localRoots, 64, &setOffset, &boolOut, heapTop, heapWords);
            if (evalStatus != GPU_TLA_IR_STATUS_OK || setOffset < 0) {
                if (gpuTlaPublishEvalStatus(statusOut, evalStatus) == GPU_TLA_SUCC_STATUS_OK) {
                    printf("generic resident eval failed: phase=enum state=%d branch=%d opIndex=%d succOp=%d var=%d expr=%d irStatus=%d offset=%lld bool=%lld\n",
                            stateIndex, b, i, op, var, expr, evalStatus, setOffset, boolOut);
                }
                status = gpuTlaSuccStatusForEval(evalStatus);
                break;
            }
            GpuTlaValueRef setRef;
            setRef.heap = heap;
            setRef.offset = (int) setOffset;
            int size = 0;
            if (!gpuTlaEnumSize(setRef, &size)) {
                status = GPU_TLA_SUCC_STATUS_UNSUPPORTED_ENUM;
                break;
            }
            if (size == 0) {
                emptyBranch = true;
                break;
            }
            enumSets[enumCount] = setOffset;
            enumVars[enumCount] = var;
            enumLocals[enumCount] = op == GPU_TLA_SUCC_ENUM_LOCAL ? 1 : 0;
            enumSizes[enumCount] = size;
            enumCount++;
        }
        if (status != GPU_TLA_SUCC_STATUS_OK || emptyBranch) continue;

        long long combinations = 1LL;
        for (int i = 0; i < enumCount; i++) {
            combinations *= enumSizes[i];
            if (combinations < 0) {
                status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                break;
            }
        }
        if (status != GPU_TLA_SUCC_STATUS_OK) break;

        for (long long combo = 0; combo < combinations; combo++) {
            for (int i = 0; i < varCount; i++) tempRoots[i] = currentRoots[i];
        for (int i = 0; i < 64; i++) localRoots[i] = -1LL;
            long long q = combo;
            for (int e = 0; e < enumCount; e++) {
                const int choice = (int) (q % enumSizes[e]);
                q /= enumSizes[e];
                GpuTlaValueRef setRef;
                setRef.heap = heap;
                setRef.offset = (int) enumSets[e];
                int choiceOffset = -1;
                if (!gpuTlaEnumChoiceOffset(heap, heapTop, heapWords, setRef, choice, &choiceOffset)) {
                    if (atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_EVAL_FAILED) == GPU_TLA_SUCC_STATUS_OK) {
                        printf("generic resident eval failed: phase=choice state=%d branch=%d enumIndex=%d choice=%d setOffset=%lld\n",
                                stateIndex, b, e, choice, enumSets[e]);
                    }
                    status = GPU_TLA_SUCC_STATUS_EVAL_FAILED;
                    break;
                }
                if (enumLocals[e]) localRoots[enumVars[e]] = choiceOffset;
                else tempRoots[enumVars[e]] = choiceOffset;
            }
            if (status != GPU_TLA_SUCC_STATUS_OK) break;

            bool keep = true;
            for (int i = 0; i < opCount && keep; i++) {
                const int rec = 3 * (opStart + i);
                const int op = (int) opTable[rec];
                const int var = (int) opTable[rec + 1];
                const int expr = (int) opTable[rec + 2];
                if (op == GPU_TLA_SUCC_ENUM || op == GPU_TLA_SUCC_ENUM_LOCAL) continue;
                if (expr < 0) {
                    status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                    keep = false;
                    break;
                }
                long long valueOffset = -1LL;
                long long boolOut = 0LL;
                const int evalStatus = gpuTlaEvalExprRef(exprProgram, exprStarts, exprLens, expr, heap,
                        currentRoots, tempRoots, varCount, localRoots, 64, &valueOffset, &boolOut, heapTop, heapWords);
                if (evalStatus != GPU_TLA_IR_STATUS_OK) {
                    if (gpuTlaPublishEvalStatus(statusOut, evalStatus) == GPU_TLA_SUCC_STATUS_OK) {
                        printf("generic resident eval failed: phase=apply state=%d branch=%d opIndex=%d succOp=%d var=%d expr=%d irStatus=%d offset=%lld bool=%lld\n",
                                stateIndex, b, i, op, var, expr, evalStatus, valueOffset, boolOut);
                    }
                    status = gpuTlaSuccStatusForEval(evalStatus);
                    keep = false;
                    break;
                }
                if (op == GPU_TLA_SUCC_ASSIGN) {
                    if (var < 0 || var >= varCount || valueOffset < 0) {
                        status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                        keep = false;
                        break;
                    }
                    tempRoots[var] = valueOffset;
                } else if (op == GPU_TLA_SUCC_FILTER) {
                    keep = boolOut != 0LL;
                } else {
                    status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                    keep = false;
                    break;
                }
            }
            if (!keep || status != GPU_TLA_SUCC_STATUS_OK) continue;

            atomicAdd(generated, 1ULL);
            const unsigned long long fp = gpuTlaStateFingerprint(heap, tempRoots, varCount);
            if (!genericInsertFingerprint(visited, hashCapacity, (long) fp, overflow)) continue;
            const int nextPos = atomicAdd(nextSize, 1);
            if (nextPos >= maxStates) {
                atomicExch(overflow, 1);
                atomicExch(statusOut, GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED);
                continue;
            }
            for (int v = 0; v < varCount; v++) {
                allStateRoots[(size_t) nextPos * varCount + v] = (int) tempRoots[v];
            }
            next[nextPos] = nextPos;
            atomicAdd(stateCount, 1);
        }
    }

    if (status == GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW) {
        atomicExch(statusOut, status);
    } else if (status != GPU_TLA_SUCC_STATUS_OK) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, status);
    }
}


__device__ int genericResidentObjectWords(long long header) {
    const int tag = gpuTlaTag(header);
    const int arity = gpuTlaArity(header);
    if (arity < 0) return -1;
    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
        return 1;
    case GPU_TLA_TAG_STRING:
    case GPU_TLA_TAG_OPAQUE:
        return 2;
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL:
        return 3;
    case GPU_TLA_TAG_TUPLE:
    case GPU_TLA_TAG_SET_ENUM:
        return 1 + arity;
    case GPU_TLA_TAG_FCN_RCD:
        return gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL ? 3 + arity : 1 + 2 * arity;
    case GPU_TLA_TAG_RECORD:
        return 1 + 2 * arity;
    default:
        return -1;
    }
}

__device__ __noinline__ bool genericResidentCompactValue(const long long* sourceHeap,
        int sourceTop, int fixedPrefixWords, long long* targetHeap, int targetCapacity,
        int* targetTop, int sourceOffset, int depth, int* targetOffset) {
    if (sourceOffset < 0 || sourceOffset >= sourceTop || depth > 128) return false;
    if (sourceOffset < fixedPrefixWords) {
        *targetOffset = sourceOffset;
        return true;
    }

    const long long header = sourceHeap[sourceOffset];
    const int tag = gpuTlaTag(header);
    const int arity = gpuTlaArity(header);
    const int words = genericResidentObjectWords(header);
    if (words <= 0 || sourceOffset > sourceTop - words || *targetTop > targetCapacity - words) {
        return false;
    }

    const int out = *targetTop;
    *targetTop += words;
    targetHeap[out] = header;
    *targetOffset = out;

    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
        return true;
    case GPU_TLA_TAG_STRING:
    case GPU_TLA_TAG_OPAQUE:
        targetHeap[out + 1] = sourceHeap[sourceOffset + 1];
        return true;
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL:
        targetHeap[out + 1] = sourceHeap[sourceOffset + 1];
        targetHeap[out + 2] = sourceHeap[sourceOffset + 2];
        return true;
    case GPU_TLA_TAG_TUPLE:
    case GPU_TLA_TAG_SET_ENUM:
        for (int i = 0; i < arity; i++) {
            int child = -1;
            if (!genericResidentCompactValue(sourceHeap, sourceTop, fixedPrefixWords,
                    targetHeap, targetCapacity, targetTop,
                    (int) sourceHeap[sourceOffset + 1 + i], depth + 1, &child)) {
                return false;
            }
            targetHeap[out + 1 + i] = child;
        }
        return true;
    case GPU_TLA_TAG_FCN_RCD:
        if (gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            targetHeap[out + 1] = sourceHeap[sourceOffset + 1];
            targetHeap[out + 2] = sourceHeap[sourceOffset + 2];
            for (int i = 0; i < arity; i++) {
                int child = -1;
                if (!genericResidentCompactValue(sourceHeap, sourceTop, fixedPrefixWords,
                        targetHeap, targetCapacity, targetTop,
                        (int) sourceHeap[sourceOffset + 3 + i], depth + 1, &child)) {
                    return false;
                }
                targetHeap[out + 3 + i] = child;
            }
            return true;
        }
        if (gpuTlaAux(header) != GPU_TLA_FCN_DOMAIN_EXPLICIT) return false;
        for (int i = 0; i < 2 * arity; i++) {
            int child = -1;
            if (!genericResidentCompactValue(sourceHeap, sourceTop, fixedPrefixWords,
                    targetHeap, targetCapacity, targetTop,
                    (int) sourceHeap[sourceOffset + 1 + i], depth + 1, &child)) {
                return false;
            }
            targetHeap[out + 1 + i] = child;
        }
        return true;
    case GPU_TLA_TAG_RECORD:
        for (int i = 0; i < arity; i++) {
            targetHeap[out + 1 + i] = sourceHeap[sourceOffset + 1 + i];
        }
        for (int i = 0; i < arity; i++) {
            int child = -1;
            if (!genericResidentCompactValue(sourceHeap, sourceTop, fixedPrefixWords,
                    targetHeap, targetCapacity, targetTop,
                    (int) sourceHeap[sourceOffset + 1 + arity + i], depth + 1, &child)) {
                return false;
            }
            targetHeap[out + 1 + arity + i] = child;
        }
        return true;
    default:
        return false;
    }
}

__device__ __noinline__ bool genericResidentCompactValueAtomic(const long long* sourceHeap,
        int sourceTop, int fixedPrefixWords, long long* targetHeap, int targetCapacity,
        int* targetTop, int sourceOffset, int depth, int* targetOffset) {
    if (sourceOffset < 0 || sourceOffset >= sourceTop || depth > 128) return false;
    if (sourceOffset < fixedPrefixWords) {
        *targetOffset = sourceOffset;
        return true;
    }
    const long long header = sourceHeap[sourceOffset];
    const int tag = gpuTlaTag(header);
    const int arity = gpuTlaArity(header);
    const int words = genericResidentObjectWords(header);
    if (words <= 0 || sourceOffset > sourceTop - words) return false;
    const int out = atomicAdd(targetTop, words);
    if (out < fixedPrefixWords || out > targetCapacity - words) return false;
    targetHeap[out] = header;
    *targetOffset = out;
    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
        return true;
    case GPU_TLA_TAG_STRING:
    case GPU_TLA_TAG_OPAQUE:
        targetHeap[out + 1] = sourceHeap[sourceOffset + 1];
        return true;
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL:
        targetHeap[out + 1] = sourceHeap[sourceOffset + 1];
        targetHeap[out + 2] = sourceHeap[sourceOffset + 2];
        return true;
    case GPU_TLA_TAG_TUPLE:
    case GPU_TLA_TAG_SET_ENUM:
        for (int i = 0; i < arity; i++) {
            int child = -1;
            if (!genericResidentCompactValueAtomic(sourceHeap, sourceTop, fixedPrefixWords,
                    targetHeap, targetCapacity, targetTop,
                    (int) sourceHeap[sourceOffset + 1 + i], depth + 1, &child)) return false;
            targetHeap[out + 1 + i] = child;
        }
        return true;
    case GPU_TLA_TAG_FCN_RCD:
        if (gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            targetHeap[out + 1] = sourceHeap[sourceOffset + 1];
            targetHeap[out + 2] = sourceHeap[sourceOffset + 2];
            for (int i = 0; i < arity; i++) {
                int child = -1;
                if (!genericResidentCompactValueAtomic(sourceHeap, sourceTop, fixedPrefixWords,
                        targetHeap, targetCapacity, targetTop,
                        (int) sourceHeap[sourceOffset + 3 + i], depth + 1, &child)) return false;
                targetHeap[out + 3 + i] = child;
            }
            return true;
        }
        if (gpuTlaAux(header) != GPU_TLA_FCN_DOMAIN_EXPLICIT) return false;
        for (int i = 0; i < 2 * arity; i++) {
            int child = -1;
            if (!genericResidentCompactValueAtomic(sourceHeap, sourceTop, fixedPrefixWords,
                    targetHeap, targetCapacity, targetTop,
                    (int) sourceHeap[sourceOffset + 1 + i], depth + 1, &child)) return false;
            targetHeap[out + 1 + i] = child;
        }
        return true;
    case GPU_TLA_TAG_RECORD:
        for (int i = 0; i < arity; i++) targetHeap[out + 1 + i] = sourceHeap[sourceOffset + 1 + i];
        for (int i = 0; i < arity; i++) {
            int child = -1;
            if (!genericResidentCompactValueAtomic(sourceHeap, sourceTop, fixedPrefixWords,
                    targetHeap, targetCapacity, targetTop,
                    (int) sourceHeap[sourceOffset + 1 + arity + i], depth + 1, &child)) return false;
            targetHeap[out + 1 + arity + i] = child;
        }
        return true;
    default:
        return false;
    }
}

__device__ __noinline__ bool genericResidentMeasureValue(const long long* sourceHeap,
        int sourceTop, int fixedPrefixWords, int sourceOffset, int depth, int* wordsOut) {
    if (sourceOffset < 0 || sourceOffset >= sourceTop || depth > 128) return false;
    if (sourceOffset < fixedPrefixWords) {
        *wordsOut = 0;
        return true;
    }
    const long long header = sourceHeap[sourceOffset];
    const int tag = gpuTlaTag(header);
    const int arity = gpuTlaArity(header);
    const int words = genericResidentObjectWords(header);
    if (words <= 0 || sourceOffset > sourceTop - words) return false;

    long long total = words;
    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
    case GPU_TLA_TAG_STRING:
    case GPU_TLA_TAG_OPAQUE:
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL:
        *wordsOut = words;
        return true;
    case GPU_TLA_TAG_TUPLE:
    case GPU_TLA_TAG_SET_ENUM:
        for (int i = 0; i < arity; i++) {
            int childWords = 0;
            if (!genericResidentMeasureValue(sourceHeap, sourceTop, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + i], depth + 1, &childWords)) return false;
            total += childWords;
        }
        break;
    case GPU_TLA_TAG_FCN_RCD:
        if (gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            for (int i = 0; i < arity; i++) {
                int childWords = 0;
                if (!genericResidentMeasureValue(sourceHeap, sourceTop, fixedPrefixWords,
                        (int) sourceHeap[sourceOffset + 3 + i], depth + 1, &childWords)) return false;
                total += childWords;
            }
            break;
        }
        if (gpuTlaAux(header) != GPU_TLA_FCN_DOMAIN_EXPLICIT) return false;
        for (int i = 0; i < 2 * arity; i++) {
            int childWords = 0;
            if (!genericResidentMeasureValue(sourceHeap, sourceTop, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + i], depth + 1, &childWords)) return false;
            total += childWords;
        }
        break;
    case GPU_TLA_TAG_RECORD:
        for (int i = 0; i < arity; i++) {
            int childWords = 0;
            if (!genericResidentMeasureValue(sourceHeap, sourceTop, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + arity + i], depth + 1, &childWords)) return false;
            total += childWords;
        }
        break;
    default:
        return false;
    }
    if (total < 0LL || total > 2147483647LL) return false;
    *wordsOut = (int) total;
    return true;
}

__global__ void genericResidentMeasureBatchKernel(const long long* sourceHeap, int sourceTop,
        int fixedPrefixWords, const int* roots, const int* stateIndices,
        int firstState, int stateCount, int varCount,
        int* sizes, int* statusOut) {
    const int localState = blockIdx.x * blockDim.x + threadIdx.x;
    if (localState >= stateCount) return;
    const int state = stateIndices[firstState + localState];
    long long total = 0LL;
    for (int var = 0; var < varCount; var++) {
        int words = 0;
        const int sourceOffset = roots[(size_t) state * varCount + var];
        if (!genericResidentMeasureValue(sourceHeap, sourceTop, fixedPrefixWords, sourceOffset, 0, &words)) {
            sizes[localState] = 0;
            atomicExch(statusOut, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
            return;
        }
        total += words;
        if (total > 2147483647LL) {
            sizes[localState] = 0;
            atomicExch(statusOut, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
            return;
        }
    }
    sizes[localState] = (int) total;
}

__global__ void genericResidentCopyMeasuredBatchKernel(const long long* sourceHeap, int sourceTop,
        long long* targetHeap, int targetCapacity, int fixedPrefixWords, int targetBase,
        int* roots, const int* stateIndices, int firstState, int stateCount, int varCount, const int* offsets,
        const int* sizes, int* statusOut) {
    const int localState = blockIdx.x * blockDim.x + threadIdx.x;
    if (localState >= stateCount) return;
    const int state = stateIndices[firstState + localState];
    int localTop = targetBase + offsets[localState];
    const int expectedTop = localTop + sizes[localState];
    if (localTop < fixedPrefixWords || expectedTop < localTop || expectedTop > targetCapacity) {
        atomicExch(statusOut, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
        return;
    }
    for (int var = 0; var < varCount; var++) {
        const size_t rootIndex = (size_t) state * varCount + var;
        int compactOffset = -1;
        if (!genericResidentCompactValue(sourceHeap, sourceTop, fixedPrefixWords,
                targetHeap, targetCapacity, &localTop, roots[rootIndex], 0, &compactOffset)) {
            atomicExch(statusOut, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
            return;
        }
        roots[rootIndex] = compactOffset;
    }
    if (localTop != expectedTop) {
        atomicExch(statusOut, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
    }
}

static bool genericResidentCompactBatchMeasured(const long long* sourceHeap, int sourceTop,
        long long* targetHeap, int targetCapacity, int fixedPrefixWords, int* roots,
        const int* stateIndices, int firstState, int stateCount, int varCount,
        int* targetTop, int* statusOut) {
    if (stateCount <= 0) return true;
    if (gGenericCompactSizes == NULL || gGenericCompactOffsets == NULL || stateCount > gGenericMaxStates) {
        const int status = GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED;
        cudaMemcpy(statusOut, &status, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }

    const int threads = 128;
    const int blocks = (stateCount + threads - 1) / threads;
    genericResidentMeasureBatchKernel<<<blocks, threads>>>(sourceHeap, sourceTop, fixedPrefixWords, roots,
            stateIndices, firstState, stateCount, varCount, gGenericCompactSizes, statusOut);
    cudaError_t launchErr = cudaGetLastError();
    cudaError_t syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
    if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
        const int status = 1000 + (int) (launchErr != cudaSuccess ? launchErr : syncErr);
        cudaMemcpy(statusOut, &status, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }
    int status = GPU_TLA_SUCC_STATUS_OK;
    cudaMemcpy(&status, statusOut, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != GPU_TLA_SUCC_STATUS_OK) return false;

    try {
        thrust::exclusive_scan(thrust::device,
                thrust::device_pointer_cast(gGenericCompactSizes),
                thrust::device_pointer_cast(gGenericCompactSizes + stateCount),
                thrust::device_pointer_cast(gGenericCompactOffsets));
    } catch (...) {
        const int scanStatus = GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW;
        cudaMemcpy(statusOut, &scanStatus, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }
    const cudaError_t scanErr = cudaDeviceSynchronize();
    if (scanErr != cudaSuccess) {
        const int scanStatus = 1000 + (int) scanErr;
        cudaMemcpy(statusOut, &scanStatus, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }

    int targetBase = fixedPrefixWords;
    int lastOffset = 0;
    int lastSize = 0;
    cudaMemcpy(&targetBase, targetTop, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastOffset, gGenericCompactOffsets + stateCount - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastSize, gGenericCompactSizes + stateCount - 1, sizeof(int), cudaMemcpyDeviceToHost);
    const long long targetEndLong = (long long) targetBase + (long long) lastOffset + (long long) lastSize;
    if (targetBase < fixedPrefixWords || targetEndLong < targetBase || targetEndLong > targetCapacity) {
        const int overflowStatus = GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW;
        cudaMemcpy(statusOut, &overflowStatus, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }

    genericResidentCopyMeasuredBatchKernel<<<blocks, threads>>>(sourceHeap, sourceTop, targetHeap, targetCapacity,
            fixedPrefixWords, targetBase, roots, stateIndices, firstState, stateCount, varCount,
            gGenericCompactOffsets, gGenericCompactSizes, statusOut);
    launchErr = cudaGetLastError();
    syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
    if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
        const int copyStatus = 1000 + (int) (launchErr != cudaSuccess ? launchErr : syncErr);
        cudaMemcpy(statusOut, &copyStatus, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }
    cudaMemcpy(&status, statusOut, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != GPU_TLA_SUCC_STATUS_OK) return false;

    const int targetEnd = (int) targetEndLong;
    cudaMemcpy(targetTop, &targetEnd, sizeof(int), cudaMemcpyHostToDevice);
    return true;
}

__global__ void genericResidentCompactBatchKernel(const long long* sourceHeap, int sourceTop,
        long long* targetHeap, int targetCapacity, int fixedPrefixWords, int* roots,
        int firstState, int stateCount, int varCount, int* targetTop, int* statusOut) {
    const int localState = blockIdx.x * blockDim.x + threadIdx.x;
    if (localState >= stateCount) return;
    const int state = firstState + localState;
    for (int var = 0; var < varCount; var++) {
        const size_t rootIndex = (size_t) state * varCount + var;
        const int sourceOffset = (int) roots[rootIndex];
        int compactOffset = -1;
        if (!genericResidentCompactValueAtomic(sourceHeap, sourceTop, fixedPrefixWords,
                targetHeap, targetCapacity, targetTop, sourceOffset, 0, &compactOffset)) {
            atomicExch(statusOut, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
            return;
        }
        roots[rootIndex] = compactOffset;
    }
}

__global__ void genericResidentCompactLayerKernel(const long long* sourceHeap,
        long long* targetHeap, int heapCapacity, int fixedPrefixWords,
        const int* sourceRoots, int* targetRoots, int stateCount,
        int varCount, int* heapTop, int* statusOut) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int sourceTop = *heapTop;
    if (fixedPrefixWords < 0 || fixedPrefixWords > sourceTop || sourceTop > heapCapacity) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }

    for (int i = 0; i < fixedPrefixWords; i++) targetHeap[i] = sourceHeap[i];
    int targetTop = fixedPrefixWords;
    for (int state = 0; state < stateCount; state++) {
        for (int var = 0; var < varCount; var++) {
            int compactOffset = -1;
            const int sourceOffset = (int) sourceRoots[(size_t) state * varCount + var];
            if (!genericResidentCompactValue(sourceHeap, sourceTop, fixedPrefixWords,
                    targetHeap, heapCapacity, &targetTop, sourceOffset, 0, &compactOffset)) {
                atomicExch(statusOut, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
                return;
            }
            targetRoots[(size_t) state * varCount + var] = compactOffset;
        }
    }
    *heapTop = targetTop;
}

static bool genericResidentCompactLayer(int nextSize, int fixedPrefixWords) {
    int sourceHeapTop = 0;
    cudaMemcpy(&sourceHeapTop, gGenericHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
    if (sourceHeapTop > gGenericPeakHeapWords) gGenericPeakHeapWords = sourceHeapTop;
    genericResidentCompactLayerKernel<<<1, 1>>>(gGenericHeap, gGenericCompactHeap,
            gGenericHeapWords, fixedPrefixWords, gGenericNextStateRoots,
            gGenericStateRoots, nextSize, gGenericVarCount, gGenericHeapTop,
            gGenericStatus);
    const cudaError_t launchErr = cudaGetLastError();
    const cudaError_t syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
    if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
        const int status = 1000 + (launchErr != cudaSuccess ? (int) launchErr : (int) syncErr);
        cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }
    int status = GPU_TLA_SUCC_STATUS_OK;
    cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
    if (status != GPU_TLA_SUCC_STATUS_OK) return false;
    long long* oldHeap = gGenericHeap;
    gGenericHeap = gGenericCompactHeap;
    gGenericCompactHeap = oldHeap;
    return true;
}

static int genericResidentReadHeapTop() {
    int heapTop = 0;
    if (gGenericHeapTop) cudaMemcpy(&heapTop, gGenericHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
    if (heapTop > gGenericPeakHeapWords) gGenericPeakHeapWords = heapTop;
    return heapTop;
}

static bool genericResidentFinishLayer(int nextSize, int fixedPrefixWords, int heapTopBefore) {
    const int heapTopAfter = genericResidentReadHeapTop();
    const int allocatedWords = heapTopAfter > heapTopBefore ? heapTopAfter - heapTopBefore : 0;
    if (!gGenericGcEnabled) {
        gGenericLastLayerAllocatedWords = allocatedWords;
        int* roots = gGenericStateRoots;
        gGenericStateRoots = gGenericNextStateRoots;
        gGenericNextStateRoots = roots;
        gGenericLayersSinceGc++;
        gGenericGcSkippedCount++;
        return true;
    }
    const long long remainingWords = (long long) gGenericHeapWords - heapTopAfter;
    const long long recentAllocation = allocatedWords > gGenericLastLayerAllocatedWords
            ? allocatedWords : gGenericLastLayerAllocatedWords;
    const long long predictedNextWords = recentAllocation * 3LL;
    const bool usageTrigger = (long long) heapTopAfter * 100LL
            >= (long long) gGenericHeapWords * gGenericGcThresholdPercent;
    const bool reserveTrigger = remainingWords < predictedNextWords;
    const bool intervalTrigger = gGenericGcMaxSkippedLayers <= 0
            || gGenericLayersSinceGc >= gGenericGcMaxSkippedLayers;
    const bool compact = nextSize <= 0 || usageTrigger || reserveTrigger || intervalTrigger;

    gGenericLastLayerAllocatedWords = allocatedWords;
    if (compact) {
        if (!genericResidentCompactLayer(nextSize, fixedPrefixWords)) return false;
        gGenericLayersSinceGc = 0;
        gGenericGcCount++;
    } else {
        int* roots = gGenericStateRoots;
        gGenericStateRoots = gGenericNextStateRoots;
        gGenericNextStateRoots = roots;
        gGenericLayersSinceGc++;
        gGenericGcSkippedCount++;
    }
    return true;
}

static bool genericResidentRunBatchedLayer(const long long* branchTable, int branchWords,
        const long long* opTable, int opWords, const long long* exprProgram,
        const long long* exprStarts, const long long* exprLens, int fixedPrefixWords) {
    if (!gGenericGcEnabled || gGenericCompactHeap == NULL || gGenericCompactHeapTop == NULL) return false;
    const int sourceBaseTop = genericResidentReadHeapTop();
    if (sourceBaseTop < fixedPrefixWords || sourceBaseTop > gGenericHeapWords) {
        const int status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
        cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }
    cudaError_t err = cudaMemcpy(gGenericCompactHeap, gGenericHeap,
            (size_t) fixedPrefixWords * sizeof(long long), cudaMemcpyDeviceToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(gGenericCompactHeapTop, &fixedPrefixWords,
            sizeof(int), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemset(gGenericNextSize, 0, sizeof(int));
    if (err == cudaSuccess) err = cudaMemset(gGenericStatus, 0, sizeof(int));
    if (err == cudaSuccess) err = cudaMemset(gGenericOverflow, 0, sizeof(int));
    if (err != cudaSuccess) {
        const int status = 1000 + (int) err;
        cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }

    const int threads = 128;
    for (int begin = 0; begin < gGenericHostCurrentSize; begin += gGenericBatchStates) {
        const int count = (gGenericHostCurrentSize - begin) < gGenericBatchStates
                ? (gGenericHostCurrentSize - begin) : gGenericBatchStates;
        int batchFirst = 0;
        cudaMemcpy(&batchFirst, gGenericNextSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(gGenericHeapTop, &sourceBaseTop, sizeof(int), cudaMemcpyHostToDevice);
        const int blocks = (count + threads - 1) / threads;
        genericResidentStepKernel<<<blocks, threads>>>(branchTable, branchWords, opTable, opWords,
                exprProgram, exprStarts, exprLens, gGenericHeap, gGenericStateRoots,
                gGenericNextStateRoots, gGenericCurrent + begin, count, gGenericNext,
                gGenericNextSize, gGenericStateCount, gGenericVisited, gGenericOverflow,
                gGenericStatus, gGenericHeapTop, gGenericHeapWords, gGenericGenerated,
                gGenericHashCapacity, gGenericMaxStates, gGenericVarCount);
        cudaError_t launchErr = cudaGetLastError();
        cudaError_t syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
        if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
            const int status = 1000 + (int) (launchErr != cudaSuccess ? launchErr : syncErr);
            cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
            return false;
        }
        int status = 0;
        int overflow = 0;
        int batchEnd = batchFirst;
        int sourceTop = sourceBaseTop;
        cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&overflow, gGenericOverflow, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&batchEnd, gGenericNextSize, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&sourceTop, gGenericHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
        if (sourceTop > gGenericPeakHeapWords) gGenericPeakHeapWords = sourceTop;
        if (status != GPU_TLA_SUCC_STATUS_OK || overflow != 0) return false;
        if (batchEnd > batchFirst) {
            const int compactCount = batchEnd - batchFirst;
            if (!genericResidentCompactBatchMeasured(gGenericHeap, sourceTop,
                    gGenericCompactHeap, gGenericHeapWords, fixedPrefixWords,
                    gGenericNextStateRoots, gGenericNext, batchFirst, compactCount, gGenericVarCount,
                    gGenericCompactHeapTop, gGenericStatus)) {
                return false;
            }
        }
    }

    int targetTop = 0;
    int nextSize = 0;
    cudaMemcpy(&targetTop, gGenericCompactHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&nextSize, gGenericNextSize, sizeof(int), cudaMemcpyDeviceToHost);
    if (targetTop < fixedPrefixWords || targetTop > gGenericHeapWords) {
        const int status = GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW;
        cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        return false;
    }
    if (targetTop > gGenericPeakHeapWords) gGenericPeakHeapWords = targetTop;
    long long* oldHeap = gGenericHeap;
    gGenericHeap = gGenericCompactHeap;
    gGenericCompactHeap = oldHeap;
    cudaMemcpy(gGenericHeapTop, &targetTop, sizeof(int), cudaMemcpyHostToDevice);
    int* oldRoots = gGenericStateRoots;
    gGenericStateRoots = gGenericNextStateRoots;
    gGenericNextStateRoots = oldRoots;
    int* oldCurrent = gGenericCurrent;
    gGenericCurrent = gGenericNext;
    gGenericNext = oldCurrent;
    gGenericHostCurrentSize = nextSize;
    gGenericDepth++;
    gGenericLastLayerAllocatedWords = targetTop - fixedPrefixWords;
    gGenericLayersSinceGc = 0;
    gGenericGcCount++;
    return true;
}

__global__ void genericResidentGraphStepKernel(const long long* branchTable, int branchWords,
        const long long* opTable, int opWords, const long long* exprProgram,
        const long long* exprStarts, const long long* exprLens, GenericResidentGraphControl* ctl) {
    long long* heap = ctl->heap;
    int* stateRoots = ctl->stateRoots;
    int* allStateRoots = ctl->stateRoots;
    const int* current = ctl->current;
    const int currentSize = ctl->currentSize;
    int* next = ctl->next;
    int* nextSize = ctl->nextSize;
    int* stateCount = ctl->stateCount;
    unsigned long long* visited = ctl->visited;
    int* overflow = ctl->overflow;
    int* statusOut = ctl->status;
    int* heapTop = ctl->heapTop;
    const int heapWords = ctl->heapWords;
    unsigned long long* generated = ctl->generated;
    const int hashCapacity = ctl->hashCapacity;
    const int maxStates = ctl->maxStates;
    const int varCount = ctl->varCount;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= currentSize) return;
    if (varCount <= 0 || varCount > 128 || branchWords % 2 != 0 || opWords % 3 != 0) {
        atomicExch(statusOut, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }

    const int stateIndex = current[idx];
    if (stateIndex < 0 || stateIndex >= maxStates) {
        atomicExch(statusOut, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }

    long long currentRoots[128];
    long long tempRoots[128];
    long long localRoots[64];
    long long enumSets[32];
    int enumVars[32];
    int enumLocals[32];
    int enumSizes[32];
    for (int i = 0; i < varCount; i++) currentRoots[i] = stateRoots[(size_t) stateIndex * varCount + i];

    int status = GPU_TLA_SUCC_STATUS_OK;
    const int branchCount = branchWords / 2;
    for (int b = 0; b < branchCount && status == GPU_TLA_SUCC_STATUS_OK; b++) {
        const int opStart = (int) branchTable[2 * b];
        const int opCount = (int) branchTable[2 * b + 1];
        if (opStart < 0 || opCount < 0 || (opStart + opCount) * 3 > opWords) {
            status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
            break;
        }

        for (int i = 0; i < varCount; i++) tempRoots[i] = currentRoots[i];
        for (int i = 0; i < 64; i++) localRoots[i] = -1LL;
        int enumCount = 0;
        bool emptyBranch = false;

        for (int i = 0; i < opCount; i++) {
            const int rec = 3 * (opStart + i);
            const int op = (int) opTable[rec];
            const int var = (int) opTable[rec + 1];
            const int expr = (int) opTable[rec + 2];
            if (op != GPU_TLA_SUCC_ENUM && op != GPU_TLA_SUCC_ENUM_LOCAL) continue;
            if (enumCount >= 32 || var < 0 || (op == GPU_TLA_SUCC_ENUM ? var >= varCount : var >= 64) || expr < 0) {
                status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                break;
            }
            long long setOffset = -1LL;
            long long boolOut = 0LL;
            const int evalStatus = gpuTlaEvalExprRef(exprProgram, exprStarts, exprLens, expr, heap,
                    currentRoots, tempRoots, varCount, localRoots, 64, &setOffset, &boolOut, heapTop, heapWords);
            if (evalStatus != GPU_TLA_IR_STATUS_OK || setOffset < 0) {
                if (gpuTlaPublishEvalStatus(statusOut, evalStatus) == GPU_TLA_SUCC_STATUS_OK) {
                    printf("generic resident eval failed: phase=enum state=%d branch=%d opIndex=%d succOp=%d var=%d expr=%d irStatus=%d offset=%lld bool=%lld\n",
                            stateIndex, b, i, op, var, expr, evalStatus, setOffset, boolOut);
                }
                status = gpuTlaSuccStatusForEval(evalStatus);
                break;
            }
            GpuTlaValueRef setRef;
            setRef.heap = heap;
            setRef.offset = (int) setOffset;
            int size = 0;
            if (!gpuTlaEnumSize(setRef, &size)) {
                status = GPU_TLA_SUCC_STATUS_UNSUPPORTED_ENUM;
                break;
            }
            if (size == 0) {
                emptyBranch = true;
                break;
            }
            enumSets[enumCount] = setOffset;
            enumVars[enumCount] = var;
            enumLocals[enumCount] = op == GPU_TLA_SUCC_ENUM_LOCAL ? 1 : 0;
            enumSizes[enumCount] = size;
            enumCount++;
        }
        if (status != GPU_TLA_SUCC_STATUS_OK || emptyBranch) continue;

        long long combinations = 1LL;
        for (int i = 0; i < enumCount; i++) {
            combinations *= enumSizes[i];
            if (combinations < 0) {
                status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                break;
            }
        }
        if (status != GPU_TLA_SUCC_STATUS_OK) break;

        for (long long combo = 0; combo < combinations; combo++) {
            for (int i = 0; i < varCount; i++) tempRoots[i] = currentRoots[i];
        for (int i = 0; i < 64; i++) localRoots[i] = -1LL;
            long long q = combo;
            for (int e = 0; e < enumCount; e++) {
                const int choice = (int) (q % enumSizes[e]);
                q /= enumSizes[e];
                GpuTlaValueRef setRef;
                setRef.heap = heap;
                setRef.offset = (int) enumSets[e];
                int choiceOffset = -1;
                if (!gpuTlaEnumChoiceOffset(heap, heapTop, heapWords, setRef, choice, &choiceOffset)) {
                    if (atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_EVAL_FAILED) == GPU_TLA_SUCC_STATUS_OK) {
                        printf("generic resident eval failed: phase=choice state=%d branch=%d enumIndex=%d choice=%d setOffset=%lld\n",
                                stateIndex, b, e, choice, enumSets[e]);
                    }
                    status = GPU_TLA_SUCC_STATUS_EVAL_FAILED;
                    break;
                }
                if (enumLocals[e]) localRoots[enumVars[e]] = choiceOffset;
                else tempRoots[enumVars[e]] = choiceOffset;
            }
            if (status != GPU_TLA_SUCC_STATUS_OK) break;

            bool keep = true;
            for (int i = 0; i < opCount && keep; i++) {
                const int rec = 3 * (opStart + i);
                const int op = (int) opTable[rec];
                const int var = (int) opTable[rec + 1];
                const int expr = (int) opTable[rec + 2];
                if (op == GPU_TLA_SUCC_ENUM || op == GPU_TLA_SUCC_ENUM_LOCAL) continue;
                if (expr < 0) {
                    status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                    keep = false;
                    break;
                }
                long long valueOffset = -1LL;
                long long boolOut = 0LL;
                const int evalStatus = gpuTlaEvalExprRef(exprProgram, exprStarts, exprLens, expr, heap,
                        currentRoots, tempRoots, varCount, localRoots, 64, &valueOffset, &boolOut, heapTop, heapWords);
                if (evalStatus != GPU_TLA_IR_STATUS_OK) {
                    if (gpuTlaPublishEvalStatus(statusOut, evalStatus) == GPU_TLA_SUCC_STATUS_OK) {
                        printf("generic resident eval failed: phase=apply state=%d branch=%d opIndex=%d succOp=%d var=%d expr=%d irStatus=%d offset=%lld bool=%lld\n",
                                stateIndex, b, i, op, var, expr, evalStatus, valueOffset, boolOut);
                    }
                    status = gpuTlaSuccStatusForEval(evalStatus);
                    keep = false;
                    break;
                }
                if (op == GPU_TLA_SUCC_ASSIGN) {
                    if (var < 0 || var >= varCount || valueOffset < 0) {
                        status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                        keep = false;
                        break;
                    }
                    tempRoots[var] = valueOffset;
                } else if (op == GPU_TLA_SUCC_FILTER) {
                    keep = boolOut != 0LL;
                } else {
                    status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
                    keep = false;
                    break;
                }
            }
            if (!keep || status != GPU_TLA_SUCC_STATUS_OK) continue;

            atomicAdd(generated, 1ULL);
            const unsigned long long fp = gpuTlaStateFingerprint(heap, tempRoots, varCount);
            if (!genericInsertFingerprint(visited, hashCapacity, (long) fp, overflow)) continue;
            const int pos = atomicAdd(stateCount, 1);
            if (pos >= maxStates) {
                atomicExch(overflow, 1);
                atomicExch(statusOut, GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED);
                continue;
            }
            for (int v = 0; v < varCount; v++) allStateRoots[(size_t) pos * varCount + v] = (int) tempRoots[v];
            const int nextPos = atomicAdd(nextSize, 1);
            if (nextPos >= maxStates) {
                atomicExch(overflow, 1);
                atomicExch(statusOut, GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED);
                continue;
            }
            next[nextPos] = pos;
        }
    }

    if (status == GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW) {
        atomicExch(statusOut, status);
    } else if (status != GPU_TLA_SUCC_STATUS_OK) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, status);
    }
}


__global__ void genericResidentGraphControlKernel(GenericResidentGraphControl* ctl,
        cudaGraphConditionalHandle handle) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const int nextSize = ctl->nextSize ? *(ctl->nextSize) : 0;
    const int status = ctl->status ? *(ctl->status) : GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
    const int overflow = ctl->overflow ? *(ctl->overflow) : 1;
    int* tmp = ctl->current;
    ctl->current = ctl->next;
    ctl->next = tmp;
    ctl->currentSize = nextSize;
    ctl->depth++;
    const unsigned int keepGoing = (nextSize > 0 && status == 0 && overflow == 0 && ctl->depth < ctl->maxDepth) ? 1U : 0U;
    cudaGraphSetConditional(handle, keepGoing);
}

extern "C"
JNIEXPORT void JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierInit(JNIEnv* env, jclass,
        jint maxStates, jint varCount, jint heapWords, jint hashCapacity, jint stackBytes, jboolean gcEnabled,
        jint gcThresholdPercent, jint gcMaxSkippedLayers, jint batchStates, jint frontierMode,
        jint gpuOutputHeapWords, jint gpuOutputMaxStates, jint gpuShareCapacity,
        jint gpuJavaBatchBytes) {
    freeGenericResidentGpu();
    if (maxStates <= 0 || varCount <= 0 || varCount > 128 || heapWords <= 0) return;
    gGenericMaxStates = (int) maxStates;
    gGenericVarCount = (int) varCount;
    gGenericHeapWords = (int) heapWords;
    gGenericBaseHeapWords = 0;
    gGenericHashCapacity = hashCapacity > 0 ? nextPow2Int((int) hashCapacity) : nextPow2Int(gGenericMaxStates * 2);
    gGenericGpuSegmented = frontierMode == 2;
    gGenericSegmented = frontierMode != 0;
    gGenericGcEnabled = frontierMode == 1 || (frontierMode == 0 && gcEnabled == JNI_TRUE);
    gGenericGcThresholdPercent = gcThresholdPercent < 10 ? 10
            : (gcThresholdPercent > 95 ? 95 : (int) gcThresholdPercent);
    gGenericGcMaxSkippedLayers = gcMaxSkippedLayers < 0 ? 0 : (int) gcMaxSkippedLayers;
    gGenericBatchStates = batchStates < 1024 ? 1024 : (int) batchStates;
    gGpuOutputHeapWords = gpuOutputHeapWords > 0 ? (int) gpuOutputHeapWords : 0;
    gGpuOutputMaxStates = gpuOutputMaxStates > 0 ? (int) gpuOutputMaxStates : 0;
    gGpuShareCapacity = gpuShareCapacity > 0 ? nextPow2Int((int) gpuShareCapacity) : 0;
    gGpuJavaBatchBytes = gpuJavaBatchBytes >= 1048576
            ? (int) gpuJavaBatchBytes : 64 * 1024 * 1024;
    resetGenericTransferStats();
    const size_t requestedStackBytes = stackBytes < 4096 ? 4096 : (size_t) stackBytes;
    cudaError_t stackErr = cudaDeviceSetLimit(cudaLimitStackSize, requestedStackBytes);
    if (stackErr != cudaSuccess) {
        char message[192];
        snprintf(message, sizeof(message), "failed to configure generic GPU thread stack: %s",
                cudaGetErrorString(stackErr));
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        freeGenericResidentGpu();
        return;
    }
    size_t preflightFreeBytes = 0;
    size_t preflightTotalBytes = 0;
    const cudaError_t memoryInfoErr = cudaMemGetInfo(&preflightFreeBytes, &preflightTotalBytes);
    if (memoryInfoErr == cudaSuccess) {
        size_t heapBytes = 0;
        size_t rootsBytes = 0;
        size_t scanBytes = 0;
        size_t frontierBytes = 0;
        size_t visitedBytes = 0;
        const size_t requestedBytes = genericResidentAllocationBytes(&heapBytes, &rootsBytes,
                &scanBytes, &frontierBytes, &visitedBytes);
        size_t modeBytes = 0;
        if (gGenericGpuSegmented) {
            const size_t acceptedRoots = (size_t) gGenericMaxStates * gGenericVarCount * sizeof(int);
            const size_t acceptedIndices = (size_t) gGenericMaxStates * sizeof(int);
            const size_t outputRoots = (size_t) gGpuOutputMaxStates * gGenericVarCount * sizeof(int);
            const size_t outputHeap = (size_t) gGpuOutputHeapWords * sizeof(long long);
            const size_t shareTables = (size_t) gGpuShareCapacity * 4U * sizeof(int);
            modeBytes = acceptedRoots + acceptedIndices + sizeof(int)
                    + 2U * (2U * outputRoots + outputHeap + 4U * sizeof(int)
                            + 2U * sizeof(unsigned long long))
                    + shareTables + 2U * sizeof(unsigned long long);
        }
        const size_t totalRequestedBytes = requestedBytes + modeBytes;
        if (totalRequestedBytes > preflightFreeBytes) {
            genericResidentThrowPreflightError(env, totalRequestedBytes, preflightFreeBytes, preflightTotalBytes,
                    heapBytes, rootsBytes, scanBytes, frontierBytes, visitedBytes);
            freeGenericResidentGpu();
            return;
        }
    }

    cudaError_t allocationErr = cudaSuccess;
    const char* allocationName = "heap";
    if ((allocationErr = cudaMalloc(&gGenericHeap, (size_t) gGenericHeapWords * sizeof(long long))) != cudaSuccess) goto allocation_failed;
    allocationName = "compact heap";
    if (gGenericGcEnabled && (allocationErr = cudaMalloc(&gGenericCompactHeap, (size_t) gGenericHeapWords * sizeof(long long))) != cudaSuccess) goto allocation_failed;
    allocationName = "state roots";
    if ((allocationErr = cudaMalloc(&gGenericStateRoots, (size_t) gGenericMaxStates * gGenericVarCount * sizeof(int))) != cudaSuccess) goto allocation_failed;
    allocationName = "next state roots";
    if ((allocationErr = cudaMalloc(&gGenericNextStateRoots, (size_t) gGenericMaxStates * gGenericVarCount * sizeof(int))) != cudaSuccess) goto allocation_failed;
    allocationName = "compact scan sizes";
    if ((allocationErr = cudaMalloc(&gGenericCompactSizes, (size_t) gGenericMaxStates * sizeof(int))) != cudaSuccess) goto allocation_failed;
    allocationName = "compact scan offsets";
    if ((allocationErr = cudaMalloc(&gGenericCompactOffsets, (size_t) gGenericMaxStates * sizeof(int))) != cudaSuccess) goto allocation_failed;
    allocationName = "current frontier";
    if ((allocationErr = cudaMalloc(&gGenericCurrent, (size_t) gGenericMaxStates * sizeof(int))) != cudaSuccess) goto allocation_failed;
    allocationName = "next frontier";
    if ((allocationErr = cudaMalloc(&gGenericNext, (size_t) gGenericMaxStates * sizeof(int))) != cudaSuccess) goto allocation_failed;
    allocationName = "visited fingerprint table";
    if ((allocationErr = cudaMalloc(&gGenericVisited, (size_t) gGenericHashCapacity * sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
    allocationName = "frontier counters";
    if ((allocationErr = cudaMalloc(&gGenericCurrentSize, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMalloc(&gGenericNextSize, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMalloc(&gGenericStateCount, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMalloc(&gGenericOverflow, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMalloc(&gGenericStatus, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMalloc(&gGenericHeapTop, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if (gGenericGcEnabled && (allocationErr = cudaMalloc(&gGenericCompactHeapTop, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMalloc(&gGenericGenerated, sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
    if (gGenericGpuSegmented) {
        allocationName = "GPU-segmented streams";
        if ((allocationErr = cudaStreamCreateWithFlags(&gGpuComputeStream, cudaStreamNonBlocking)) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaStreamCreateWithFlags(&gGpuCopyStream, cudaStreamNonBlocking)) != cudaSuccess) goto allocation_failed;
        allocationName = "GPU-segmented accepted staging";
        if ((allocationErr = cudaMalloc(&gGpuAcceptedRoots,
                (size_t) gGenericMaxStates * gGenericVarCount * sizeof(int))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMalloc(&gGpuAcceptedIndices,
                (size_t) gGenericMaxStates * sizeof(int))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMalloc(&gGpuAcceptedCount, sizeof(int))) != cudaSuccess) goto allocation_failed;
        allocationName = "GPU-segmented output buffers";
        for (int i = 0; i < 2; i++) {
            const size_t packetBytes = (size_t) gGpuOutputHeapWords * sizeof(long long)
                    + (size_t) gGpuOutputMaxStates * gGenericVarCount * sizeof(int);
            if ((allocationErr = cudaMalloc(&gGpuOutputHeap[i],
                    packetBytes)) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaMalloc(&gGpuOutputRoots[i],
                    (size_t) gGpuOutputMaxStates * gGenericVarCount * sizeof(int))) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaMalloc(&gGpuOutputControl[i], 4U * sizeof(int))) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaHostAlloc((void**) &gGpuPinnedHeap[i],
                    packetBytes, cudaHostAllocDefault)) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaHostAlloc((void**) &gGpuPinnedControl[i],
                    4U * sizeof(int), cudaHostAllocDefault)) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaMalloc(&gGpuOutputShareHits[i],
                    sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaMalloc(&gGpuOutputShareMisses[i],
                    sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaEventCreateWithFlags(&gGpuComputeDone[i], cudaEventDisableTiming)) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaEventCreateWithFlags(&gGpuCopyDone[i], cudaEventDisableTiming)) != cudaSuccess) goto allocation_failed;
            if ((allocationErr = cudaEventRecord(gGpuCopyDone[i], gGpuCopyStream)) != cudaSuccess) goto allocation_failed;
        }
        allocationName = "GPU-segmented sharing table";
        if ((allocationErr = cudaMalloc(&gGpuShareKeys,
                (size_t) gGpuShareCapacity * sizeof(int))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMalloc(&gGpuShareValues,
                (size_t) gGpuShareCapacity * sizeof(int))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMalloc(&gGpuShareSizes,
                (size_t) gGpuShareCapacity * sizeof(int))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMalloc(&gGpuShareOffsets,
                (size_t) gGpuShareCapacity * sizeof(int))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMalloc(&gGpuShareHits, sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMalloc(&gGpuShareMisses, sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
    }
    if ((allocationErr = cudaMemset(gGenericHeap, 0, (size_t) gGenericHeapWords * sizeof(long long))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericStateRoots, 0, (size_t) gGenericMaxStates * gGenericVarCount * sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericVisited, 0, (size_t) gGenericHashCapacity * sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericCurrentSize, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericNextSize, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericStateCount, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericOverflow, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericStatus, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericHeapTop, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if (gGenericGcEnabled && (allocationErr = cudaMemset(gGenericCompactHeapTop, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
    if ((allocationErr = cudaMemset(gGenericGenerated, 0, sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
    if (gGenericGpuSegmented) {
        if ((allocationErr = cudaMemset(gGpuAcceptedCount, 0, sizeof(int))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMemset(gGpuShareHits, 0, sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
        if ((allocationErr = cudaMemset(gGpuShareMisses, 0, sizeof(unsigned long long))) != cudaSuccess) goto allocation_failed;
    }
    gGenericHostCurrentSize = 0;
    gGenericDepth = 0;
    gGenericPeakHeapWords = 0;
    gGenericLayersSinceGc = 0;
    gGenericLastLayerAllocatedWords = 0;
    gGenericGcCount = 0ULL;
    gGenericGcSkippedCount = 0ULL;
    gGenericSegmented = frontierMode != 0;
    gGenericHostFrontierSize = 0;
    gGenericSegmentSourceTop = 0;
    return;

allocation_failed:
    size_t freeBytes = 0;
    size_t totalBytes = 0;
    const int failedMaxStates = gGenericMaxStates;
    const int failedVarCount = gGenericVarCount;
    const int failedHeapWords = gGenericHeapWords;
    const int failedHashCapacity = gGenericHashCapacity;
    cudaMemGetInfo(&freeBytes, &totalBytes);
    freeGenericResidentGpu();
    char message[256];
    snprintf(message, sizeof(message),
            "generic GPU resident initialization failed while allocating %s: %s (maxStates=%d, varCount=%d, heapWords=%d, hashCapacity=%d, CUDA free=%zuMiB/%zuMiB)",
            allocationName, cudaGetErrorString(allocationErr), failedMaxStates, failedVarCount, failedHeapWords,
            failedHashCapacity, freeBytes / (1024 * 1024), totalBytes / (1024 * 1024));
    jclass errorClass = env->FindClass("java/lang/RuntimeException");
    if (errorClass != NULL) env->ThrowNew(errorClass, message);
}

extern "C"
JNIEXPORT void JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierLoadInitial(JNIEnv* env, jclass,
        jlongArray heapArray, jlongArray rootsArray) {
    if (gGenericHeap == NULL || gGenericStateRoots == NULL) return;
    const int heapLen = env->GetArrayLength(heapArray);
    const int rootsLen = env->GetArrayLength(rootsArray);
    if (heapLen <= 0 || heapLen > gGenericHeapWords || rootsLen != gGenericVarCount) return;

    jlong* hHeapJ = env->GetLongArrayElements(heapArray, NULL);
    jlong* hRootsJ = env->GetLongArrayElements(rootsArray, NULL);
    long long* dRoots = NULL;
    cudaMemcpy(gGenericHeap, hHeapJ, (size_t) heapLen * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMalloc(&dRoots, (size_t) rootsLen * sizeof(long long));
    cudaMemcpy(dRoots, hRootsJ, (size_t) rootsLen * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemset(gGenericCurrentSize, 0, sizeof(int));
    cudaMemset(gGenericNextSize, 0, sizeof(int));
    cudaMemset(gGenericStateCount, 0, sizeof(int));
    cudaMemset(gGenericOverflow, 0, sizeof(int));
    cudaMemset(gGenericStatus, 0, sizeof(int));
    cudaMemset(gGenericHeapTop, 0, sizeof(int));
    cudaMemset(gGenericGenerated, 0, sizeof(unsigned long long));
    genericResidentLoadInitialKernel<<<1, 1>>>(gGenericHeap, dRoots, rootsLen, gGenericStateRoots, gGenericCurrent,
            gGenericVisited, gGenericCurrentSize, gGenericStateCount, gGenericOverflow,
            gGenericHashCapacity, gGenericMaxStates);
    cudaError_t loadLaunchErr = cudaGetLastError();
    cudaError_t loadSyncErr = cudaDeviceSynchronize();
    if (loadLaunchErr != cudaSuccess || loadSyncErr != cudaSuccess) {
        int status = 1000 + (loadLaunchErr != cudaSuccess ? (int) loadLaunchErr : (int) loadSyncErr);
        cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(&gGenericHostCurrentSize, gGenericCurrentSize, sizeof(int), cudaMemcpyDeviceToHost);
    gGenericBaseHeapWords = heapLen;
    gGenericPeakHeapWords = heapLen;
    if (gGenericHeapTop) cudaMemcpy(gGenericHeapTop, &heapLen, sizeof(int), cudaMemcpyHostToDevice);
    gGenericDepth = 0;
    cudaFree(dRoots);
    env->ReleaseLongArrayElements(heapArray, hHeapJ, JNI_ABORT);
    env->ReleaseLongArrayElements(rootsArray, hRootsJ, JNI_ABORT);
    gGenericHostFrontierSize = gGenericHostCurrentSize;
    if (gGenericGpuSegmented) gGpuSegmentedDistinctCount = (unsigned long long) gGenericHostCurrentSize;
}

extern "C"
JNIEXPORT void JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierLoadSegment(JNIEnv* env, jclass,
        jlongArray heapArray, jlongArray rootsArray, jint fixedPrefixWordsArg) {
    if (!gGenericSegmented || gGenericHeap == NULL || gGenericStateRoots == NULL) {
        jclass errorClass = env->FindClass("java/lang/IllegalStateException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "generic GPU host-segmented frontier is not initialized");
        return;
    }
    const int heapLen = env->GetArrayLength(heapArray);
    const int rootsLen = env->GetArrayLength(rootsArray);
    const int fixedPrefixWords = (int) fixedPrefixWordsArg;
    if (heapLen <= 0 || fixedPrefixWords < gGenericBaseHeapWords
            || fixedPrefixWords + heapLen > gGenericHeapWords
            || rootsLen <= 0 || rootsLen % gGenericVarCount != 0) {
        jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "invalid generic GPU frontier segment shape");
        return;
    }
    const int stateCount = rootsLen / gGenericVarCount;
    if (stateCount > gGenericMaxStates) {
        jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "generic GPU frontier input segment exceeds segment capacity");
        return;
    }
    jlong* hHeap = env->GetLongArrayElements(heapArray, NULL);
    jlong* hRoots = env->GetLongArrayElements(rootsArray, NULL);
    if (hHeap == NULL || hRoots == NULL) {
        if (hHeap) env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
        if (hRoots) env->ReleaseLongArrayElements(rootsArray, hRoots, JNI_ABORT);
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "unable to access generic GPU frontier segment");
        return;
    }
    int* hostIndices = (int*) malloc((size_t) stateCount * sizeof(int));
    int* hostRoots32 = (int*) malloc((size_t) rootsLen * sizeof(int));
    if (hostIndices == NULL || hostRoots32 == NULL) {
        if (hostIndices) free(hostIndices);
        if (hostRoots32) free(hostRoots32);
        env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
        env->ReleaseLongArrayElements(rootsArray, hRoots, JNI_ABORT);
        jclass errorClass = env->FindClass("java/lang/OutOfMemoryError");
        if (errorClass != NULL) env->ThrowNew(errorClass, "unable to allocate generic GPU frontier segment indexes");
        return;
    }
    for (int i = 0; i < stateCount; i++) hostIndices[i] = i;
    bool rootsInRange = true;
    for (int i = 0; i < rootsLen; i++) {
        if (hRoots[i] < 0 || hRoots[i] > 2147483647LL) {
            rootsInRange = false;
            break;
        }
        hostRoots32[i] = (int) hRoots[i];
    }
    if (!rootsInRange) {
        free(hostIndices);
        free(hostRoots32);
        env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
        env->ReleaseLongArrayElements(rootsArray, hRoots, JNI_ABORT);
        jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "generic GPU frontier root offset exceeds 32-bit heap address range");
        return;
    }
    cudaError_t err = cudaMemcpy(gGenericHeap + fixedPrefixWords, hHeap,
            (size_t) heapLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(gGenericStateRoots, hostRoots32,
            (size_t) rootsLen * sizeof(int), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemcpy(gGenericCurrent, hostIndices,
            (size_t) stateCount * sizeof(int), cudaMemcpyHostToDevice);
    const int sourceTop = fixedPrefixWords + heapLen;
    if (err == cudaSuccess) err = cudaMemcpy(gGenericHeapTop, &sourceTop, sizeof(int), cudaMemcpyHostToDevice);
    if (err == cudaSuccess) err = cudaMemset(gGenericNextSize, 0, sizeof(int));
    if (err == cudaSuccess) err = cudaMemset(gGenericOverflow, 0, sizeof(int));
    if (err == cudaSuccess) err = cudaMemset(gGenericStatus, 0, sizeof(int));
    if (err != cudaSuccess) {
        char message[192];
        snprintf(message, sizeof(message), "generic GPU frontier segment upload failed: %s", cudaGetErrorString(err));
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
    }
    gGenericHostCurrentSize = stateCount;
    gGenericHostFrontierSize = stateCount;
    gGenericSegmentSourceTop = sourceTop;
    free(hostIndices);
    free(hostRoots32);
    env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
    env->ReleaseLongArrayElements(rootsArray, hRoots, JNI_ABORT);
}

__global__ void gpuSegmentFillIndicesKernel(int* indices, int count) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) indices[index] = index;
}

__device__ int gpuSegmentClaimShareSlot(int* keys, int capacity, int sourceOffset,
        bool* owner) {
    const unsigned long long hash = gpuTlaMix64((unsigned long long) (unsigned int) sourceOffset);
    for (int probe = 0; probe < capacity; probe++) {
        const int slot = (int) ((hash + (unsigned long long) probe)
                & (unsigned long long) (capacity - 1));
        int key = atomicAdd(keys + slot, 0);
        if (key == sourceOffset) {
            *owner = false;
            return slot;
        }
        if (key == -1) {
            key = atomicCAS(keys + slot, -1, sourceOffset);
            if (key == -1) {
                *owner = true;
                return slot;
            }
            if (key == sourceOffset) {
                *owner = false;
                return slot;
            }
        }
    }
    return -1;
}

__device__ __noinline__ bool gpuSegmentMarkValue(const long long* sourceHeap,
        int sourceTop, int fixedPrefixWords, int sourceOffset, int* shareKeys,
        int shareCapacity, int depth, int* statusOut, unsigned long long* shareHits,
        unsigned long long* shareMisses) {
    if (sourceOffset < 0 || sourceOffset >= sourceTop || depth > 128) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return false;
    }
    if (sourceOffset < fixedPrefixWords) return true;
    bool owner = false;
    const int slot = gpuSegmentClaimShareSlot(shareKeys, shareCapacity, sourceOffset, &owner);
    if (slot < 0) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_SHARE_OVERFLOW);
        return false;
    }
    if (!owner) {
        atomicAdd(shareHits, 1ULL);
        return true;
    }
    atomicAdd(shareMisses, 1ULL);

    const long long header = sourceHeap[sourceOffset];
    const int tag = gpuTlaTag(header);
    const int arity = gpuTlaArity(header);
    const int words = genericResidentObjectWords(header);
    if (words <= 0 || sourceOffset > sourceTop - words) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return false;
    }
    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
    case GPU_TLA_TAG_STRING:
    case GPU_TLA_TAG_OPAQUE:
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL:
        return true;
    case GPU_TLA_TAG_TUPLE:
    case GPU_TLA_TAG_SET_ENUM:
        for (int i = 0; i < arity; i++) {
            if (!gpuSegmentMarkValue(sourceHeap, sourceTop, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + i], shareKeys, shareCapacity,
                    depth + 1, statusOut, shareHits, shareMisses)) return false;
        }
        return true;
    case GPU_TLA_TAG_FCN_RCD: {
        const int first = gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL ? 3 : 1;
        const int count = gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL ? arity : 2 * arity;
        if (gpuTlaAux(header) != GPU_TLA_FCN_DOMAIN_INTERVAL
                && gpuTlaAux(header) != GPU_TLA_FCN_DOMAIN_EXPLICIT) {
            atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
            return false;
        }
        for (int i = 0; i < count; i++) {
            if (!gpuSegmentMarkValue(sourceHeap, sourceTop, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + first + i], shareKeys, shareCapacity,
                    depth + 1, statusOut, shareHits, shareMisses)) return false;
        }
        return true;
    }
    case GPU_TLA_TAG_RECORD:
        for (int i = 0; i < arity; i++) {
            if (!gpuSegmentMarkValue(sourceHeap, sourceTop, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + arity + i], shareKeys,
                    shareCapacity, depth + 1, statusOut, shareHits, shareMisses)) return false;
        }
        return true;
    default:
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return false;
    }
}

__global__ void gpuSegmentInitOutputKernel(int* outputControl, int stateCount,
        int fixedPrefixWords, unsigned long long* shareHits,
        unsigned long long* shareMisses) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    outputControl[0] = GPU_TLA_SUCC_STATUS_OK;
    outputControl[1] = stateCount;
    outputControl[2] = 0;
    outputControl[3] = fixedPrefixWords;
    *shareHits = 0ULL;
    *shareMisses = 0ULL;
}

__global__ void gpuSegmentMarkAcceptedKernel(const long long* sourceHeap, int sourceTop,
        int fixedPrefixWords, const int* acceptedRoots, int acceptedFirst,
        int stateCount, int varCount, int* shareKeys, int shareCapacity,
        int* statusOut, unsigned long long* shareHits,
        unsigned long long* shareMisses) {
    const int root = blockIdx.x * blockDim.x + threadIdx.x;
    const int rootCount = stateCount * varCount;
    if (root >= rootCount) return;
    const int sourceOffset = acceptedRoots[(size_t) acceptedFirst * varCount + root];
    gpuSegmentMarkValue(sourceHeap, sourceTop, fixedPrefixWords, sourceOffset,
            shareKeys, shareCapacity, 0, statusOut, shareHits, shareMisses);
}

__global__ void gpuSegmentMeasureMarkedKernel(const long long* sourceHeap, int sourceTop,
        const int* shareKeys, int shareCapacity, int* sizes, int* statusOut) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= shareCapacity) return;
    const int sourceOffset = shareKeys[slot];
    if (sourceOffset == -1) {
        sizes[slot] = 0;
        return;
    }
    if (sourceOffset < 0 || sourceOffset >= sourceTop) {
        sizes[slot] = 0;
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }
    const int words = genericResidentObjectWords(sourceHeap[sourceOffset]);
    if (words <= 0 || sourceOffset > sourceTop - words) {
        sizes[slot] = 0;
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }
    sizes[slot] = words;
}

__global__ void gpuSegmentAssignOffsetsKernel(const int* shareKeys, int shareCapacity,
        const int* sizes, const int* offsets, int fixedPrefixWords, int outputCapacity,
        int* shareValues, int* outputTop, int* statusOut) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= shareCapacity || shareKeys[slot] == -1) return;
    const int offset = offsets[slot];
    const int words = sizes[slot];
    if (offset < 0 || words <= 0 || offset > outputCapacity - words) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
        return;
    }
    shareValues[slot] = fixedPrefixWords + offset;
    atomicMax(outputTop, offset + words);
}

__device__ bool gpuSegmentFindMappedOffset(const int* shareKeys,
        const int* shareValues, int shareCapacity, int fixedPrefixWords,
        int sourceOffset, int* targetOffset) {
    if (sourceOffset < fixedPrefixWords) {
        *targetOffset = sourceOffset;
        return true;
    }
    const unsigned long long hash = gpuTlaMix64((unsigned long long) (unsigned int) sourceOffset);
    for (int probe = 0; probe < shareCapacity; probe++) {
        const int slot = (int) ((hash + (unsigned long long) probe)
                & (unsigned long long) (shareCapacity - 1));
        const int key = shareKeys[slot];
        if (key == sourceOffset) {
            *targetOffset = shareValues[slot];
            return *targetOffset >= fixedPrefixWords;
        }
        if (key == -1) return false;
    }
    return false;
}

__device__ bool gpuSegmentWriteMappedReference(long long* targetWord,
        const int* shareKeys, const int* shareValues, int shareCapacity,
        int fixedPrefixWords, int sourceOffset) {
    int targetOffset = -1;
    if (!gpuSegmentFindMappedOffset(shareKeys, shareValues, shareCapacity,
            fixedPrefixWords, sourceOffset, &targetOffset)) return false;
    *targetWord = targetOffset;
    return true;
}

__global__ void gpuSegmentCopyMarkedKernel(const long long* sourceHeap, int sourceTop,
        int fixedPrefixWords, const int* shareKeys, const int* shareValues,
        int shareCapacity, long long* outputHeap, int outputCapacity,
        int* statusOut) {
    const int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= shareCapacity) return;
    const int sourceOffset = shareKeys[slot];
    if (sourceOffset == -1) return;
    const int absoluteTarget = shareValues[slot];
    const int target = absoluteTarget - fixedPrefixWords;
    const long long header = sourceHeap[sourceOffset];
    const int tag = gpuTlaTag(header);
    const int arity = gpuTlaArity(header);
    const int words = genericResidentObjectWords(header);
    if (target < 0 || words <= 0 || target > outputCapacity - words
            || sourceOffset < fixedPrefixWords || sourceOffset > sourceTop - words) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW);
        return;
    }
    outputHeap[target] = header;
    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT:
        return;
    case GPU_TLA_TAG_STRING:
    case GPU_TLA_TAG_OPAQUE:
        outputHeap[target + 1] = sourceHeap[sourceOffset + 1];
        return;
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL:
        outputHeap[target + 1] = sourceHeap[sourceOffset + 1];
        outputHeap[target + 2] = sourceHeap[sourceOffset + 2];
        return;
    case GPU_TLA_TAG_TUPLE:
    case GPU_TLA_TAG_SET_ENUM:
        for (int i = 0; i < arity; i++) {
            if (!gpuSegmentWriteMappedReference(outputHeap + target + 1 + i,
                    shareKeys, shareValues, shareCapacity, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + i])) {
                atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
                return;
            }
        }
        return;
    case GPU_TLA_TAG_FCN_RCD:
        if (gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL) {
            outputHeap[target + 1] = sourceHeap[sourceOffset + 1];
            outputHeap[target + 2] = sourceHeap[sourceOffset + 2];
            for (int i = 0; i < arity; i++) {
                if (!gpuSegmentWriteMappedReference(outputHeap + target + 3 + i,
                        shareKeys, shareValues, shareCapacity, fixedPrefixWords,
                        (int) sourceHeap[sourceOffset + 3 + i])) {
                    atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
                    return;
                }
            }
            return;
        }
        if (gpuTlaAux(header) != GPU_TLA_FCN_DOMAIN_EXPLICIT) {
            atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
            return;
        }
        for (int i = 0; i < 2 * arity; i++) {
            if (!gpuSegmentWriteMappedReference(outputHeap + target + 1 + i,
                    shareKeys, shareValues, shareCapacity, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + i])) {
                atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
                return;
            }
        }
        return;
    case GPU_TLA_TAG_RECORD:
        for (int i = 0; i < arity; i++) {
            outputHeap[target + 1 + i] = sourceHeap[sourceOffset + 1 + i];
        }
        for (int i = 0; i < arity; i++) {
            if (!gpuSegmentWriteMappedReference(outputHeap + target + 1 + arity + i,
                    shareKeys, shareValues, shareCapacity, fixedPrefixWords,
                    (int) sourceHeap[sourceOffset + 1 + arity + i])) {
                atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
                return;
            }
        }
        return;
    default:
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
    }
}

__global__ void gpuSegmentMapAcceptedRootsKernel(const int* acceptedRoots,
        int acceptedFirst, int stateCount, int varCount, const int* shareKeys,
        const int* shareValues, int shareCapacity, int fixedPrefixWords,
        int* outputRoots, int* statusOut) {
    const int root = blockIdx.x * blockDim.x + threadIdx.x;
    const int rootCount = stateCount * varCount;
    if (root >= rootCount) return;
    const int sourceOffset = acceptedRoots[(size_t) acceptedFirst * varCount + root];
    int targetOffset = -1;
    if (!gpuSegmentFindMappedOffset(shareKeys, shareValues, shareCapacity,
            fixedPrefixWords, sourceOffset, &targetOffset)) {
        atomicCAS(statusOut, GPU_TLA_SUCC_STATUS_OK, GPU_TLA_SUCC_STATUS_BAD_PROGRAM);
        return;
    }
    outputRoots[root] = targetOffset;
}

__global__ void gpuSegmentPackRootsKernel(long long* outputPacket, int outputHeapWords,
        const int* outputRoots, int rootCount) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= rootCount) return;
    int* packetRoots = reinterpret_cast<int*>(outputPacket + outputHeapWords);
    packetRoots[index] = outputRoots[index];
}

__global__ void gpuSegmentCommitShareStatsKernel(const unsigned long long* chunkHits,
        const unsigned long long* chunkMisses, unsigned long long* totalHits,
        unsigned long long* totalMisses) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    atomicAdd(totalHits, *chunkHits);
    atomicAdd(totalMisses, *chunkMisses);
}

__global__ void gpuSegmentCollectGenerationKernel(const int* status, const int* overflow,
        const int* acceptedCount, const int* heapTop, int* outputControl) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    outputControl[0] = *status;
    outputControl[1] = *overflow;
    outputControl[2] = *acceptedCount;
    outputControl[3] = *heapTop;
}
struct GpuSegmentPendingCopy {
    bool active;
    int sequence;
    int states;
    int heapWords;
};

struct GpuSegmentHostBatch {
    std::vector<long long> heap;
    std::vector<int> roots;
    std::vector<int> descriptors;
};

static size_t gpuSegmentHostBatchBytes(const GpuSegmentHostBatch& batch) {
    return batch.heap.size() * sizeof(long long)
            + batch.roots.size() * sizeof(int)
            + batch.descriptors.size() * sizeof(int);
}

static bool gpuSegmentFlushPending(JNIEnv* env, int slot, GpuSegmentPendingCopy* pending,
        std::vector<GpuSegmentHostBatch>& batches, GpuSegmentHostBatch& currentBatch) {
    if (!pending[slot].active) return true;
    const std::chrono::steady_clock::time_point waitBegin = std::chrono::steady_clock::now();
    const cudaError_t waitErr = cudaEventSynchronize(gGpuCopyDone[slot]);
    gTransferD2HWaitNanos += genericElapsedNanos(waitBegin);
    if (waitErr != cudaSuccess) {
        char message[256];
        snprintf(message, sizeof(message), "generic GPU segmented D2H synchronization failed: %s",
                cudaGetErrorString(waitErr));
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        return false;
    }
    const size_t rootCount = (size_t) pending[slot].states * gGenericVarCount;
    const size_t appendBytes = (size_t) pending[slot].heapWords * sizeof(long long)
            + rootCount * sizeof(int) + 4U * sizeof(int);
    if (!currentBatch.descriptors.empty()
            && gpuSegmentHostBatchBytes(currentBatch) + appendBytes > (size_t) gGpuJavaBatchBytes) {
        batches.push_back(GpuSegmentHostBatch());
        batches.back().heap.swap(currentBatch.heap);
        batches.back().roots.swap(currentBatch.roots);
        batches.back().descriptors.swap(currentBatch.descriptors);
    }
    if (currentBatch.heap.size() > 2147483647U - (size_t) pending[slot].heapWords
            || currentBatch.roots.size() > 2147483647U - rootCount) {
        jclass errorClass = env->FindClass("java/lang/OutOfMemoryError");
        if (errorClass != NULL) env->ThrowNew(errorClass,
                "generic GPU segmented JNI batch exceeds Java array capacity");
        return false;
    }
    currentBatch.descriptors.push_back((int) currentBatch.heap.size());
    currentBatch.descriptors.push_back(pending[slot].heapWords);
    currentBatch.descriptors.push_back((int) currentBatch.roots.size());
    currentBatch.descriptors.push_back((int) rootCount);
    currentBatch.heap.insert(currentBatch.heap.end(), gGpuPinnedHeap[slot],
            gGpuPinnedHeap[slot] + pending[slot].heapWords);
    const int* packetRoots = reinterpret_cast<const int*>(
            gGpuPinnedHeap[slot] + pending[slot].heapWords);
    currentBatch.roots.insert(currentBatch.roots.end(), packetRoots, packetRoots + rootCount);
    pending[slot].active = false;
    return true;
}

static jobjectArray gpuSegmentBuildJavaBatches(JNIEnv* env,
        std::vector<GpuSegmentHostBatch>& batches) {
    jclass objectClass = env->FindClass("java/lang/Object");
    if (objectClass == NULL) return NULL;
    if (batches.size() > 715827882U) {
        jclass errorClass = env->FindClass("java/lang/OutOfMemoryError");
        if (errorClass != NULL) env->ThrowNew(errorClass,
                "generic GPU segmented result has too many JNI batches");
        return NULL;
    }
    jobjectArray result = env->NewObjectArray((jsize) (batches.size() * 3U), objectClass, NULL);
    if (result == NULL) return NULL;
    for (size_t batchIndex = 0; batchIndex < batches.size(); batchIndex++) {
        GpuSegmentHostBatch& batch = batches[batchIndex];
        jlongArray heapArray = env->NewLongArray((jsize) batch.heap.size());
        jintArray rootsArray = env->NewIntArray((jsize) batch.roots.size());
        jintArray descriptorsArray = env->NewIntArray((jsize) batch.descriptors.size());
        if (heapArray == NULL || rootsArray == NULL || descriptorsArray == NULL) return NULL;
        if (!batch.heap.empty()) env->SetLongArrayRegion(heapArray, 0, (jsize) batch.heap.size(),
                (const jlong*) batch.heap.data());
        if (!batch.roots.empty()) env->SetIntArrayRegion(rootsArray, 0, (jsize) batch.roots.size(),
                (const jint*) batch.roots.data());
        if (!batch.descriptors.empty()) env->SetIntArrayRegion(descriptorsArray, 0,
                (jsize) batch.descriptors.size(), (const jint*) batch.descriptors.data());
        env->SetObjectArrayElement(result, (jsize) (3U * batchIndex), heapArray);
        env->SetObjectArrayElement(result, (jsize) (3U * batchIndex + 1U), rootsArray);
        env->SetObjectArrayElement(result, (jsize) (3U * batchIndex + 2U), descriptorsArray);
        env->DeleteLocalRef(heapArray);
        env->DeleteLocalRef(rootsArray);
        env->DeleteLocalRef(descriptorsArray);
        if (env->ExceptionCheck()) return NULL;
    }
    return result;
}


static bool gpuSegmentProcessInput(JNIEnv* env, const jlong* hostDynamic, int dynamicWords,
        const jint* hostRoots, int rootsLen, int fixedPrefixWords,
        int branchLen, int opLen, int exprProgramLen,
        GpuSegmentPendingCopy* pending, std::vector<GpuSegmentHostBatch>& batches,
        GpuSegmentHostBatch& currentBatch, int* sequence) {
    const int stateCount = rootsLen / gGenericVarCount;
    cudaError_t err = cudaSuccess;
    const std::chrono::steady_clock::time_point h2dBegin = std::chrono::steady_clock::now();
    if (dynamicWords > 0) {
        err = cudaMemcpyAsync(gGenericHeap + fixedPrefixWords, hostDynamic,
                (size_t) dynamicWords * sizeof(long long), cudaMemcpyHostToDevice,
                gGpuComputeStream);
        gTransferH2DBytes += (unsigned long long) dynamicWords * sizeof(long long);
        gTransferH2DCalls++;
    }
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(gGenericStateRoots, hostRoots,
                (size_t) rootsLen * sizeof(int), cudaMemcpyHostToDevice,
                gGpuComputeStream);
        gTransferH2DBytes += (unsigned long long) rootsLen * sizeof(int);
        gTransferH2DCalls++;
    }
    if (err == cudaSuccess) err = cudaStreamSynchronize(gGpuComputeStream);
    gTransferH2DWaitNanos += genericElapsedNanos(h2dBegin);
    if (err != cudaSuccess) {
        char message[256];
        snprintf(message, sizeof(message), "generic GPU segmented upload failed: %s",
                cudaGetErrorString(err));
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        return false;
    }

    const int sourceTop = fixedPrefixWords + dynamicWords;
    const int threads = 128;
    gpuSegmentFillIndicesKernel<<<(stateCount + threads - 1) / threads, threads, 0,
            gGpuComputeStream>>>(gGenericCurrent, stateCount);
    cudaMemsetAsync(gGpuAcceptedCount, 0, sizeof(int), gGpuComputeStream);
    cudaMemsetAsync(gGenericStateCount, 0, sizeof(int), gGpuComputeStream);
    cudaMemsetAsync(gGenericOverflow, 0, sizeof(int), gGpuComputeStream);
    cudaMemsetAsync(gGenericStatus, 0, sizeof(int), gGpuComputeStream);
    cudaMemcpyAsync(gGenericHeapTop, &sourceTop, sizeof(int), cudaMemcpyHostToDevice,
            gGpuComputeStream);
    for (int begin = 0; begin < stateCount; begin += gGenericBatchStates) {
        const int count = stateCount - begin < gGenericBatchStates
                ? stateCount - begin : gGenericBatchStates;
        genericResidentStepKernel<<<(count + threads - 1) / threads, threads, 0,
                gGpuComputeStream>>>(
                gGenericResidentBranch, branchLen, gGenericResidentOp, opLen,
                gGenericResidentExprProgram, gGenericResidentExprStarts, gGenericResidentExprLens,
                gGenericHeap, gGenericStateRoots, gGpuAcceptedRoots, gGenericCurrent + begin, count,
                gGpuAcceptedIndices, gGpuAcceptedCount, gGenericStateCount, gGenericVisited,
                gGenericOverflow, gGenericStatus, gGenericHeapTop, gGenericHeapWords,
                gGenericGenerated, gGenericHashCapacity, gGenericMaxStates, gGenericVarCount);
    }
    gpuSegmentCollectGenerationKernel<<<1, 1, 0, gGpuComputeStream>>>(gGenericStatus,
            gGenericOverflow, gGpuAcceptedCount, gGenericHeapTop, gGpuOutputControl[0]);
    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaMemcpyAsync(gGpuPinnedControl[0], gGpuOutputControl[0], 4U * sizeof(int),
                cudaMemcpyDeviceToHost, gGpuComputeStream);
    }
    gTransferD2HControlBytes += 4U * sizeof(int);
    gTransferD2HCalls++;
    if (err == cudaSuccess) err = cudaStreamSynchronize(gGpuComputeStream);
    int status = err == cudaSuccess ? gGpuPinnedControl[0][0] : 1000 + (int) err;
    const int overflow = err == cudaSuccess ? gGpuPinnedControl[0][1] : 1;
    const int acceptedCount = err == cudaSuccess ? gGpuPinnedControl[0][2] : -1;
    const int generatedTop = err == cudaSuccess ? gGpuPinnedControl[0][3] : sourceTop;
    if (generatedTop > gGenericPeakHeapWords) gGenericPeakHeapWords = generatedTop;
    if (status != GPU_TLA_SUCC_STATUS_OK || overflow != 0
            || acceptedCount < 0 || acceptedCount > gGenericMaxStates) {
        if (status == GPU_TLA_SUCC_STATUS_OK) status = GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED;
        cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        char message[256];
        snprintf(message, sizeof(message),
                "generic GPU segmented successor staging failed (status=%d, accepted=%d, capacity=%d)",
                status, acceptedCount, gGenericMaxStates);
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        return false;
    }
    gGpuSegmentedDistinctCount += (unsigned long long) acceptedCount;

    int acceptedFirst = 0;
    while (acceptedFirst < acceptedCount) {
        const int slot = *sequence & 1;
        if (!gpuSegmentFlushPending(env, slot, pending, batches, currentBatch)) return false;
        int candidate = acceptedCount - acceptedFirst;
        if (candidate > gGpuOutputMaxStates) candidate = gGpuOutputMaxStates;
        int control[4] = { GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW, 0, 0, fixedPrefixWords };
        while (true) {
            cudaMemsetAsync(gGpuShareKeys, 0xff,
                    (size_t) gGpuShareCapacity * sizeof(int), gGpuComputeStream);
            gpuSegmentInitOutputKernel<<<1, 1, 0, gGpuComputeStream>>>(
                    gGpuOutputControl[slot], candidate, fixedPrefixWords,
                    gGpuOutputShareHits[slot], gGpuOutputShareMisses[slot]);
            const int rootCount = candidate * gGenericVarCount;
            gpuSegmentMarkAcceptedKernel<<<(rootCount + threads - 1) / threads,
                    threads, 0, gGpuComputeStream>>>(
                    gGenericHeap, generatedTop, fixedPrefixWords, gGpuAcceptedRoots,
                    acceptedFirst, candidate, gGenericVarCount, gGpuShareKeys,
                    gGpuShareCapacity, gGpuOutputControl[slot],
                    gGpuOutputShareHits[slot], gGpuOutputShareMisses[slot]);
            const int shareBlocks = (gGpuShareCapacity + threads - 1) / threads;
            gpuSegmentMeasureMarkedKernel<<<shareBlocks, threads, 0, gGpuComputeStream>>>(
                    gGenericHeap, generatedTop, gGpuShareKeys, gGpuShareCapacity,
                    gGpuShareSizes, gGpuOutputControl[slot]);
            try {
                thrust::exclusive_scan(thrust::cuda::par.on(gGpuComputeStream),
                        thrust::device_pointer_cast(gGpuShareSizes),
                        thrust::device_pointer_cast(gGpuShareSizes + gGpuShareCapacity),
                        thrust::device_pointer_cast(gGpuShareOffsets));
            } catch (...) {
                const int scanStatus = GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW;
                cudaMemcpyAsync(gGpuOutputControl[slot], &scanStatus, sizeof(int),
                        cudaMemcpyHostToDevice, gGpuComputeStream);
            }
            gpuSegmentAssignOffsetsKernel<<<shareBlocks, threads, 0, gGpuComputeStream>>>(
                    gGpuShareKeys, gGpuShareCapacity, gGpuShareSizes, gGpuShareOffsets,
                    fixedPrefixWords, gGpuOutputHeapWords, gGpuShareValues,
                    gGpuOutputControl[slot] + 2, gGpuOutputControl[slot]);
            gpuSegmentCopyMarkedKernel<<<shareBlocks, threads, 0, gGpuComputeStream>>>(
                    gGenericHeap, generatedTop, fixedPrefixWords, gGpuShareKeys,
                    gGpuShareValues, gGpuShareCapacity, gGpuOutputHeap[slot],
                    gGpuOutputHeapWords, gGpuOutputControl[slot]);
            gpuSegmentMapAcceptedRootsKernel<<<(rootCount + threads - 1) / threads,
                    threads, 0, gGpuComputeStream>>>(gGpuAcceptedRoots, acceptedFirst,
                    candidate, gGenericVarCount, gGpuShareKeys, gGpuShareValues,
                    gGpuShareCapacity, fixedPrefixWords, gGpuOutputRoots[slot],
                    gGpuOutputControl[slot]);
            err = cudaMemcpyAsync(gGpuPinnedControl[slot], gGpuOutputControl[slot],
                    4U * sizeof(int), cudaMemcpyDeviceToHost, gGpuComputeStream);
            const std::chrono::steady_clock::time_point controlWaitBegin =
                    std::chrono::steady_clock::now();
            if (err == cudaSuccess) err = cudaStreamSynchronize(gGpuComputeStream);
            gTransferD2HControlBytes += 4U * sizeof(int);
            gTransferD2HCalls++;
            gTransferD2HWaitNanos += genericElapsedNanos(controlWaitBegin);
            if (err != cudaSuccess) {
                jclass errorClass = env->FindClass("java/lang/RuntimeException");
                if (errorClass != NULL) env->ThrowNew(errorClass,
                        "generic GPU segmented parallel compaction failed");
                return false;
            }
            for (int i = 0; i < 4; i++) control[i] = gGpuPinnedControl[slot][i];
            if (control[0] == GPU_TLA_SUCC_STATUS_OK) break;
            if ((control[0] != GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW
                    && control[0] != GPU_TLA_SUCC_STATUS_SHARE_OVERFLOW) || candidate <= 1) {
                cudaMemcpy(gGenericStatus, &control[0], sizeof(int), cudaMemcpyHostToDevice);
                jclass errorClass = env->FindClass("java/lang/IllegalStateException");
                if (errorClass != NULL) env->ThrowNew(errorClass,
                        "one accepted state exceeds GPU-segmented output heap or sharing capacity");
                return false;
            }
            candidate = (candidate + 1) / 2;
        }

        const int outputWords = control[2];
        const int rootCount = candidate * gGenericVarCount;
        gpuSegmentCommitShareStatsKernel<<<1, 1, 0, gGpuComputeStream>>>(
                gGpuOutputShareHits[slot], gGpuOutputShareMisses[slot],
                gGpuShareHits, gGpuShareMisses);
        gpuSegmentPackRootsKernel<<<(rootCount + threads - 1) / threads,
                threads, 0, gGpuComputeStream>>>(gGpuOutputHeap[slot], outputWords,
                gGpuOutputRoots[slot], rootCount);
        cudaEventRecord(gGpuComputeDone[slot], gGpuComputeStream);
        cudaStreamWaitEvent(gGpuCopyStream, gGpuComputeDone[slot], 0);
        const size_t payloadBytes = (size_t) outputWords * sizeof(long long)
                + (size_t) rootCount * sizeof(int);
        err = cudaMemcpyAsync(gGpuPinnedHeap[slot], gGpuOutputHeap[slot], payloadBytes,
                cudaMemcpyDeviceToHost, gGpuCopyStream);
        if (err != cudaSuccess) {
            jclass errorClass = env->FindClass("java/lang/RuntimeException");
            if (errorClass != NULL) env->ThrowNew(errorClass,
                    "generic GPU segmented combined D2H launch failed");
            return false;
        }
        gTransferD2HCalls++;
        cudaEventRecord(gGpuCopyDone[slot], gGpuCopyStream);
        gTransferD2HPayloadBytes += payloadBytes;
        gTransferSegmentHeaders++;
        gTransferAcceptedStates += candidate;
        gTransferDynamicHeapWords += outputWords;
        pending[slot].active = true;
        pending[slot].sequence = *sequence;
        pending[slot].states = candidate;
        pending[slot].heapWords = outputWords;
        acceptedFirst += candidate;
        (*sequence)++;
    }
    return true;
}

extern "C"
JNIEXPORT jobjectArray JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierRunGpuSegment(JNIEnv* env, jclass,
        jlongArray dynamicHeapArray, jintArray absoluteRootsArray, jintArray segmentDescriptorsArray,
        jint fixedPrefixWordsArg, jlongArray branchArray, jlongArray opArray,
        jlongArray exprProgramArray, jlongArray exprStartsArray, jlongArray exprLensArray,
        jlongArray constantHeapArray) {
    if (!gGenericGpuSegmented || gGenericHeap == NULL || gGpuAcceptedRoots == NULL) {
        jclass errorClass = env->FindClass("java/lang/IllegalStateException");
        if (errorClass != NULL) env->ThrowNew(errorClass,
                "generic GPU segmented frontier is not initialized");
        return NULL;
    }
    const int dynamicWords = env->GetArrayLength(dynamicHeapArray);
    const int rootsLen = env->GetArrayLength(absoluteRootsArray);
    const int descriptorsLen = env->GetArrayLength(segmentDescriptorsArray);
    const int fixedPrefixWords = (int) fixedPrefixWordsArg;
    const int branchLen = env->GetArrayLength(branchArray);
    const int opLen = env->GetArrayLength(opArray);
    const int exprProgramLen = env->GetArrayLength(exprProgramArray);
    const int exprCount = env->GetArrayLength(exprStartsArray);
    const int constantLen = env->GetArrayLength(constantHeapArray);
    if (fixedPrefixWords != gGenericBaseHeapWords + constantLen || fixedPrefixWords < 0
            || dynamicWords < 0 || rootsLen <= 0 || descriptorsLen <= 0
            || descriptorsLen % 4 != 0 || branchLen <= 0 || exprProgramLen <= 0
            || exprCount != env->GetArrayLength(exprLensArray)) {
        jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
        if (errorClass != NULL) env->ThrowNew(errorClass,
                "invalid generic GPU segmented input batch shape");
        return NULL;
    }
    if (!genericResidentEnsureProgram(env, branchArray, opArray, exprProgramArray,
            exprStartsArray, exprLensArray, constantHeapArray, branchLen, opLen,
            exprProgramLen, exprCount, constantLen)) return NULL;

    jlong* hostDynamic = dynamicWords > 0
            ? env->GetLongArrayElements(dynamicHeapArray, NULL) : NULL;
    jint* hostRoots = env->GetIntArrayElements(absoluteRootsArray, NULL);
    jint* hostDescriptors = env->GetIntArrayElements(segmentDescriptorsArray, NULL);
    if ((dynamicWords > 0 && hostDynamic == NULL) || hostRoots == NULL
            || hostDescriptors == NULL) {
        if (hostDynamic != NULL) env->ReleaseLongArrayElements(dynamicHeapArray, hostDynamic, JNI_ABORT);
        if (hostRoots != NULL) env->ReleaseIntArrayElements(absoluteRootsArray, hostRoots, JNI_ABORT);
        if (hostDescriptors != NULL) env->ReleaseIntArrayElements(
                segmentDescriptorsArray, hostDescriptors, JNI_ABORT);
        return NULL;
    }

    std::vector<GpuSegmentHostBatch> batches;
    GpuSegmentHostBatch currentBatch;
    GpuSegmentPendingCopy pending[2] = { { false, -1, 0, 0 }, { false, -1, 0, 0 } };
    int sequence = 0;
    bool ok = true;
    for (int descriptor = 0; descriptor < descriptorsLen && ok; descriptor += 4) {
        const int heapOffset = hostDescriptors[descriptor];
        const int heapLength = hostDescriptors[descriptor + 1];
        const int rootsOffset = hostDescriptors[descriptor + 2];
        const int rootsLength = hostDescriptors[descriptor + 3];
        const int stateCount = rootsLength > 0 ? rootsLength / gGenericVarCount : 0;
        if (heapOffset < 0 || heapLength < 0 || heapOffset > dynamicWords - heapLength
                || rootsOffset < 0 || rootsLength <= 0 || rootsOffset > rootsLen - rootsLength
                || rootsLength % gGenericVarCount != 0 || stateCount > gGenericMaxStates
                || fixedPrefixWords > gGenericHeapWords - heapLength) {
            jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
            if (errorClass != NULL) env->ThrowNew(errorClass,
                    "invalid segment in generic GPU segmented input batch");
            ok = false;
            break;
        }
        ok = gpuSegmentProcessInput(env,
                hostDynamic == NULL ? NULL : hostDynamic + heapOffset, heapLength,
                hostRoots + rootsOffset, rootsLength, fixedPrefixWords,
                branchLen, opLen, exprProgramLen, pending, batches, currentBatch, &sequence);
    }
    while (ok && (pending[0].active || pending[1].active)) {
        const int slot = !pending[0].active ? 1 : (!pending[1].active ? 0
                : (pending[0].sequence < pending[1].sequence ? 0 : 1));
        ok = gpuSegmentFlushPending(env, slot, pending, batches, currentBatch);
    }
    if (ok && !currentBatch.descriptors.empty()) {
        batches.push_back(GpuSegmentHostBatch());
        batches.back().heap.swap(currentBatch.heap);
        batches.back().roots.swap(currentBatch.roots);
        batches.back().descriptors.swap(currentBatch.descriptors);
    }
    if (hostDynamic != NULL) env->ReleaseLongArrayElements(dynamicHeapArray, hostDynamic, JNI_ABORT);
    env->ReleaseIntArrayElements(absoluteRootsArray, hostRoots, JNI_ABORT);
    env->ReleaseIntArrayElements(segmentDescriptorsArray, hostDescriptors, JNI_ABORT);
    if (!ok) return NULL;
    gGenericHostCurrentSize = 0;
    return gpuSegmentBuildJavaBatches(env, batches);
}

static int genericHostObjectWords(long long header) {
    const int tag = gpuTlaTag(header);
    const int arity = gpuTlaArity(header);
    if (arity < 0) return -1;
    switch (tag) {
    case GPU_TLA_TAG_NULL:
    case GPU_TLA_TAG_BOOL:
    case GPU_TLA_TAG_INT: return 1;
    case GPU_TLA_TAG_STRING:
    case GPU_TLA_TAG_OPAQUE: return 2;
    case GPU_TLA_TAG_MODEL:
    case GPU_TLA_TAG_INTERVAL: return 3;
    case GPU_TLA_TAG_TUPLE:
    case GPU_TLA_TAG_SET_ENUM: return 1 + arity;
    case GPU_TLA_TAG_FCN_RCD:
        return gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL ? 3 + arity : 1 + 2 * arity;
    case GPU_TLA_TAG_RECORD: return 1 + 2 * arity;
    default: return -1;
    }
}

static bool genericHostRebaseHeap(std::vector<long long>& heap, int base) {
    size_t offset = 0;
    while (offset < heap.size()) {
        const long long header = heap[offset];
        const int tag = gpuTlaTag(header);
        const int arity = gpuTlaArity(header);
        const int words = genericHostObjectWords(header);
        if (words <= 0 || offset + (size_t) words > heap.size()) return false;
        if (tag == GPU_TLA_TAG_TUPLE || tag == GPU_TLA_TAG_SET_ENUM) {
            for (int i = 0; i < arity; i++) heap[offset + 1 + i] += base;
        } else if (tag == GPU_TLA_TAG_FCN_RCD) {
            const int first = gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL ? 3 : 1;
            const int count = gpuTlaAux(header) == GPU_TLA_FCN_DOMAIN_INTERVAL ? arity : 2 * arity;
            for (int i = 0; i < count; i++) heap[offset + first + i] += base;
        } else if (tag == GPU_TLA_TAG_RECORD) {
            for (int i = 0; i < arity; i++) heap[offset + 1 + arity + i] += base;
        }
        offset += (size_t) words;
    }
    return true;
}

static jlongArray genericSegmentPacket(JNIEnv* env, int status, int count, int heapWords, int varCount,
        const std::vector<long long>& heap, const std::vector<long long>& roots) {
    const size_t total = 5 + heap.size() + roots.size();
    if (total > 2147483647U) {
        jclass errorClass = env->FindClass("java/lang/OutOfMemoryError");
        if (errorClass != NULL) env->ThrowNew(errorClass, "generic GPU frontier segment packet exceeds Java array capacity");
        return NULL;
    }
    jlongArray result = env->NewLongArray((jsize) total);
    if (result == NULL) return NULL;
    std::vector<jlong> packet(total, 0);
    packet[0] = status;
    packet[1] = count;
    packet[2] = heapWords;
    packet[3] = varCount;
    packet[4] = count;
    for (size_t i = 0; i < heap.size(); i++) packet[5 + i] = (jlong) heap[i];
    for (size_t i = 0; i < roots.size(); i++) packet[5 + heap.size() + i] = (jlong) roots[i];
    env->SetLongArrayRegion(result, 0, (jsize) total, packet.data());
    return result;
}

extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierRunSegment(JNIEnv* env, jclass,
        jlongArray branchArray, jlongArray opArray, jlongArray exprProgramArray,
        jlongArray exprStartsArray, jlongArray exprLensArray, jlongArray constantHeapArray) {
    std::vector<long long> emptyHeap;
    std::vector<long long> emptyRoots;
    if (!gGenericSegmented || gGenericHeap == NULL || gGenericHostCurrentSize <= 0) {
        return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_OK, 0, 0, gGenericVarCount, emptyHeap, emptyRoots);
    }
    const int branchLen = env->GetArrayLength(branchArray);
    const int opLen = env->GetArrayLength(opArray);
    const int exprProgramLen = env->GetArrayLength(exprProgramArray);
    const int exprStartsLen = env->GetArrayLength(exprStartsArray);
    const int constantLen = env->GetArrayLength(constantHeapArray);
    if (branchLen <= 0 || exprProgramLen <= 0 || exprStartsLen != env->GetArrayLength(exprLensArray)) {
        return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_BAD_PROGRAM, 0, 0, gGenericVarCount, emptyHeap, emptyRoots);
    }
    if (!genericResidentEnsureProgram(env, branchArray, opArray, exprProgramArray,
            exprStartsArray, exprLensArray, constantHeapArray, branchLen, opLen,
            exprProgramLen, exprStartsLen, constantLen)) {
        return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_BAD_PROGRAM, 0, 0, gGenericVarCount, emptyHeap, emptyRoots);
    }
    const int sourceTop = gGenericSegmentSourceTop;
    const int fixedPrefixWords = gGenericBaseHeapWords + constantLen;
    if (sourceTop <= fixedPrefixWords || sourceTop > gGenericHeapWords) {
        return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_BAD_PROGRAM, 0, 0, gGenericVarCount, emptyHeap, emptyRoots);
    }
    cudaMemcpy(gGenericHeapTop, &sourceTop, sizeof(int), cudaMemcpyHostToDevice);
    const int threads = 128;
    for (int begin = 0; begin < gGenericHostCurrentSize; begin += gGenericBatchStates) {
        const int count = (gGenericHostCurrentSize - begin) < gGenericBatchStates
                ? (gGenericHostCurrentSize - begin) : gGenericBatchStates;
        cudaMemset(gGenericNextSize, 0, sizeof(int));
        cudaMemset(gGenericOverflow, 0, sizeof(int));
        cudaMemset(gGenericStatus, 0, sizeof(int));
        cudaMemcpy(gGenericHeapTop, &sourceTop, sizeof(int), cudaMemcpyHostToDevice);
        const int blocks = (count + threads - 1) / threads;
        genericResidentStepKernel<<<blocks, threads>>>(gGenericResidentBranch, branchLen, gGenericResidentOp, opLen,
                gGenericResidentExprProgram, gGenericResidentExprStarts, gGenericResidentExprLens,
                gGenericHeap, gGenericStateRoots, gGenericNextStateRoots, gGenericCurrent + begin, count,
                gGenericNext, gGenericNextSize, gGenericStateCount, gGenericVisited, gGenericOverflow,
                gGenericStatus, gGenericHeapTop, gGenericHeapWords, gGenericGenerated,
                gGenericHashCapacity, gGenericMaxStates, gGenericVarCount);
        cudaError_t launchErr = cudaGetLastError();
        cudaError_t syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
        int status = GPU_TLA_SUCC_STATUS_OK;
        int overflow = 0;
        int batchCount = 0;
        int batchHeapTop = sourceTop;
        if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
            status = 1000 + (launchErr != cudaSuccess ? (int) launchErr : (int) syncErr);
        } else {
            cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&overflow, gGenericOverflow, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&batchCount, gGenericNextSize, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&batchHeapTop, gGenericHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
        }
        if (status != GPU_TLA_SUCC_STATUS_OK || overflow != 0) {
            if (status == GPU_TLA_SUCC_STATUS_OK) status = GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED;
            return genericSegmentPacket(env, status, 0, 0, gGenericVarCount, emptyHeap, emptyRoots);
        }
        if (batchCount <= 0) continue;

        cudaMemset(gGenericCompactHeapTop, 0, sizeof(int));
        cudaMemset(gGenericStatus, 0, sizeof(int));
        const int compactBlocks = (batchCount + threads - 1) / threads;
        genericResidentCompactBatchKernel<<<compactBlocks, threads>>>(gGenericHeap, batchHeapTop,
                gGenericCompactHeap, gGenericHeapWords, 0, gGenericNextStateRoots, 0, batchCount,
                gGenericVarCount, gGenericCompactHeapTop, gGenericStatus);
        launchErr = cudaGetLastError();
        syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
        cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
        int compactTop = 0;
        cudaMemcpy(&compactTop, gGenericCompactHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
        if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
            status = 1000 + (launchErr != cudaSuccess ? (int) launchErr : (int) syncErr);
        }
        if (status != GPU_TLA_SUCC_STATUS_OK || compactTop < 0 || compactTop > gGenericHeapWords) {
            if (status == GPU_TLA_SUCC_STATUS_OK) status = GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW;
            return genericSegmentPacket(env, status, 0, 0, gGenericVarCount, emptyHeap, emptyRoots);
        }

        std::vector<long long> batchHeap((size_t) compactTop);
        std::vector<int> batchRoots((size_t) batchCount * gGenericVarCount);
        if (compactTop > 0) cudaMemcpy(batchHeap.data(), gGenericCompactHeap,
                (size_t) compactTop * sizeof(long long), cudaMemcpyDeviceToHost);
        cudaMemcpy(batchRoots.data(), gGenericNextStateRoots,
                batchRoots.size() * sizeof(int), cudaMemcpyDeviceToHost);
        const int heapBase = (int) emptyHeap.size();
        if (!genericHostRebaseHeap(batchHeap, heapBase)) {
            return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_HEAP_OVERFLOW, 0, 0,
                    gGenericVarCount, emptyHeap, emptyRoots);
        }
        emptyHeap.insert(emptyHeap.end(), batchHeap.begin(), batchHeap.end());
        for (size_t i = 0; i < batchRoots.size(); i++) emptyRoots.push_back(batchRoots[i] + heapBase);
    }
    gGenericHostCurrentSize = 0;
    return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_OK,
            (int) (emptyRoots.size() / gGenericVarCount), (int) emptyHeap.size(), gGenericVarCount,
            emptyHeap, emptyRoots);
}

extern "C"
JNIEXPORT void JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierCommitLayer(JNIEnv* env, jclass,
        jint nextFrontierSize) {
    if (!gGenericSegmented) return;
    if (nextFrontierSize < 0) {
        jclass errorClass = env->FindClass("java/lang/IllegalArgumentException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "negative generic GPU frontier size");
        return;
    }
    gGenericHostFrontierSize = (int) nextFrontierSize;
    gGenericHostCurrentSize = (int) nextFrontierSize;
    gGenericDepth++;
}

extern "C"
JNIEXPORT jint JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierStep(JNIEnv* env, jclass,
        jlongArray branchArray, jlongArray opArray, jlongArray exprProgramArray,
        jlongArray exprStartsArray, jlongArray exprLensArray, jlongArray constantHeapArray) {
    if (gGenericHeap == NULL || gGenericHostCurrentSize <= 0) return 0;
    const int branchLen = env->GetArrayLength(branchArray);
    const int opLen = env->GetArrayLength(opArray);
    const int exprProgramLen = env->GetArrayLength(exprProgramArray);
    const int exprStartsLen = env->GetArrayLength(exprStartsArray);
    const int exprLensLen = env->GetArrayLength(exprLensArray);
    const int constantLen = env->GetArrayLength(constantHeapArray);
    if (branchLen <= 0 || exprProgramLen <= 0 || exprStartsLen != exprLensLen
            || gGenericBaseHeapWords + constantLen > gGenericHeapWords) {
        int one = 1;
        if (gGenericOverflow) cudaMemcpy(gGenericOverflow, &one, sizeof(int), cudaMemcpyHostToDevice);
        if (gGenericStatus) {
            int status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
            cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        }
        return 0;
    }

    if (!genericResidentEnsureProgram(env, branchArray, opArray, exprProgramArray,
            exprStartsArray, exprLensArray, constantHeapArray, branchLen, opLen,
            exprProgramLen, exprStartsLen, constantLen)) {
        return 0;
    }
    if (gGenericGcEnabled) {
        const int fixedPrefixWords = gGenericBaseHeapWords + constantLen;
        if (!genericResidentRunBatchedLayer(gGenericResidentBranch, branchLen, gGenericResidentOp, opLen,
                gGenericResidentExprProgram, gGenericResidentExprStarts, gGenericResidentExprLens,
                fixedPrefixWords)) {
            return 0;
        }
        return (jint) gGenericHostCurrentSize;
    }
    long long* dBranch = gGenericResidentBranch;
    long long* dOp = gGenericResidentOp;
    long long* dExprProgram = gGenericResidentExprProgram;
    long long* dExprStarts = gGenericResidentExprStarts;
    long long* dExprLens = gGenericResidentExprLens;
    int heapTopBefore = genericResidentReadHeapTop();

    cudaMemset(gGenericNextSize, 0, sizeof(int));
    cudaMemset(gGenericStatus, 0, sizeof(int));
    int threads = 128;
    int blocks = (gGenericHostCurrentSize + threads - 1) / threads;
    genericResidentStepKernel<<<blocks, threads>>>(dBranch, branchLen, dOp, opLen, dExprProgram,
            dExprStarts, dExprLens, gGenericHeap, gGenericStateRoots, gGenericNextStateRoots,
            gGenericCurrent, gGenericHostCurrentSize, gGenericNext, gGenericNextSize,
            gGenericStateCount, gGenericVisited, gGenericOverflow, gGenericStatus, gGenericHeapTop, gGenericHeapWords,
            gGenericGenerated, gGenericHashCapacity, gGenericMaxStates, gGenericVarCount);
    cudaError_t stepLaunchErr = cudaGetLastError();
    cudaError_t stepSyncErr = cudaDeviceSynchronize();
    if (stepLaunchErr != cudaSuccess || stepSyncErr != cudaSuccess) {
        fprintf(stderr, "genericFrontierStep CUDA failure: launch=%d/%s sync=%d/%s current=%d blocks=%d threads=%d branchLen=%d opLen=%d exprProgramLen=%d heapWords=%d\n",
                (int) stepLaunchErr, cudaGetErrorString(stepLaunchErr), (int) stepSyncErr, cudaGetErrorString(stepSyncErr),
                gGenericHostCurrentSize, blocks, threads, branchLen, opLen, exprProgramLen, gGenericHeapWords);
        char message[256];
        snprintf(message, sizeof(message), "generic GPU frontier kernel failed: launch=%s, sync=%s",
                cudaGetErrorString(stepLaunchErr), cudaGetErrorString(stepSyncErr));
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
    }

    int nextSize = 0;
    int status = 0;
    int overflow = 0;
    cudaMemcpy(&nextSize, gGenericNextSize, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&overflow, gGenericOverflow, sizeof(int), cudaMemcpyDeviceToHost);
    if (status == 0 && overflow == 0) {
        const int fixedPrefixWords = gGenericBaseHeapWords + constantLen;
        if (!genericResidentFinishLayer(nextSize, fixedPrefixWords, heapTopBefore)) {
            cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
        }
    }
    if (status == 0 && overflow == 0) {
        int* tmp = gGenericCurrent;
        gGenericCurrent = gGenericNext;
        gGenericNext = tmp;
        gGenericHostCurrentSize = nextSize;
    }
    gGenericDepth++;

    return (jint) nextSize;
}


extern "C"
JNIEXPORT jint JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierDrain(JNIEnv* env, jclass,
        jlongArray branchArray, jlongArray opArray, jlongArray exprProgramArray,
        jlongArray exprStartsArray, jlongArray exprLensArray, jlongArray constantHeapArray,
        jint maxDepth) {
    if (gGenericHeap == NULL || gGenericHostCurrentSize <= 0) return 0;
    const int branchLen = env->GetArrayLength(branchArray);
    const int opLen = env->GetArrayLength(opArray);
    const int exprProgramLen = env->GetArrayLength(exprProgramArray);
    const int exprStartsLen = env->GetArrayLength(exprStartsArray);
    const int exprLensLen = env->GetArrayLength(exprLensArray);
    const int constantLen = env->GetArrayLength(constantHeapArray);
    if (branchLen <= 0 || exprProgramLen <= 0 || exprStartsLen != exprLensLen
            || gGenericBaseHeapWords + constantLen > gGenericHeapWords) {
        int one = 1;
        if (gGenericOverflow) cudaMemcpy(gGenericOverflow, &one, sizeof(int), cudaMemcpyHostToDevice);
        if (gGenericStatus) {
            int status = GPU_TLA_SUCC_STATUS_BAD_PROGRAM;
            cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        }
        return 0;
    }

    if (!genericResidentEnsureProgram(env, branchArray, opArray, exprProgramArray,
            exprStartsArray, exprLensArray, constantHeapArray, branchLen, opLen,
            exprProgramLen, exprStartsLen, constantLen)) {
        return 0;
    }
    if (gGenericGcEnabled) {
        int depthBudget = maxDepth > 0 ? (int) maxDepth : 0x7fffffff;
        const int fixedPrefixWords = gGenericBaseHeapWords + constantLen;
        while (gGenericHostCurrentSize > 0 && depthBudget > 0) {
            if (!genericResidentRunBatchedLayer(gGenericResidentBranch, branchLen, gGenericResidentOp, opLen,
                    gGenericResidentExprProgram, gGenericResidentExprStarts, gGenericResidentExprLens,
                    fixedPrefixWords)) {
                break;
            }
            int status = 0;
            int overflow = 0;
            if (gGenericStatus) cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
            if (gGenericOverflow) cudaMemcpy(&overflow, gGenericOverflow, sizeof(int), cudaMemcpyDeviceToHost);
            if (gGenericHostCurrentSize <= 0 || status != 0 || overflow != 0) break;
            depthBudget--;
        }
        return (jint) gGenericHostCurrentSize;
    }
    long long* dBranch = gGenericResidentBranch;
    long long* dOp = gGenericResidentOp;
    long long* dExprProgram = gGenericResidentExprProgram;
    long long* dExprStarts = gGenericResidentExprStarts;
    long long* dExprLens = gGenericResidentExprLens;
    int heapTopBefore = genericResidentReadHeapTop();

    const bool disableGenericGraph = true;
    bool conditionalGraphFinished = false;
    GenericResidentGraphControl hCtl;
    memset(&hCtl, 0, sizeof(hCtl));
    hCtl.heap = gGenericHeap;
    hCtl.stateRoots = gGenericStateRoots;
    hCtl.current = gGenericCurrent;
    hCtl.next = gGenericNext;
    hCtl.nextSize = gGenericNextSize;
    hCtl.stateCount = gGenericStateCount;
    hCtl.visited = gGenericVisited;
    hCtl.overflow = gGenericOverflow;
    hCtl.status = gGenericStatus;
    hCtl.heapTop = gGenericHeapTop;
    hCtl.generated = gGenericGenerated;
    hCtl.heapWords = gGenericHeapWords;
    hCtl.currentSize = gGenericHostCurrentSize;
    hCtl.hashCapacity = gGenericHashCapacity;
    hCtl.maxStates = gGenericMaxStates;
    hCtl.varCount = gGenericVarCount;
    hCtl.depth = gGenericDepth;
    hCtl.maxDepth = maxDepth > 0 ? (int) maxDepth : 0x7fffffff;

    GenericResidentGraphControl* dCtl = NULL;
    cudaGraph_t condGraph = NULL;
    cudaGraphExec_t condExec = NULL;
    cudaGraphNode_t condNode = NULL;
    cudaGraph_t bodyGraph = NULL;
    cudaGraphConditionalHandle condHandle = 0;
    cudaStream_t condStream = NULL;
    cudaError_t condErr = disableGenericGraph ? cudaErrorInvalidValue : cudaMalloc(&dCtl, sizeof(GenericResidentGraphControl));
    if (condErr == cudaSuccess) condErr = cudaMemcpy(dCtl, &hCtl, sizeof(hCtl), cudaMemcpyHostToDevice);
    if (condErr == cudaSuccess) condErr = cudaStreamCreateWithFlags(&condStream, cudaStreamNonBlocking);
    if (condErr == cudaSuccess) condErr = cudaGraphCreate(&condGraph, 0);
    if (condErr == cudaSuccess) {
        condErr = cudaGraphConditionalHandleCreate(&condHandle, condGraph, 1, cudaGraphCondAssignDefault);
    }
    cudaConditionalNodeParams condParams;
    memset(&condParams, 0, sizeof(condParams));
    condParams.handle = condHandle;
    condParams.type = cudaGraphCondTypeWhile;
    condParams.size = 1;
    if (condErr == cudaSuccess) {
        cudaGraphNodeParams* nodeParams = (cudaGraphNodeParams*) malloc(sizeof(cudaGraphNodeParams));
        if (nodeParams == NULL) {
            condErr = cudaErrorMemoryAllocation;
        } else {
            memset(nodeParams, 0, sizeof(cudaGraphNodeParams));
            nodeParams->type = cudaGraphNodeTypeConditional;
            nodeParams->conditional = condParams;
            condErr = cudaGraphAddNode(&condNode, condGraph, NULL, 0, nodeParams);
            if (condErr == cudaSuccess && nodeParams->conditional.phGraph_out != NULL) {
                bodyGraph = nodeParams->conditional.phGraph_out[0];
            }
            free(nodeParams);
        }
        if (condErr == cudaSuccess && bodyGraph == NULL) {
            condErr = cudaErrorInvalidValue;
        }
    }
    if (condErr == cudaSuccess) {
        cudaGraphNode_t memsetNextNode = NULL;
        cudaGraphNode_t memsetStatusNode = NULL;
        cudaGraphNode_t stepNode = NULL;
        cudaGraphNode_t controlNode = NULL;

        cudaMemsetParams nextMemset;
        memset(&nextMemset, 0, sizeof(nextMemset));
        nextMemset.dst = gGenericNextSize;
        nextMemset.value = 0;
        nextMemset.elementSize = 1;
        nextMemset.width = sizeof(int);
        nextMemset.height = 1;
        condErr = cudaGraphAddMemsetNode(&memsetNextNode, bodyGraph, NULL, 0, &nextMemset);
        if (condErr == cudaSuccess) {
            cudaMemsetParams statusMemset;
            memset(&statusMemset, 0, sizeof(statusMemset));
            statusMemset.dst = gGenericStatus;
            statusMemset.value = 0;
            statusMemset.elementSize = 1;
            statusMemset.width = sizeof(int);
            statusMemset.height = 1;
            condErr = cudaGraphAddMemsetNode(&memsetStatusNode, bodyGraph, NULL, 0, &statusMemset);
        }
        if (condErr == cudaSuccess) {
            void* stepArgs[] = { &dBranch, (void*) &branchLen, &dOp, (void*) &opLen, &dExprProgram,
                    &dExprStarts, &dExprLens, &dCtl };
            cudaKernelNodeParams stepParams;
            memset(&stepParams, 0, sizeof(stepParams));
            stepParams.func = (void*) genericResidentGraphStepKernel;
            stepParams.gridDim = dim3((gGenericMaxStates + 127) / 128, 1, 1);
            stepParams.blockDim = dim3(128, 1, 1);
            stepParams.sharedMemBytes = 0;
            stepParams.kernelParams = stepArgs;
            cudaGraphNode_t deps[2] = { memsetNextNode, memsetStatusNode };
            condErr = cudaGraphAddKernelNode(&stepNode, bodyGraph, deps, 2, &stepParams);
        }
        if (condErr == cudaSuccess) {
            void* controlArgs[] = { &dCtl, &condHandle };
            cudaKernelNodeParams controlParams;
            memset(&controlParams, 0, sizeof(controlParams));
            controlParams.func = (void*) genericResidentGraphControlKernel;
            controlParams.gridDim = dim3(1, 1, 1);
            controlParams.blockDim = dim3(1, 1, 1);
            controlParams.sharedMemBytes = 0;
            controlParams.kernelParams = controlArgs;
            condErr = cudaGraphAddKernelNode(&controlNode, bodyGraph, &stepNode, 1, &controlParams);
        }
    }
    if (condErr == cudaSuccess) condErr = cudaGraphInstantiate(&condExec, condGraph, 0);
    if (condErr == cudaSuccess) condErr = cudaGraphLaunch(condExec, condStream);
    if (condErr == cudaSuccess) condErr = cudaStreamSynchronize(condStream);
    if (condErr == cudaSuccess) {
        cudaMemcpy(&hCtl, dCtl, sizeof(hCtl), cudaMemcpyDeviceToHost);
        gGenericCurrent = hCtl.current;
        gGenericNext = hCtl.next;
        gGenericHostCurrentSize = hCtl.currentSize;
        gGenericDepth = hCtl.depth;
        conditionalGraphFinished = true;
    }
    if (condExec) cudaGraphExecDestroy(condExec);
    if (condGraph) cudaGraphDestroy(condGraph);
    if (condStream) cudaStreamDestroy(condStream);
    if (dCtl) cudaFree(dCtl);

    if (!conditionalGraphFinished) {
    const int threads = 128;
    int currentSizeArg = gGenericHostCurrentSize;
    int* currentPtrArg = gGenericCurrent;
    int* nextPtrArg = gGenericNext;
    void* kernelArgs[] = {
            &dBranch, (void*) &branchLen, &dOp, (void*) &opLen, &dExprProgram,
            &dExprStarts, &dExprLens, &gGenericHeap, &gGenericStateRoots, &gGenericStateRoots,
            &currentPtrArg, &currentSizeArg, &nextPtrArg, &gGenericNextSize,
            &gGenericStateCount, &gGenericVisited, &gGenericOverflow, &gGenericStatus, &gGenericHeapTop, &gGenericHeapWords,
            &gGenericGenerated, &gGenericHashCapacity, &gGenericMaxStates, &gGenericVarCount };

    cudaStream_t graphStream = NULL;
    cudaGraph_t graph = NULL;
    cudaGraphExec_t graphExec = NULL;
    cudaGraphNode_t memsetNextNode = NULL;
    cudaGraphNode_t memsetStatusNode = NULL;
    cudaGraphNode_t kernelNode = NULL;
    bool graphReady = false;

    cudaError_t graphErr = disableGenericGraph ? cudaErrorInvalidValue : cudaStreamCreateWithFlags(&graphStream, cudaStreamNonBlocking);
    if (graphErr == cudaSuccess) graphErr = cudaGraphCreate(&graph, 0);
    if (graphErr == cudaSuccess) {
        cudaMemsetParams nextMemset;
        memset(&nextMemset, 0, sizeof(nextMemset));
        nextMemset.dst = gGenericNextSize;
        nextMemset.value = 0;
        nextMemset.elementSize = 1;
        nextMemset.width = sizeof(int);
        nextMemset.height = 1;
        graphErr = cudaGraphAddMemsetNode(&memsetNextNode, graph, NULL, 0, &nextMemset);
    }
    if (graphErr == cudaSuccess) {
        cudaMemsetParams statusMemset;
        memset(&statusMemset, 0, sizeof(statusMemset));
        statusMemset.dst = gGenericStatus;
        statusMemset.value = 0;
        statusMemset.elementSize = 1;
        statusMemset.width = sizeof(int);
        statusMemset.height = 1;
        graphErr = cudaGraphAddMemsetNode(&memsetStatusNode, graph, NULL, 0, &statusMemset);
    }
    cudaKernelNodeParams kernelParams;
    memset(&kernelParams, 0, sizeof(kernelParams));
    kernelParams.func = (void*) genericResidentStepKernel;
    kernelParams.gridDim = dim3(1, 1, 1);
    kernelParams.blockDim = dim3(threads, 1, 1);
    kernelParams.sharedMemBytes = 0;
    kernelParams.kernelParams = kernelArgs;
    if (graphErr == cudaSuccess) {
        cudaGraphNode_t deps[2] = { memsetNextNode, memsetStatusNode };
        graphErr = cudaGraphAddKernelNode(&kernelNode, graph, deps, 2, &kernelParams);
    }
    if (graphErr == cudaSuccess) {
        graphErr = cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
        graphReady = graphErr == cudaSuccess;
    }

    int depthBudget = maxDepth > 0 ? (int) maxDepth : 0x7fffffff;
    int nextSize = gGenericHostCurrentSize;
    while (gGenericHostCurrentSize > 0 && depthBudget > 0) {
        currentSizeArg = gGenericHostCurrentSize;
        currentPtrArg = gGenericCurrent;
        nextPtrArg = gGenericNext;
        const int blocks = (currentSizeArg + threads - 1) / threads;

        cudaError_t stepLaunchErr = cudaSuccess;
        cudaError_t stepSyncErr = cudaSuccess;
        if (graphReady) {
            kernelParams.gridDim = dim3(blocks, 1, 1);
            kernelParams.blockDim = dim3(threads, 1, 1);
            kernelParams.kernelParams = kernelArgs;
            stepLaunchErr = cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &kernelParams);
            if (stepLaunchErr == cudaSuccess) stepLaunchErr = cudaGraphLaunch(graphExec, graphStream);
            stepSyncErr = cudaStreamSynchronize(graphStream);
        } else {
            cudaMemset(gGenericNextSize, 0, sizeof(int));
            cudaMemset(gGenericStatus, 0, sizeof(int));
            genericResidentStepKernel<<<blocks, threads>>>(dBranch, branchLen, dOp, opLen, dExprProgram,
                    dExprStarts, dExprLens, gGenericHeap, gGenericStateRoots, gGenericNextStateRoots,
                    gGenericCurrent, gGenericHostCurrentSize, gGenericNext, gGenericNextSize,
                    gGenericStateCount, gGenericVisited, gGenericOverflow, gGenericStatus, gGenericHeapTop, gGenericHeapWords,
                    gGenericGenerated, gGenericHashCapacity, gGenericMaxStates, gGenericVarCount);
            stepLaunchErr = cudaGetLastError();
            stepSyncErr = cudaDeviceSynchronize();
        }
        if (stepLaunchErr != cudaSuccess || stepSyncErr != cudaSuccess) {
            int status = 1000 + (stepLaunchErr != cudaSuccess ? (int) stepLaunchErr : (int) stepSyncErr);
            cudaMemcpy(gGenericStatus, &status, sizeof(int), cudaMemcpyHostToDevice);
        }

        int status = 0;
        int overflow = 0;
        cudaMemcpy(&nextSize, gGenericNextSize, sizeof(int), cudaMemcpyDeviceToHost);
        if (gGenericStatus) cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
        if (gGenericOverflow) cudaMemcpy(&overflow, gGenericOverflow, sizeof(int), cudaMemcpyDeviceToHost);
        if (status == 0 && overflow == 0) {
            const int fixedPrefixWords = gGenericBaseHeapWords + constantLen;
            if (!genericResidentFinishLayer(nextSize, fixedPrefixWords, heapTopBefore)) {
                cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
            }
            heapTopBefore = genericResidentReadHeapTop();
        }
        if (status == 0 && overflow == 0) {
            int* tmp = gGenericCurrent;
            gGenericCurrent = gGenericNext;
            gGenericNext = tmp;
            gGenericHostCurrentSize = nextSize;
        }
        gGenericDepth++;
        depthBudget--;
        if (nextSize <= 0 || status != 0 || overflow != 0) break;
    }

    if (graphExec) cudaGraphExecDestroy(graphExec);
    if (graph) cudaGraphDestroy(graph);
    if (graphStream) cudaStreamDestroy(graphStream);
    }

    return (jint) gGenericHostCurrentSize;
}

extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierStats(JNIEnv* env, jclass) {
    jlongArray result = env->NewLongArray(12);
    unsigned long long generated = 0ULL;
    int distinct = 0;
    int overflow = 0;
    int status = 0;
    int heapWords = 0;
    if (gGenericGenerated) cudaMemcpy(&generated, gGenericGenerated, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (gGenericGpuSegmented) {
        distinct = gGpuSegmentedDistinctCount > 2147483647ULL
                ? 2147483647 : (int) gGpuSegmentedDistinctCount;
    } else if (gGenericStateCount) {
        cudaMemcpy(&distinct, gGenericStateCount, sizeof(int), cudaMemcpyDeviceToHost);
    }
    if (gGenericOverflow) cudaMemcpy(&overflow, gGenericOverflow, sizeof(int), cudaMemcpyDeviceToHost);
    if (gGenericStatus) cudaMemcpy(&status, gGenericStatus, sizeof(int), cudaMemcpyDeviceToHost);
    if (gGenericHeapTop) cudaMemcpy(&heapWords, gGenericHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
    jlong stats[12];
    stats[0] = (jlong) generated;
    stats[1] = gGenericGpuSegmented
            ? (jlong) gGpuSegmentedDistinctCount : (jlong) distinct;
    stats[2] = (jlong) (gGenericSegmented ? gGenericHostFrontierSize : gGenericHostCurrentSize);
    stats[3] = (jlong) gGenericDepth;
    stats[4] = (jlong) overflow;
    stats[5] = (jlong) status;
    stats[6] = (jlong) heapWords;
    stats[7] = (jlong) gGenericPeakHeapWords;
    stats[8] = (jlong) gGenericGcCount;
    stats[9] = (jlong) gGenericGcSkippedCount;
    stats[10] = (jlong) gGenericLayersSinceGc;
    stats[11] = gGenericGcEnabled ? 1L : 0L;
    env->SetLongArrayRegion(result, 0, 12, stats);
    return result;
}

extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierTransferStats(JNIEnv* env, jclass) {
    unsigned long long shareHits = 0ULL;
    unsigned long long shareMisses = 0ULL;
    if (gGpuShareHits) cudaMemcpy(&shareHits, gGpuShareHits, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    if (gGpuShareMisses) cudaMemcpy(&shareMisses, gGpuShareMisses, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    jlong stats[12];
    stats[0] = (jlong) gTransferH2DBytes;
    stats[1] = (jlong) gTransferD2HPayloadBytes;
    stats[2] = (jlong) gTransferD2HControlBytes;
    stats[3] = (jlong) gTransferH2DCalls;
    stats[4] = (jlong) gTransferD2HCalls;
    stats[5] = (jlong) gTransferH2DWaitNanos;
    stats[6] = (jlong) gTransferD2HWaitNanos;
    stats[7] = (jlong) gTransferSegmentHeaders;
    stats[8] = (jlong) gTransferAcceptedStates;
    stats[9] = (jlong) gTransferDynamicHeapWords;
    stats[10] = (jlong) shareHits;
    stats[11] = (jlong) shareMisses;
    jlongArray result = env->NewLongArray(12);
    if (result != NULL) env->SetLongArrayRegion(result, 0, 12, stats);
    return result;
}

extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_genericFrontierSnapshot(JNIEnv* env, jclass) {
    std::vector<long long> emptyHeap;
    std::vector<long long> emptyRoots;
    if (gGenericHeap == NULL || gGenericStateRoots == NULL || gGenericCurrent == NULL
            || gGenericNextStateRoots == NULL || gGenericHostCurrentSize <= 0) {
        return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_OK, 0, 0, gGenericVarCount,
                emptyHeap, emptyRoots);
    }

    int heapTop = 0;
    cudaError_t err = cudaMemcpy(&heapTop, gGenericHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
    const int count = gGenericHostCurrentSize;
    if (err != cudaSuccess || heapTop < 0 || heapTop > gGenericHeapWords
            || count < 0 || count > gGenericMaxStates || gGenericVarCount <= 0) {
        return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_BAD_PROGRAM, 0, 0,
                gGenericVarCount, emptyHeap, emptyRoots);
    }

    std::vector<long long> heap((size_t) heapTop);
    if (heapTop > 0) {
        err = cudaMemcpy(heap.data(), gGenericHeap, (size_t) heapTop * sizeof(long long),
                cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        return genericSegmentPacket(env, 1000 + (int) err, 0, 0, gGenericVarCount,
                emptyHeap, emptyRoots);
    }

    const long long totalRootsLong = (long long) count * gGenericVarCount;
    if (totalRootsLong > 2147483647LL) {
        return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_CAPACITY_EXCEEDED, 0, 0,
                gGenericVarCount, emptyHeap, emptyRoots);
    }
    const int totalRoots = (int) totalRootsLong;
    std::vector<int> roots((size_t) totalRoots);
    const int threads = 128;
    const int blocks = (totalRoots + threads - 1) / threads;
    genericResidentSnapshotRootsKernel<<<blocks, threads>>>(gGenericStateRoots, gGenericCurrent,
            count, gGenericVarCount, gGenericNextStateRoots);
    const cudaError_t launchErr = cudaGetLastError();
    const cudaError_t syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
    if (launchErr != cudaSuccess || syncErr != cudaSuccess) {
        const int status = 1000 + (int) (launchErr != cudaSuccess ? launchErr : syncErr);
        return genericSegmentPacket(env, status, 0, 0, gGenericVarCount,
                emptyHeap, emptyRoots);
    }
    if (totalRoots > 0) {
        err = cudaMemcpy(roots.data(), gGenericNextStateRoots,
                (size_t) totalRoots * sizeof(int), cudaMemcpyDeviceToHost);
    }
    if (err != cudaSuccess) {
        return genericSegmentPacket(env, 1000 + (int) err, 0, 0, gGenericVarCount,
                emptyHeap, emptyRoots);
    }

    std::vector<long long> rootsAsLong((size_t) totalRoots);
    for (int i = 0; i < totalRoots; i++) rootsAsLong[(size_t) i] = roots[(size_t) i];
    return genericSegmentPacket(env, GPU_TLA_SUCC_STATUS_OK, count, heapTop,
            gGenericVarCount, heap, rootsAsLong);
}


extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_evalIR(JNIEnv* env, jclass,
        jlongArray programArray, jlongArray heapArray, jlongArray currentRootsArray, jlongArray nextRootsArray) {
    jlongArray result = env->NewLongArray(4);
    jlong defaultResult[4] = { (jlong) GPU_TLA_IR_STATUS_TYPE_ERROR, 0L, -1L, 0L };

    const int programLen = env->GetArrayLength(programArray);
    if (!gpuTlaConfigureRuntime(env)) return NULL;
    const int heapLen = env->GetArrayLength(heapArray);
    const int currentLen = env->GetArrayLength(currentRootsArray);
    const int nextLen = env->GetArrayLength(nextRootsArray);
    if (programLen <= 0 || heapLen <= 0 || currentLen <= 0 || currentLen != nextLen) {
        env->SetLongArrayRegion(result, 0, 4, defaultResult);
        return result;
    }

    jlong* hProgramJ = env->GetLongArrayElements(programArray, NULL);
    jlong* hHeapJ = env->GetLongArrayElements(heapArray, NULL);
    jlong* hCurrentJ = env->GetLongArrayElements(currentRootsArray, NULL);
    jlong* hNextJ = env->GetLongArrayElements(nextRootsArray, NULL);

    long long* dProgram = NULL;
    long long* dHeap = NULL;
    long long* dCurrent = NULL;
    long long* dNext = NULL;
    long long* dOut = NULL;
    int* dHeapTop = NULL;
    long long hOut[4] = { (long long) GPU_TLA_IR_STATUS_TYPE_ERROR, 0LL, -1LL, 0LL };
    const int heapCapacity = heapLen + 4096;

    cudaMalloc(&dProgram, (size_t) programLen * sizeof(long long));
    cudaMalloc(&dHeap, (size_t) heapCapacity * sizeof(long long));
    cudaMalloc(&dCurrent, (size_t) currentLen * sizeof(long long));
    cudaMalloc(&dNext, (size_t) nextLen * sizeof(long long));
    cudaMalloc(&dOut, 4 * sizeof(long long));
    cudaMalloc(&dHeapTop, sizeof(int));
    cudaMemcpy(dProgram, hProgramJ, (size_t) programLen * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemset(dHeap, 0, (size_t) heapCapacity * sizeof(long long));
    cudaMemcpy(dHeap, hHeapJ, (size_t) heapLen * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(dHeapTop, &heapLen, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dCurrent, hCurrentJ, (size_t) currentLen * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(dNext, hNextJ, (size_t) nextLen * sizeof(long long), cudaMemcpyHostToDevice);

    gpuTlaEvalIrKernel<<<1, 1>>>(dProgram, programLen, dHeap, dCurrent, dNext, currentLen, dOut, dHeapTop, heapCapacity);
    cudaDeviceSynchronize();
    cudaMemcpy(hOut, dOut, 4 * sizeof(long long), cudaMemcpyDeviceToHost);

    cudaFree(dProgram);
    cudaFree(dHeap);
    cudaFree(dCurrent);
    cudaFree(dNext);
    cudaFree(dOut);
    cudaFree(dHeapTop);

    env->ReleaseLongArrayElements(programArray, hProgramJ, JNI_ABORT);
    env->ReleaseLongArrayElements(heapArray, hHeapJ, JNI_ABORT);
    env->ReleaseLongArrayElements(currentRootsArray, hCurrentJ, JNI_ABORT);
    env->ReleaseLongArrayElements(nextRootsArray, hNextJ, JNI_ABORT);

    jlong out[4] = { (jlong) hOut[0], (jlong) hOut[1], (jlong) hOut[2], (jlong) hOut[3] };
    env->SetLongArrayRegion(result, 0, 4, out);
    return result;
}

extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_fingerprintState(JNIEnv* env, jclass,
        jlongArray heapArray, jlongArray rootsArray) {
    if (!gpuTlaConfigureRuntime(env)) return NULL;
    const int heapLen = env->GetArrayLength(heapArray);
    const int varCount = env->GetArrayLength(rootsArray);
    jlongArray result = env->NewLongArray(1);
    if (heapLen <= 0 || varCount <= 0) return result;

    jlong* hHeap = env->GetLongArrayElements(heapArray, NULL);
    jlong* hRoots = env->GetLongArrayElements(rootsArray, NULL);
    long long* dHeap = NULL;
    long long* dRoots = NULL;
    unsigned long long* dOut = NULL;
    unsigned long long fingerprint = 0ULL;
    cudaMalloc(&dHeap, (size_t) heapLen * sizeof(long long));
    cudaMalloc(&dRoots, (size_t) varCount * sizeof(long long));
    cudaMalloc(&dOut, sizeof(unsigned long long));
    cudaMemcpy(dHeap, hHeap, (size_t) heapLen * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemcpy(dRoots, hRoots, (size_t) varCount * sizeof(long long), cudaMemcpyHostToDevice);
    gpuTlaFingerprintStateKernel<<<1, 1>>>(dHeap, dRoots, varCount, dOut);
    cudaDeviceSynchronize();
    cudaMemcpy(&fingerprint, dOut, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(dHeap);
    cudaFree(dRoots);
    cudaFree(dOut);
    env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
    env->ReleaseLongArrayElements(rootsArray, hRoots, JNI_ABORT);
    const jlong out[1] = { (jlong) fingerprint };
    env->SetLongArrayRegion(result, 0, 1, out);
    return result;
}

extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_deduplicateFingerprints(JNIEnv* env, jclass,
        jlongArray fingerprintsArray) {
    if (!gpuTlaConfigureRuntime(env)) return NULL;
    const int count = env->GetArrayLength(fingerprintsArray);
    jlongArray result = env->NewLongArray(count);
    if (count <= 0) return result;
    const int capacity = nextPow2Int(count < 2 ? 2 : count * 2);
    jlong* hFingerprints = env->GetLongArrayElements(fingerprintsArray, NULL);
    long long* dFingerprints = NULL;
    unsigned long long* dTable = NULL;
    unsigned char* dKeep = NULL;
    unsigned char* hKeep = new unsigned char[(size_t) count];
    jlong* out = new jlong[(size_t) count];
    cudaMalloc(&dFingerprints, (size_t) count * sizeof(long long));
    cudaMalloc(&dTable, (size_t) capacity * sizeof(unsigned long long));
    cudaMalloc(&dKeep, (size_t) count * sizeof(unsigned char));
    cudaMemcpy(dFingerprints, hFingerprints, (size_t) count * sizeof(long long), cudaMemcpyHostToDevice);
    cudaMemset(dTable, 0, (size_t) capacity * sizeof(unsigned long long));
    cudaMemset(dKeep, 0, (size_t) count * sizeof(unsigned char));
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    gpuTlaDeduplicateFingerprintsKernel<<<blocks, threads>>>(dFingerprints, count, dTable, capacity, dKeep);
    cudaDeviceSynchronize();
    cudaMemcpy(hKeep, dKeep, (size_t) count * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    for (int i = 0; i < count; i++) out[i] = hKeep[i] == 0 ? 0L : 1L;
    env->SetLongArrayRegion(result, 0, count, out);
    delete[] out;
    delete[] hKeep;
    cudaFree(dFingerprints);
    cudaFree(dTable);
    cudaFree(dKeep);
    env->ReleaseLongArrayElements(fingerprintsArray, hFingerprints, JNI_ABORT);
    return result;
}
extern "C"
JNIEXPORT jlongArray JNICALL Java_tlc2_tool_gpu_GPUKernel_expandSuccessors(JNIEnv* env, jclass,
        jlongArray branchArray, jlongArray opArray, jlongArray exprProgramArray,
        jlongArray exprStartsArray, jlongArray exprLensArray, jlongArray heapArray,
        jlongArray currentRootsArray, jint maxSuccessors) {
    if (!gpuTlaConfigureRuntime(env)) return NULL;
    const int branchLen = env->GetArrayLength(branchArray);
    const int opLen = env->GetArrayLength(opArray);
    const int exprProgramLen = env->GetArrayLength(exprProgramArray);
    const int exprStartsLen = env->GetArrayLength(exprStartsArray);
    const int exprLensLen = env->GetArrayLength(exprLensArray);
    const int heapLen = env->GetArrayLength(heapArray);
    const int varCount = env->GetArrayLength(currentRootsArray);
    const int maxOut = maxSuccessors > 0 ? (int) maxSuccessors : 1;

    if (branchLen <= 0 || opLen < 0 || exprProgramLen <= 0 || exprStartsLen != exprLensLen
            || heapLen <= 0 || varCount <= 0) {
        jlongArray bad = env->NewLongArray(4);
        jlong raw[4] = { (jlong) GPU_TLA_SUCC_STATUS_BAD_PROGRAM, (jlong) varCount, 0L, 0L };
        env->SetLongArrayRegion(bad, 0, 4, raw);
        return bad;
    }

    const long long requestedHeapCapacityLong = (long long) heapLen + 4096LL
            + (long long) maxOut * (varCount + 1LL) * 4LL;
    if (requestedHeapCapacityLong > 2147483647LL) {
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "generic GPU successor heap request exceeds 32-bit capacity");
        return NULL;
    }
    const int requestedHeapCapacity = (int) requestedHeapCapacityLong;
    const cudaError_t cacheErr = ensureGenericSuccessorCache(branchLen, opLen, exprProgramLen,
            exprStartsLen, varCount, maxOut, requestedHeapCapacity);
    if (cacheErr != cudaSuccess) {
        char message[256];
        snprintf(message, sizeof(message), "generic GPU successor cache allocation failed: %s",
                cudaGetErrorString(cacheErr));
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        return NULL;
    }

    jlong* hBranch = env->GetLongArrayElements(branchArray, NULL);
    jlong* hOp = env->GetLongArrayElements(opArray, NULL);
    jlong* hExprProgram = env->GetLongArrayElements(exprProgramArray, NULL);
    jlong* hExprStarts = env->GetLongArrayElements(exprStartsArray, NULL);
    jlong* hExprLens = env->GetLongArrayElements(exprLensArray, NULL);
    jlong* hHeap = env->GetLongArrayElements(heapArray, NULL);
    jlong* hRoots = env->GetLongArrayElements(currentRootsArray, NULL);

    cudaError_t ioErr = cudaMemcpy(gSuccBranch, hBranch,
            (size_t) branchLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess && opLen > 0) ioErr = cudaMemcpy(gSuccOp, hOp,
            (size_t) opLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess) ioErr = cudaMemcpy(gSuccExprProgram, hExprProgram,
            (size_t) exprProgramLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess) ioErr = cudaMemcpy(gSuccExprStarts, hExprStarts,
            (size_t) exprStartsLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess) ioErr = cudaMemcpy(gSuccExprLens, hExprLens,
            (size_t) exprLensLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess) ioErr = cudaMemcpy(gSuccHeap, hHeap,
            (size_t) heapLen * sizeof(long long), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess) ioErr = cudaMemcpy(gSuccRoots, hRoots,
            (size_t) varCount * sizeof(long long), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess) ioErr = cudaMemcpy(gSuccHeapTop, &heapLen,
            sizeof(int), cudaMemcpyHostToDevice);
    if (ioErr == cudaSuccess) ioErr = cudaMemset(gSuccOut, 0, 4 * sizeof(long long));

    long long header[4] = { 0LL, (long long) varCount, 0LL, 0LL };
    int returnedHeapWords = 0;
    cudaError_t launchErr = cudaSuccess;
    cudaError_t syncErr = cudaSuccess;
    if (ioErr == cudaSuccess) {
        gpuTlaExpandSuccessorsKernel<<<1, 1>>>(gSuccBranch, branchLen, gSuccOp, opLen,
                gSuccExprProgram, gSuccExprStarts, gSuccExprLens, gSuccHeap, gSuccRoots,
                varCount, maxOut, gSuccHeapTop, gSuccHeapCapacity, gSuccOut);
        launchErr = cudaGetLastError();
        syncErr = launchErr == cudaSuccess ? cudaDeviceSynchronize() : launchErr;
    }
    if (ioErr == cudaSuccess && launchErr == cudaSuccess && syncErr == cudaSuccess) {
        ioErr = cudaMemcpy(header, gSuccOut, 4 * sizeof(long long), cudaMemcpyDeviceToHost);
    }
    if (ioErr == cudaSuccess) {
        ioErr = cudaMemcpy(&returnedHeapWords, gSuccHeapTop, sizeof(int), cudaMemcpyDeviceToHost);
    }

    if (ioErr != cudaSuccess || launchErr != cudaSuccess || syncErr != cudaSuccess) {
        char message[512];
        snprintf(message, sizeof(message),
                "generic GPU successor execution failed: transfer=%s, launch=%s, sync=%s",
                cudaGetErrorString(ioErr), cudaGetErrorString(launchErr), cudaGetErrorString(syncErr));
        env->ReleaseLongArrayElements(branchArray, hBranch, JNI_ABORT);
        env->ReleaseLongArrayElements(opArray, hOp, JNI_ABORT);
        env->ReleaseLongArrayElements(exprProgramArray, hExprProgram, JNI_ABORT);
        env->ReleaseLongArrayElements(exprStartsArray, hExprStarts, JNI_ABORT);
        env->ReleaseLongArrayElements(exprLensArray, hExprLens, JNI_ABORT);
        env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
        env->ReleaseLongArrayElements(currentRootsArray, hRoots, JNI_ABORT);
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        return NULL;
    }

    int count = header[2] < 0LL ? 0 : (int) header[2];
    if (count > maxOut) count = maxOut;
    if (returnedHeapWords < heapLen) returnedHeapWords = heapLen;
    if (returnedHeapWords > gSuccHeapCapacity) returnedHeapWords = gSuccHeapCapacity;
    const long long compactLenLong = 5LL + count + (long long) count * varCount + returnedHeapWords;
    if (compactLenLong > 2147483647LL) {
        env->ReleaseLongArrayElements(branchArray, hBranch, JNI_ABORT);
        env->ReleaseLongArrayElements(opArray, hOp, JNI_ABORT);
        env->ReleaseLongArrayElements(exprProgramArray, hExprProgram, JNI_ABORT);
        env->ReleaseLongArrayElements(exprStartsArray, hExprStarts, JNI_ABORT);
        env->ReleaseLongArrayElements(exprLensArray, hExprLens, JNI_ABORT);
        env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
        env->ReleaseLongArrayElements(currentRootsArray, hRoots, JNI_ABORT);
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, "generic GPU successor result exceeds Java array capacity");
        return NULL;
    }

    const int compactLen = (int) compactLenLong;
    jlongArray result = env->NewLongArray(compactLen);
    jlong* compact = new jlong[(size_t) compactLen];
    compact[0] = (jlong) header[0];
    compact[1] = (jlong) header[1];
    compact[2] = (jlong) count;
    compact[3] = (jlong) header[3];
    int cursor = 4;
    if (count > 0) {
        ioErr = cudaMemcpy(compact + cursor, gSuccOut + 4,
                (size_t) count * sizeof(long long), cudaMemcpyDeviceToHost);
        cursor += count;
        if (ioErr == cudaSuccess) ioErr = cudaMemcpy(compact + cursor, gSuccOut + 4 + maxOut,
                (size_t) count * varCount * sizeof(long long), cudaMemcpyDeviceToHost);
        cursor += count * varCount;
    }
    compact[cursor++] = (jlong) returnedHeapWords;
    if (ioErr == cudaSuccess && returnedHeapWords > 0) {
        ioErr = cudaMemcpy(compact + cursor, gSuccHeap,
                (size_t) returnedHeapWords * sizeof(long long), cudaMemcpyDeviceToHost);
    }

    env->ReleaseLongArrayElements(branchArray, hBranch, JNI_ABORT);
    env->ReleaseLongArrayElements(opArray, hOp, JNI_ABORT);
    env->ReleaseLongArrayElements(exprProgramArray, hExprProgram, JNI_ABORT);
    env->ReleaseLongArrayElements(exprStartsArray, hExprStarts, JNI_ABORT);
    env->ReleaseLongArrayElements(exprLensArray, hExprLens, JNI_ABORT);
    env->ReleaseLongArrayElements(heapArray, hHeap, JNI_ABORT);
    env->ReleaseLongArrayElements(currentRootsArray, hRoots, JNI_ABORT);

    if (ioErr != cudaSuccess) {
        char message[256];
        snprintf(message, sizeof(message), "generic GPU successor result transfer failed: %s",
                cudaGetErrorString(ioErr));
        delete[] compact;
        jclass errorClass = env->FindClass("java/lang/RuntimeException");
        if (errorClass != NULL) env->ThrowNew(errorClass, message);
        return NULL;
    }
    env->SetLongArrayRegion(result, 0, compactLen, compact);
    delete[] compact;
    return result;
}
