package tlc2.tool.gpu;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.fail;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;
import org.junit.Assume;

import tlc2.tool.StateVec;
import tlc2.tool.TLCState;
import tlc2.tool.impl.FastTool;
import tlc2.tool.impl.Tool;
import tlc2.value.impl.IntValue;
import tlc2.value.impl.RecordValue;
import util.SimpleFilenameToStream;
import util.ToolIO;

public class GpuTlaCompilerTest {

    @Test
    public void semanticStateRoundTripsForPropertyChecking() throws Exception {
        final Path models = findModels();
        ToolIO.setUserDir(models.toString());
        final FastTool tool = new FastTool("", "GGpuDynamicCollectionsSmall",
                "GGpuDynamicCollectionsSmall",
                new SimpleFilenameToStream(new String[] { models.toString() }), Tool.Mode.Simulation);
        final TLCState initial = tool.getInitStates().elementAt(0);
        final ValueEncoder.EncodedSemanticState encoded = ValueEncoder.encodeSemanticState(initial);
        final TLCState decoded = ValueEncoder.decodeSemanticState(
                encoded, encoded.heap, encoded.roots, 0, initial.getLevel());

        assertEquals(initial, decoded);
        assertEquals(initial.fingerPrint(tool), decoded.fingerPrint(tool));
        assertEquals(initial.getLevel(), decoded.getLevel());
    }

    @Test
    public void compilerRegistersRecordFieldsCreatedOnlyByGpu() throws Exception {
        compile("GGpuTwoPhaseExistsSmall");

        final long typeId = ValueEncoder.stableStringId("type");
        final long[] heap = new long[] {
                semanticHeader(ValueEncoder.TAG_INT, 0, 0L),
                semanticHeader(ValueEncoder.TAG_RECORD, 1, 0L), typeId, 0L
        };
        final String[] names = new String[] { "rmState", "msgs" };
        final ValueEncoder.EncodedSemanticState template = new ValueEncoder.EncodedSemanticState(
                names,
                new long[] { ValueEncoder.stableStringId(names[0]), ValueEncoder.stableStringId(names[1]) },
                new int[] { 1, 1 }, heap);

        final TLCState decoded = ValueEncoder.decodeSemanticState(
                template, heap, new int[] { 1, 1 }, 0, 1);
        final RecordValue record = (RecordValue) decoded.lookup("msgs");
        assertEquals(1, record.names.length);
        assertEquals("type", record.names[0].toString());
        assertEquals(IntValue.ValZero, record.values[0]);
    }

    private static long semanticHeader(final int tag, final int arity, final long aux) {
        return ((long) (tag & 0xff) << 56)
                | ((long) (arity & 0x00ffffff) << 32)
                | (aux & 0xffffffffL);
    }

    @Test
    public void compilesCompleteFiniteActionSubset() throws Exception {
        final GpuTlaSuccessorIR relations = compile("GGpuRelationsSmall");
        assertTrue(isRelationPackage(relations));
        assertTrue(hasSuccessorOp(relations, GpuTlaSuccessorIR.OP_RELATION_EXISTS));
        assertTrue(hasOrderedNextEnums(relations, 1, 0));

        final GpuTlaSuccessorIR collections = compile("GGpuDynamicCollectionsSmall");
        assertTrue(hasSuccessorOp(collections, GpuTlaSuccessorIR.OP_ENUM_LOCAL_NESTED));
        assertTrue(hasExprOp(collections, GpuTlaIR.OP_FCN_FROM_PAIRS));
        assertTrue(hasExprOp(collections, GpuTlaIR.OP_LOAD_NEXT_VAR));

        final GpuTlaSuccessorIR tupleExists = compile("GGpuTupleExistsSmall");
        assertTrue(hasSuccessorOp(tupleExists, GpuTlaSuccessorIR.OP_ENUM_LOCAL));
        assertTrue(hasExprOp(tupleExists, GpuTlaIR.OP_SAVE_LOCAL));

        final GpuTlaSuccessorIR wrappers = compile("GGpuActionWrappersSmall");
        assertFalse(isRelationPackage(wrappers));
        assertTrue(hasExprOp(wrappers, GpuTlaIR.OP_LOAD_NEXT_VAR));
        assertTrue(hasExprOp(wrappers, GpuTlaIR.OP_NOT));

        final GpuTlaSuccessorIR conditional = compile("GGpuActionIfSmall");
        assertFalse(isRelationPackage(conditional));
        assertTrue(hasSuccessorOp(conditional, GpuTlaSuccessorIR.OP_ASSIGN));
        assertTrue(hasExprOp(conditional, GpuTlaIR.OP_NOT));

        final GpuTlaSuccessorIR guardedDynamic = compile("GGpuGuardedDynamicExistsSmall");
        assertTrue(hasSuccessorOp(guardedDynamic, GpuTlaSuccessorIR.OP_ENUM_LOCAL_NESTED));

        final GpuTlaSuccessorIR arithmetic = compile("GGpuDivisionModuloSmall");
        assertTrue(hasExprOp(arithmetic, GpuTlaIR.OP_DIV));
        assertTrue(hasExprOp(arithmetic, GpuTlaIR.OP_MOD));

        final GpuTlaSuccessorIR finiteCollections = compile("GGpuCardinalityConcatSmall");
        assertTrue(hasExprOp(finiteCollections, GpuTlaIR.OP_CARDINALITY));
        assertTrue(hasExprOp(finiteCollections, GpuTlaIR.OP_CONCAT));
        assertTrue(hasExprOp(finiteCollections, GpuTlaIR.OP_LEN));
    }

    @Test
    public void compilesConfiguredInvariantsForDeviceChecking() throws Exception {
        final Path models = findModels();
        ToolIO.setUserDir(models.toString());
        final FastTool tool = new FastTool("", "GGpuDynamicCollectionsSmall",
                "GGpuDynamicCollectionsSmall",
                new SimpleFilenameToStream(new String[] { models.toString() }), Tool.Mode.MC);
        final StateVec initial = tool.getInitStates();
        assertTrue(initial.size() > 0);
        final GpuTlaSuccessorIR ir = GpuTlaSuccessorCompiler.compileActions(
                tool, initial.elementAt(0), tool.getActions(), tool.getInvariants());

        assertArrayEquals(new int[] { 0 }, ir.invariantOrdinals());
        assertTrue(ir.invariantExprFirst() >= 0);
    }

    @Test
    public void compilesInvariantForLaterInitialStates() throws Exception {
        final Path models = findModels();
        ToolIO.setUserDir(models.toString());
        final FastTool tool = new FastTool("", "GGpuMultiInitialInvariantViolationSmall",
                "GGpuMultiInitialInvariantViolationSmall",
                new SimpleFilenameToStream(new String[] { models.toString() }), Tool.Mode.MC);
        final StateVec initial = tool.getInitStates();
        assertEquals(2, initial.size());
        final GpuTlaSuccessorIR ir = GpuTlaSuccessorCompiler.compileActions(
                tool, initial.elementAt(0), tool.getActions(), tool.getInvariants());

        assertArrayEquals(new int[] { 0 }, ir.invariantOrdinals());
        assertTrue(ir.invariantExprFirst() >= 0);
    }

    @Test
    public void deviceRejectsLaterInitialState() throws Exception {
        Assume.assumeTrue(Boolean.getBoolean("tlc.gpu.test.native"));
        System.setProperty("tlc.gpu.generic.frontier", "gpu-segmented");
        System.setProperty("tlc.gpu.generic.max.states", "64");
        System.setProperty("tlc.gpu.generic.hash.capacity", "128");
        System.setProperty("tlc.gpu.generic.heap.extra.words", "4096");
        System.setProperty("tlc.gpu.generic.frontier.segment.max.states", "16");
        System.setProperty("tlc.gpu.generic.gpu.segment.accepted.max.states", "64");
        System.setProperty("tlc.gpu.generic.gpu.segment.output.max.states", "16");
        System.setProperty("tlc.gpu.generic.gpu.segment.output.heap.words", "4096");
        System.setProperty("tlc.gpu.generic.gpu.segment.share.capacity", "1024");

        final Path models = findModels();
        ToolIO.setUserDir(models.toString());
        final FastTool tool = new FastTool("", "GGpuMultiInitialInvariantViolationSmall",
                "GGpuMultiInitialInvariantViolationSmall",
                new SimpleFilenameToStream(new String[] { models.toString() }), Tool.Mode.MC);
        final StateVec initial = tool.getInitStates();
        assertEquals(2, initial.size());

        GPUFrontier.init(64, 1);
        GPUFrontier.beginInitialGeneric(tool);
        GPUFrontier.appendInitialGeneric(initial.elementAt(0));
        GPUFrontier.appendInitialGeneric(initial.elementAt(1));
        try {
            GPUFrontier.finishInitialGeneric();
            fail("expected the GPU to reject the second initial state");
        } catch (GPUFrontier.InvariantViolationException violation) {
            assertEquals(0, violation.invariantOrdinal());
            assertEquals("1", violation.state().lookup("x").toString());
            assertEquals(TLCState.INIT_LEVEL, violation.state().getLevel());
        }
    }

    private static GpuTlaSuccessorIR compile(final String module) throws Exception {
        final Path models = findModels();
        ToolIO.setUserDir(models.toString());
        final FastTool tool = new FastTool("", module, module,
                new SimpleFilenameToStream(new String[] { models.toString() }), Tool.Mode.Simulation);
        final StateVec initial = tool.getInitStates();
        assertTrue("expected at least one initial state for " + module, initial.size() > 0);
        return GpuTlaSuccessorCompiler.compileActions(tool, initial.elementAt(0), tool.getActions());
    }

    private static Path findModels() {
        return findRepositoryRoot().resolve("gpu_generic_tests");
    }

    private static Path findRepositoryRoot() {
        Path path = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        while (path != null) {
            final Path candidate = path.resolve("gpu_generic_tests");
            if (Files.isDirectory(candidate)) return path;
            path = path.getParent();
        }
        throw new IllegalStateException("cannot locate gpu_generic_tests from " + System.getProperty("user.dir"));
    }

    private static boolean isRelationPackage(final GpuTlaSuccessorIR ir) {
        final long[] table = ir.branchTable();
        return table.length >= GpuTlaSuccessorIR.RELATION_HEADER_WORDS
                && table[0] == GpuTlaSuccessorIR.RELATION_PACKAGE_MAGIC;
    }

    private static boolean hasSuccessorOp(final GpuTlaSuccessorIR ir, final int expected) {
        final long[] ops = ir.opTable();
        for (int i = 0; i < ops.length; i += 3) if (ops[i] == expected) return true;
        return false;
    }

    private static boolean hasOrderedNextEnums(final GpuTlaSuccessorIR ir,
            final int firstVariable, final int dependentVariable) {
        final long[] ops = ir.opTable();
        for (int i = 0; i + 5 < ops.length; i += 3) {
            if (ops[i] == GpuTlaSuccessorIR.OP_ENUM && ops[i + 1] == firstVariable
                    && ops[i + 3] == GpuTlaSuccessorIR.OP_ENUM_NESTED
                    && ops[i + 4] == dependentVariable) return true;
        }
        return false;
    }

    private static boolean hasExprOp(final GpuTlaSuccessorIR ir, final int expected) {
        final long[] program = ir.exprProgram();
        for (int i = 0; i < program.length; i += 2) if (program[i] == expected) return true;
        return false;
    }
}
