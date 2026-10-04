package tlc2.tool.gpu;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Test;

import tlc2.tool.StateVec;
import tlc2.tool.impl.FastTool;
import tlc2.tool.impl.Tool;
import util.SimpleFilenameToStream;
import util.ToolIO;

public class GpuMCPaxosInvariantCompilerTest {

    @Test
    public void compilesAllMCPaxosInvariantsForDeviceChecking() throws Exception {
        final Path models = findRepositoryRoot();
        ToolIO.setUserDir(models.toString());
        final FastTool tool = new FastTool("MCPaxos", "MCPaxos",
                new SimpleFilenameToStream(new String[] { models.toString() }), Tool.Mode.MC);
        final StateVec initial = tool.getInitStates();
        assertEquals(1, initial.size());
        final GpuTlaSuccessorIR ir = GpuTlaSuccessorCompiler.compileActions(
                tool, initial.elementAt(0), tool.getActions(), tool.getInvariants());

        assertFalse(isRelationPackage(ir));
        assertFalse(hasSuccessorOp(ir, GpuTlaSuccessorIR.OP_RELATION_EXISTS));
        assertArrayEquals(new int[] { 0, 1 }, ir.invariantOrdinals());
        assertTrue(ir.invariantExprFirst() >= 0);
        final int typeOk = ir.invariantExprFirst();
        assertTrue(hasExprOp(ir, typeOk, GpuTlaIR.OP_FCN_SET_MEMBER));
        assertTrue(hasExprOp(ir, typeOk, GpuTlaIR.OP_RECORD_ARITY_EQ));
        assertTrue(hasExprOp(ir, typeOk, GpuTlaIR.OP_RECORD_FIELD_MEMBER));
        assertFalse(hasExprOp(ir, typeOk, GpuTlaIR.OP_SET_OF_FCNS));
        assertFalse(hasExprOp(ir, typeOk, GpuTlaIR.OP_RECORD_SET_ONE));
        assertFalse(hasExprOp(ir, typeOk, GpuTlaIR.OP_RECORD_SET_APPEND));
        final int invariant = typeOk + 1;
        assertTrue(hasExprOp(ir, invariant, GpuTlaIR.OP_JUMP_IF_FALSE));
        assertFalse(hasExprOp(ir, invariant, GpuTlaIR.OP_IMPLIES));
    }

    private static boolean hasExprOp(final GpuTlaSuccessorIR ir, final int expr, final int expected) {
        final int start = (int) ir.exprStarts()[expr];
        final int end = start + (int) ir.exprLens()[expr];
        final long[] program = ir.exprProgram();
        for (int i = start; i < end; i += 2) {
            if (program[i] == expected) return true;
        }
        return false;
    }

    private static boolean isRelationPackage(final GpuTlaSuccessorIR ir) {
        final long[] table = ir.branchTable();
        return table.length >= GpuTlaSuccessorIR.RELATION_HEADER_WORDS
                && table[0] == GpuTlaSuccessorIR.RELATION_PACKAGE_MAGIC;
    }

    private static boolean hasSuccessorOp(final GpuTlaSuccessorIR ir, final int expected) {
        final long[] ops = ir.opTable();
        for (int i = 0; i < ops.length; i += 3) {
            if (ops[i] == expected) return true;
        }
        return false;
    }

    private static Path findRepositoryRoot() {
        Path path = Paths.get(System.getProperty("user.dir")).toAbsolutePath();
        while (path != null) {
            if (Files.isRegularFile(path.resolve("MCPaxos.tla"))
                    && Files.isRegularFile(path.resolve("MCPaxos.cfg"))) {
                return path;
            }
            path = path.getParent();
        }
        throw new IllegalStateException("cannot locate MCPaxos model from "
                + System.getProperty("user.dir"));
    }
}
