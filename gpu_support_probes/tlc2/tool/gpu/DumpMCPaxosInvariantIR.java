package tlc2.tool.gpu;

import java.lang.reflect.Field;
import java.lang.reflect.Modifier;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import tlc2.tool.Action;
import tlc2.tool.StateVec;
import tlc2.tool.TLCState;
import tlc2.tool.impl.FastTool;
import tlc2.tool.impl.Tool;
import util.SimpleFilenameToStream;
import util.ToolIO;

public final class DumpMCPaxosInvariantIR {
    public static void main(final String[] args) throws Exception {
        final Path root = Paths.get(args.length == 0 ? "." : args[0]).toAbsolutePath();
        ToolIO.setUserDir(root.toString());
        final FastTool tool = new FastTool("MCPaxos", "MCPaxos",
                new SimpleFilenameToStream(new String[] { root.toString() }), Tool.Mode.MC);
        final StateVec initial = tool.getInitStates();
        TLCState state = initial.elementAt(0).copy();
        final int requestedDepth = args.length < 2 ? 0 : Integer.parseInt(args[1]);
        Map<String, TLCState> layer = new LinkedHashMap<String, TLCState>();
        layer.put(state.toString(), state);
        for (int depth = 0; depth < requestedDepth; depth++) {
            final Map<String, TLCState> next = new LinkedHashMap<String, TLCState>();
            for (TLCState source : layer.values()) {
                for (Action action : tool.getActions()) {
                    final StateVec successors = tool.getNextStates(action, source);
                    for (int i = 0; i < successors.size(); i++) {
                        final TLCState successor = successors.elementAt(i).deepCopy();
                        next.putIfAbsent(successor.toString(), successor);
                    }
                }
            }
            layer = next;
            System.out.printf("cpu depth=%d states=%d%n", depth + 1, layer.size());
        }
        if (!layer.isEmpty()) {
            state = layer.values().iterator().next();
            for (TLCState candidate : layer.values()) {
                if (candidate.toString().contains("type |-> \"2b\"")) {
                    state = candidate;
                    break;
                }
            }
        }
        System.out.println("sample state:\n" + state);
        final GpuTlaSuccessorIR ir = GpuTlaSuccessorCompiler.compileActions(
                tool, initial.elementAt(0), tool.getActions(), tool.getInvariants());
        if (requestedDepth > 0) {
            final GpuTlaIR invariant = GpuTlaIRCompiler.compile(tool,
                    tool.getInvariants()[1].con, state.getVars(), tool.getInvariants()[1].pred);
            final ValueEncoder.EncodedSemanticState encoded = ValueEncoder.encodeSemanticState(state);
            final GpuTlaEvaluator.Result result = GpuTlaEvaluator.evaluate(invariant, encoded, encoded);
            System.out.printf("gpu invariant status=%d bool=%s offset=%d%n",
                    result.status, result.boolValue, result.resultOffset);
        }
        final int expr = ir.invariantExprFirst() + 1;
        final int start = (int) ir.exprStarts()[expr];
        final int end = start + (int) ir.exprLens()[expr];
        final long[] program = ir.exprProgram();
        final Map<Integer, String> names = opcodeNames();
        System.out.printf("expr=%d start=%d end=%d instructions=%d%n",
                expr, start, end, (end - start) / 2);
        for (int pc = start; pc < end; pc += 2) {
            final int opcode = (int) program[pc];
            System.out.printf("%5d rel=%5d  %-24s %d%n",
                    pc, pc - start, names.getOrDefault(opcode, "OP_" + opcode), program[pc + 1]);
        }
    }

    private static Map<Integer, String> opcodeNames() throws IllegalAccessException {
        final Map<Integer, String> names = new HashMap<Integer, String>();
        for (Field field : GpuTlaIR.class.getDeclaredFields()) {
            if (Modifier.isStatic(field.getModifiers()) && field.getType() == int.class
                    && field.getName().startsWith("OP_")) {
                names.put(field.getInt(null), field.getName());
            }
        }
        return names;
    }
}
