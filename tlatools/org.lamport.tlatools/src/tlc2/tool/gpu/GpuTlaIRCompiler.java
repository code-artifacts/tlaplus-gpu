package tlc2.tool.gpu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import tla2sany.semantic.ASTConstants;
import tla2sany.semantic.AtNode;
import tla2sany.semantic.ExprNode;
import tla2sany.semantic.LetInNode;
import tla2sany.semantic.ExprOrOpArgNode;
import tla2sany.semantic.FormalParamNode;
import tla2sany.semantic.OpApplNode;
import tla2sany.semantic.OpDeclNode;
import tla2sany.semantic.OpDefNode;
import tla2sany.semantic.SemanticNode;
import tla2sany.semantic.SymbolNode;
import tlc2.tool.Action;
import tlc2.tool.BuiltInOPs;
import tlc2.tool.ITool;
import tlc2.tool.TLCState;
import tlc2.tool.ToolGlobals;
import tlc2.util.Context;
import tlc2.value.IValue;
import tlc2.value.impl.BoolValue;
import tlc2.value.impl.Enumerable;
import tlc2.value.impl.IntValue;
import tlc2.value.impl.LazyValue;
import tlc2.value.impl.StringValue;
import tlc2.value.impl.Value;
import tlc2.value.impl.ValueEnumeration;
import util.UniqueString;

public final class GpuTlaIRCompiler implements ToolGlobals {
    private final ITool tool;
    private final Context context;
    private final OpDeclNode[] vars;
    private final String[] varNames;
    private final Map<UniqueString, Integer> varIndex = new HashMap<UniqueString, Integer>();
    private final GpuTlaIR.Builder builder;
    private final GpuTlaIR.LocalSlots locals;
    private final Map<SymbolNode, OpDefNode> localOps;

    private GpuTlaIRCompiler(final ITool tool, final Context context, final OpDeclNode[] vars) {
        this(tool, context, vars, null, new GpuTlaIR.LocalSlots(), new HashMap<SymbolNode, OpDefNode>());
    }

    private GpuTlaIRCompiler(final ITool tool, final Context context, final OpDeclNode[] vars,
            final GpuTlaIR.Builder builder, final GpuTlaIR.LocalSlots locals,
            final Map<SymbolNode, OpDefNode> localOps) {
        this.tool = tool;
        this.context = context;
        this.vars = vars;
        this.varNames = new String[vars.length];
        for (int i = 0; i < vars.length; i++) {
            final UniqueString name = vars[i].getName();
            varNames[i] = name.toString();
            varIndex.put(name, Integer.valueOf(i));
        }
        this.locals = locals;
        this.localOps = localOps;
        this.builder = builder != null ? builder : new GpuTlaIR.Builder(varNames, locals);
    }

    public static GpuTlaIR compileAction(final ITool tool, final TLCState prototype, final Action action) {
        return compile(tool, action.con, prototype.getVars(), action.pred);
    }

    public static GpuTlaIR compile(final ITool tool, final Context context, final OpDeclNode[] vars,
            final SemanticNode expr) {
        return compile(tool, context, vars, expr, new GpuTlaIR.LocalSlots());
    }

    static GpuTlaIR compile(final ITool tool, final Context context, final OpDeclNode[] vars,
            final SemanticNode expr, final GpuTlaIR.LocalSlots locals) {
        final GpuTlaIRCompiler compiler = new GpuTlaIRCompiler(tool, context, vars, null, locals,
                new HashMap<SymbolNode, OpDefNode>());
        compiler.compileExpr(expr);
        compiler.builder.emit(GpuTlaIR.OP_RETURN);
        return compiler.builder.build();
    }

    static boolean isDirectOperator(final SymbolNode op) {
        final String name = op.getName().toString();
        return "/=".equals(name)
                || "\\notin".equals(name)
                || "\\subseteq".equals(name)
                || "=>".equals(name)
                || "\\equiv".equals(name)
                || "\\lnot".equals(name)
                || "~".equals(name)
                || "+".equals(name)
                || "-".equals(name)
                || "*".equals(name)
                || "<".equals(name)
                || "\\leq".equals(name)
                || "<=".equals(name)
                || ">".equals(name)
                || "\\geq".equals(name)
                || ">=".equals(name)
                || "..".equals(name)
                || "DOMAIN".equals(name)
                || "UNION".equals(name)
                || "SUBSET".equals(name);
    }

    private void compileExpr(final SemanticNode expr) {
        if (expr == null) {
            throw unsupported("null expression");
        }
        if (expr.getKind() == ASTConstants.OpApplKind) {
            compileOpAppl((OpApplNode) expr);
            return;
        }
        if (expr.getKind() == ASTConstants.LetInKind) {
            compileLet((LetInNode) expr);
            return;
        }
        if (expr.getKind() == ASTConstants.AtNodeKind) {
            compileAt((AtNode) expr);
            return;
        }
        compileConstant(expr);
    }

    private void compileOpAppl(final OpApplNode node) {
        final SymbolNode op = node.getOperator();
        final ExprOrOpArgNode[] args = node.getArgs();
        if (op.getKind() == ASTConstants.FormalParamKind) {
            final Object bound = context.lookup(op);
            if (bound instanceof LazyValue) {
                final LazyValue lazy = (LazyValue) bound;
                final GpuTlaIRCompiler nested = new GpuTlaIRCompiler(tool, lazy.con, vars, builder, locals, localOps);
                nested.compileExpr(lazy.expr);
            } else if (bound instanceof IValue) {
                emitConstant((IValue) bound);
            } else if (locals.contains(op)) {
                builder.emit(GpuTlaIR.OP_LOAD_LOCAL, locals.slot(op));
            } else {
                throw unsupported("unbound formal parameter: " + op + " identity=" + System.identityHashCode(op) + " localSlots=" + locals.size());
            }
            return;
        }
        if (op.getKind() == ASTConstants.VariableDeclKind) {
            builder.emit(GpuTlaIR.OP_LOAD_VAR, varIndex(op.getName()));
            if (args.length == 0) {
                return;
            }
            if (args.length != 1) {
                throw unsupported("function application currently supports one argument: " + node);
            }
            compileExpr(args[0]);
            builder.emit(GpuTlaIR.OP_FCN_APPLY);
            return;
        }

        final int opcode = BuiltInOPs.getOpCode(op.getName());
        final String opName = op.getName().toString();
        if ("/=".equals(opName)) {
            requireArity(args, 2, "/=");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_NEQ);
            return;
        }
        if ("\\notin".equals(opName)) {
            requireArity(args, 2, "notin");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_NOT_MEMBER);
            return;
        }
        if ("\\subseteq".equals(opName)) {
            requireArity(args, 2, "subseteq");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_SUBSETEQ);
            return;
        }
        if ("=>".equals(opName)) {
            requireArity(args, 2, "=>");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_IMPLIES);
            return;
        }
        if ("\\equiv".equals(opName)) {
            requireArity(args, 2, "equiv");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_EQUIV);
            return;
        }
        if ("\\lnot".equals(opName) || "~".equals(opName)) {
            requireArity(args, 1, "~");
            compileExpr(args[0]);
            builder.emit(GpuTlaIR.OP_NOT);
            return;
        }
        if ("+".equals(opName)) {
            compileBinary(args, GpuTlaIR.OP_ADD, "+");
            return;
        }
        if ("-".equals(opName)) {
            if (args.length == 1) {
                compileExpr(args[0]);
                builder.emit(GpuTlaIR.OP_NEG);
            } else {
                compileBinary(args, GpuTlaIR.OP_SUB, "-");
            }
            return;
        }
        if ("*".equals(opName)) {
            compileBinary(args, GpuTlaIR.OP_MUL, "*");
            return;
        }
        if ("<".equals(opName)) {
            compileBinary(args, GpuTlaIR.OP_LT, "<");
            return;
        }
        if ("\\leq".equals(opName) || "<=".equals(opName)) {
            compileBinary(args, GpuTlaIR.OP_LE, "<=");
            return;
        }
        if (">".equals(opName)) {
            compileBinary(args, GpuTlaIR.OP_GT, ">");
            return;
        }
        if ("\\geq".equals(opName) || ">=".equals(opName)) {
            compileBinary(args, GpuTlaIR.OP_GE, ">=");
            return;
        }
        if ("..".equals(opName)) {
            compileBinary(args, GpuTlaIR.OP_INTERVAL, "..");
            return;
        }
        if ("DOMAIN".equals(opName)) {
            requireArity(args, 1, "DOMAIN");
            compileExpr(args[0]);
            builder.emit(GpuTlaIR.OP_DOMAIN);
            return;
        }
        if ("UNION".equals(opName)) {
            requireArity(args, 1, "UNION");
            compileExpr(args[0]);
            builder.emit(GpuTlaIR.OP_UNION);
            return;
        }
        if ("SUBSET".equals(opName)) {
            requireArity(args, 1, "SUBSET");
            compileExpr(args[0]);
            builder.emit(GpuTlaIR.OP_SUBSET);
            return;
        }
        if (op.getKind() == ASTConstants.UserDefinedOpKind) {
            final OpDefNode opDef = (OpDefNode) op;
            final OpDefNode lexicalDef = localOps.get(op);
            final Context opContext = tool.getOpContext(opDef, args, context, true);
            final GpuTlaIRCompiler nested = new GpuTlaIRCompiler(tool, opContext, vars, builder, locals, localOps);
            nested.compileExpr(lexicalDef == null ? opDef.getBody() : lexicalDef.getBody());
            return;
        }
        switch (opcode) {
        case OPCODE_bc:
            compileBoundedChoose(node);
            return;
        case OPCODE_uc:
            compileUnboundedChoose(node);
            return;
        case OPCODE_cup:
            compileBinary(args, GpuTlaIR.OP_SET_UNION, "\\union");
            return;
        case OPCODE_cap:
            compileBinary(args, GpuTlaIR.OP_SET_INTERSECT, "\\intersect");
            return;
        case OPCODE_setdiff:
            compileBinary(args, GpuTlaIR.OP_SET_DIFF, "\\");
            return;
        case OPCODE_tup:
            compileTuple(args);
            return;
        case OPCODE_cp:
            compileCartesian(args);
            return;
        case OPCODE_fc:
            compileFunctionConstructor(node);
            return;
        case OPCODE_sso:
            compileDynamicCollection(node, false);
            return;
        case OPCODE_soa:
            compileDynamicCollection(node, true);
            return;
        case OPCODE_se:
            compileSetEnum(args);
            return;
        case OPCODE_sor:
            compileSetOfRecords(args);
            return;
        case OPCODE_sof:
            compileSetOfFunctions(args);
            return;
        case OPCODE_rc:
            compileRecordConstructor(args);
            return;
        case OPCODE_rs:
            compileRecordSelect(args);
            return;
        case OPCODE_exc:
            compileExcept(args);
            return;
        case OPCODE_prime:
            compilePrime(args);
            return;
        case OPCODE_eq:
            requireArity(args, 2, "=");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_EQ);
            return;
        case OPCODE_in:
            requireArity(args, 2, "\\in");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_MEMBER);
            return;
        case OPCODE_fa:
            requireArity(args, 2, "function application");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_FCN_APPLY);
            return;
        case OPCODE_ite:
            compileIfThenElse(args);
            return;
        case OPCODE_case:
            compileCase(args);
            return;
        case OPCODE_land:
        case OPCODE_cl:
            compileShortCircuit(args, true);
            return;
        case OPCODE_lor:
        case OPCODE_dl:
            compileShortCircuit(args, false);
            return;
        case OPCODE_lnot:
            requireArity(args, 1, "~");
            compileExpr(args[0]);
            builder.emit(GpuTlaIR.OP_NOT);
            return;
        case OPCODE_implies:
            requireArity(args, 2, "=>");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_IMPLIES);
            return;
        case OPCODE_equiv:
            requireArity(args, 2, "equiv");
            compileExpr(args[0]);
            compileExpr(args[1]);
            builder.emit(GpuTlaIR.OP_EQUIV);
            return;
        case OPCODE_be:
            requireArity(args, 1, "BoundedExists");
            compileBoundedQuantifier(node, true);
            return;
        case OPCODE_bf:
            requireArity(args, 1, "BoundedForall");
            compileBoundedQuantifier(node, false);
            return;
        case OPCODE_unchanged:
            requireArity(args, 1, "UNCHANGED");
            compileUnchanged(args[0]);
            return;
        default:
            compileConstant(node);
            return;
        }
    }

    private void compileIfThenElse(final ExprOrOpArgNode[] args) {
        requireArity(args, 3, "IF THEN ELSE");
        compileExpr(args[0]);
        final int jumpToElse = builder.emitJump(GpuTlaIR.OP_JUMP_IF_FALSE);
        compileExpr(args[1]);
        final int jumpToEnd = builder.emitJump(GpuTlaIR.OP_JUMP);
        final int elseOffset = builder.currentOffset();
        builder.patchOperand(jumpToElse, elseOffset);
        compileExpr(args[2]);
        builder.patchOperand(jumpToEnd, builder.currentOffset());
    }

    private void compileCase(final ExprOrOpArgNode[] args) {
        if (args.length == 0) {
            throw unsupported("CASE without arms");
        }
        final List<Integer> jumpsToEnd = new ArrayList<Integer>();
        for (int i = 0; i < args.length; i++) {
            if (args[i].getKind() != ASTConstants.OpApplKind) {
                throw unsupported("CASE arm is not a pair: " + args[i]);
            }
            final OpApplNode pair = (OpApplNode) args[i];
            if (BuiltInOPs.getOpCode(pair.getOperator().getName()) != OPCODE_pair) {
                throw unsupported("CASE arm is not a pair: " + args[i]);
            }
            final ExprOrOpArgNode[] pairArgs = pair.getArgs();
            requireArity(pairArgs, 2, "CASE arm");
            if (pairArgs[0] == null) {
                if (i != args.length - 1) {
                    throw unsupported("CASE OTHER arm must be last");
                }
                compileExpr(pairArgs[1]);
                break;
            }
            compileExpr(pairArgs[0]);
            final int jumpToNextArm = builder.emitJump(GpuTlaIR.OP_JUMP_IF_FALSE);
            compileExpr(pairArgs[1]);
            jumpsToEnd.add(Integer.valueOf(builder.emitJump(GpuTlaIR.OP_JUMP)));
            builder.patchOperand(jumpToNextArm, builder.currentOffset());
        }
        final int endOffset = builder.currentOffset();
        for (Integer jump : jumpsToEnd) {
            builder.patchOperand(jump.intValue(), endOffset);
        }
    }

    private void compilePrime(final ExprOrOpArgNode[] args) {
        requireArity(args, 1, "prime");
        if (args[0].getKind() != ASTConstants.OpApplKind) {
            throw unsupported("prime of non-variable expression: " + args[0]);
        }
        final OpApplNode var = (OpApplNode) args[0];
        if (var.getOperator().getKind() != ASTConstants.VariableDeclKind) {
            throw unsupported("prime currently supports variables only: " + args[0]);
        }
        builder.emit(GpuTlaIR.OP_LOAD_NEXT_VAR, varIndex(var.getOperator().getName()));
    }

    private void compileUnchanged(final SemanticNode expr) {
        if (expr.getKind() == ASTConstants.OpApplKind) {
            final OpApplNode appl = (OpApplNode) expr;
            final int opcode = BuiltInOPs.getOpCode(appl.getOperator().getName());
            if (opcode == OPCODE_tup) {
                final ExprOrOpArgNode[] args = appl.getArgs();
                compileUnchangedList(args);
                return;
            }
            if (appl.getOperator().getKind() == ASTConstants.VariableDeclKind) {
                compileUnchangedVar(appl.getOperator().getName());
                return;
            }
        }
        throw unsupported("UNCHANGED currently supports variables or tuples of variables only: " + expr);
    }

    private void compileUnchangedList(final ExprOrOpArgNode[] args) {
        if (args.length == 0) {
            emitConstant(BoolValue.ValTrue);
            return;
        }
        compileUnchanged(args[0]);
        for (int i = 1; i < args.length; i++) {
            compileUnchanged(args[i]);
            builder.emit(GpuTlaIR.OP_AND);
        }
    }

    private void compileUnchangedVar(final UniqueString name) {
        final int index = varIndex(name);
        builder.emit(GpuTlaIR.OP_LOAD_VAR, index);
        builder.emit(GpuTlaIR.OP_LOAD_NEXT_VAR, index);
        builder.emit(GpuTlaIR.OP_EQ);
    }

    private void compileBinary(final ExprOrOpArgNode[] args, final int op, final String name) {
        requireArity(args, 2, name);
        compileExpr(args[0]);
        compileExpr(args[1]);
        builder.emit(op);
    }

    private void compileLet(final LetInNode node) {
        final Map<SymbolNode, OpDefNode> defs = new HashMap<SymbolNode, OpDefNode>(localOps);
        for (OpDefNode def : node.getLets()) defs.put(def, def);
        final GpuTlaIRCompiler nested = new GpuTlaIRCompiler(tool, context, vars, builder, locals, defs);
        nested.compileExpr(node.getBody());
    }

    private void compileAt(final AtNode node) {
        builder.emit(GpuTlaIR.OP_LOAD_LOCAL, locals.slot(node.getExceptComponentRef()));
    }

    private void compileTuple(final ExprOrOpArgNode[] args) {
        for (ExprOrOpArgNode arg : args) compileExpr(arg);
        builder.emit(GpuTlaIR.OP_TUPLE, args.length);
    }

    private void compileCartesian(final ExprOrOpArgNode[] args) {
        if (args.length == 0) {
            throw unsupported("Cartesian product requires at least one set");
        }
        for (ExprOrOpArgNode arg : args) compileExpr(arg);
        builder.emit(GpuTlaIR.OP_CARTESIAN, args.length);
    }

    private void compileBoundedChoose(final OpApplNode node) {
        final FormalParamNode[][] formals = node.getBdedQuantSymbolLists();
        final boolean[] tupleFlags = node.isBdedQuantATuple();
        final ExprNode[] bounds = node.getBdedQuantBounds();
        if (formals.length != 1 || tupleFlags.length != 1 || bounds.length != 1
                || formals[0].length == 0 || node.getArgs().length != 1) {
            throw unsupported("malformed bounded CHOOSE: " + node);
        }
        if (!tupleFlags[0] && formals[0].length != 1) {
            throw unsupported("scalar bounded CHOOSE has more than one formal: " + node);
        }
        final int iterationSlot = tupleFlags[0] ? locals.slot(new Object()) : locals.slot(formals[0][0]);
        compileExpr(bounds[0]);
        final int begin = builder.emitJump(GpuTlaIR.OP_CHOOSE_BEGIN);
        final int bodyStart = builder.currentOffset();
        if (tupleFlags[0]) bindTupleFormals(iterationSlot, formals[0]);
        compileExpr(node.getArgs()[0]);
        final int next = builder.emitJump(GpuTlaIR.OP_CHOOSE_NEXT);
        final int end = builder.currentOffset();
        builder.patchOperand(begin, packLoop(iterationSlot, end));
        builder.patchOperand(next, bodyStart);
    }

    private void compileUnboundedChoose(final OpApplNode node) {
        final FormalParamNode[] formals = node.getUnbdedQuantSymbols();
        if (formals == null || formals.length != 1 || node.getArgs().length != 1) {
            throw unsupported("unbounded CHOOSE currently requires one scalar formal: " + node);
        }
        final int slot = locals.slot(formals[0]);
        final int candidate = builder.addSyntheticOpaque("tla.unbounded.choose", node.getUid());
        builder.emit(GpuTlaIR.OP_LOAD_CONST, candidate);
        builder.emit(GpuTlaIR.OP_SAVE_LOCAL, slot);
        compileExpr(node.getArgs()[0]);
        builder.emit(GpuTlaIR.OP_CHOOSE_CHECK, slot);
    }

    private void compileFunctionConstructor(final OpApplNode node) {
        final FormalParamNode[][] formals = node.getBdedQuantSymbolLists();
        final boolean[] tupleFlags = node.isBdedQuantATuple();
        final ExprNode[] bounds = node.getBdedQuantBounds();
        if (formals.length != 1 || tupleFlags.length != 1 || bounds.length != 1
                || formals[0].length == 0 || node.getArgs().length != 1) {
            throw unsupported("runtime function constructor currently supports one bound: " + node);
        }
        if (!tupleFlags[0] && formals[0].length != 1) {
            throw unsupported("scalar function constructor has more than one formal: " + node);
        }
        final int iterationSlot = tupleFlags[0] ? locals.slot(new Object()) : locals.slot(formals[0][0]);
        compileExpr(bounds[0]);
        final int begin = builder.emitJump(GpuTlaIR.OP_FCN_BEGIN);
        final int bodyStart = builder.currentOffset();
        if (tupleFlags[0]) bindTupleFormals(iterationSlot, formals[0]);
        compileExpr(node.getArgs()[0]);
        final int next = builder.emitJump(GpuTlaIR.OP_FCN_NEXT);
        final int end = builder.currentOffset();
        builder.patchOperand(begin, packLoop(iterationSlot, end));
        builder.patchOperand(next, bodyStart);
    }

    private void bindTupleFormals(final int tupleSlot, final FormalParamNode[] formals) {
        for (int i = 0; i < formals.length; i++) {
            builder.emit(GpuTlaIR.OP_LOAD_LOCAL, tupleSlot);
            emitConstant(IntValue.gen(i + 1));
            builder.emit(GpuTlaIR.OP_FCN_APPLY);
            builder.emit(GpuTlaIR.OP_SAVE_LOCAL, locals.slot(formals[i]));
        }
    }

    private void compileDynamicCollection(final OpApplNode node, final boolean map) {
        final FormalParamNode[][] formals = node.getBdedQuantSymbolLists();
        final boolean[] tupleFlags = node.isBdedQuantATuple();
        final ExprNode[] bounds = node.getBdedQuantBounds();
        if (formals.length != 1 || tupleFlags.length != 1 || bounds.length != 1
                || tupleFlags[0] || formals[0].length != 1 || node.getArgs().length != 1) {
            throw unsupported("runtime set operator currently supports one scalar bound: " + node);
        }
        final int slot = locals.slot(formals[0][0]);
        compileExpr(bounds[0]);
        final int begin = builder.emitJump(map ? GpuTlaIR.OP_MAP_BEGIN : GpuTlaIR.OP_FILTER_BEGIN);
        final int bodyStart = builder.currentOffset();
        compileExpr(node.getArgs()[0]);
        final int next = builder.emitJump(map ? GpuTlaIR.OP_MAP_NEXT : GpuTlaIR.OP_FILTER_NEXT);
        final int end = builder.currentOffset();
        builder.patchOperand(begin, packLoop(slot, end));
        builder.patchOperand(next, bodyStart);
    }

    private static long packLoop(final int slot, final int target) {
        return ((long) slot << 32) | (target & 0xffffffffL);
    }

    private void compileSetEnum(final ExprOrOpArgNode[] args) {
        for (ExprOrOpArgNode arg : args) {
            compileExpr(arg);
        }
        builder.emit(GpuTlaIR.OP_SET_ENUM, args.length);
    }

    private void compileSetOfRecords(final ExprOrOpArgNode[] args) {
        if (args.length == 0) {
            throw unsupported("empty record set constructor is not in the current GPU subset");
        }
        for (int i = 0; i < args.length; i++) {
            if (args[i].getKind() != ASTConstants.OpApplKind) {
                throw unsupported("record set field is not a pair: " + args[i]);
            }
            final OpApplNode pairNode = (OpApplNode) args[i];
            if (BuiltInOPs.getOpCode(pairNode.getOperator().getName()) != OPCODE_pair) {
                throw unsupported("record set field is not a pair: " + args[i]);
            }
            final ExprOrOpArgNode[] pair = pairNode.getArgs();
            requireArity(pair, 2, "record set field");
            final long field = fieldId(pair[0]);
            compileExpr(pair[1]);
            builder.emit(i == 0 ? GpuTlaIR.OP_RECORD_SET_ONE : GpuTlaIR.OP_RECORD_SET_APPEND, field);
        }
    }

    private void compileSetOfFunctions(final ExprOrOpArgNode[] args) {
        requireArity(args, 2, "set of functions");
        compileExpr(args[0]);
        compileExpr(args[1]);
        builder.emit(GpuTlaIR.OP_SET_OF_FCNS);
    }

    private void compileRecordConstructor(final ExprOrOpArgNode[] args) {
        if (args.length == 0) {
            throw unsupported("empty record constructor is not in the current GPU subset");
        }
        for (int i = 0; i < args.length; i++) {
            if (args[i].getKind() != ASTConstants.OpApplKind) {
                throw unsupported("record constructor field is not a pair: " + args[i]);
            }
            final OpApplNode pairNode = (OpApplNode) args[i];
            if (BuiltInOPs.getOpCode(pairNode.getOperator().getName()) != OPCODE_pair) {
                throw unsupported("record constructor field is not a pair: " + args[i]);
            }
            final ExprOrOpArgNode[] pair = pairNode.getArgs();
            requireArity(pair, 2, "record field");
            final long fieldId = fieldId(pair[0]);
            compileExpr(pair[1]);
            builder.emit(i == 0 ? GpuTlaIR.OP_RECORD_ONE : GpuTlaIR.OP_RECORD_APPEND, fieldId);
        }
    }

    private void compileRecordSelect(final ExprOrOpArgNode[] args) {
        requireArity(args, 2, "record select");
        compileExpr(args[0]);
        builder.emit(GpuTlaIR.OP_RECORD_SELECT, fieldId(args[1]));
    }

    private void compileExcept(final ExprOrOpArgNode[] args) {
        if (args.length < 2) {
            throw unsupported("EXCEPT without updates");
        }
        compileExpr(args[0]);
        for (int i = 1; i < args.length; i++) {
            if (args[i].getKind() != ASTConstants.OpApplKind) {
                throw unsupported("EXCEPT update is not a pair: " + args[i]);
            }
            final OpApplNode pairNode = (OpApplNode) args[i];
            final ExprOrOpArgNode[] pairArgs = pairNode.getArgs();
            requireArity(pairArgs, 2, "EXCEPT update");
            if (pairArgs[0].getKind() != ASTConstants.OpApplKind) {
                throw unsupported("EXCEPT path is not an application: " + pairArgs[0]);
            }
            final ExprOrOpArgNode[] path = ((OpApplNode) pairArgs[0]).getArgs();
            if (path.length == 0) throw unsupported("EXCEPT path is empty");
            for (ExprOrOpArgNode pathElem : path) compileExpr(pathElem);
            final int atSlot = locals.slot(pairNode);
            builder.emit(GpuTlaIR.OP_EXCEPT_SAVE_AT, packLoop(atSlot, path.length));
            compileExpr(pairArgs[1]);
            builder.emit(GpuTlaIR.OP_EXCEPT_UPDATE, path.length);
        }
    }

    private long fieldId(final SemanticNode node) {
        final Object value = tool.eval(node, context);
        if (!(value instanceof StringValue)) {
            throw unsupported("record field name is not a string literal: " + node);
        }
        return ValueEncoder.stableStringId(((StringValue) value).getVal().toString());
    }

    private void compileBoundedQuantifier(final OpApplNode node, final boolean isExists) {
        final FormalParamNode[][] formals = node.getBdedQuantSymbolLists();
        final boolean[] tupleFlags = node.isBdedQuantATuple();
        final ExprNode[] bounds = node.getBdedQuantBounds();
        if (formals.length == 0 || formals.length != tupleFlags.length
                || formals.length != bounds.length || node.getArgs().length != 1) {
            throw unsupported("malformed bounded quantifier: " + node);
        }
        for (FormalParamNode[] boundFormals : formals) {
            if (boundFormals.length == 0) throw unsupported("bounded quantifier has no formals: " + node);
        }
        if (formals.length == 1 && !tupleFlags[0] && formals[0].length == 1
                && !hasRuntimeBinding(bounds[0])) {
            final Object boundValue = tool.eval(bounds[0], context);
            if (boundValue instanceof Enumerable) {
                final java.util.List<Value> values = ((Enumerable) boundValue).elements().all();
                final FormalParamNode formal = formals[0][0];
                final SemanticNode body = node.getArgs()[0];
                if (values.isEmpty()) {
                    emitConstant(isExists ? BoolValue.ValFalse : BoolValue.ValTrue);
                    return;
                }
                for (int i = 0; i < values.size(); i++) {
                    final Context boundContext = context.cons(formal, values.get(i));
                    final GpuTlaIRCompiler nested = new GpuTlaIRCompiler(tool, boundContext, vars, builder,
                            locals, localOps);
                    nested.compileExpr(body);
                    if (i > 0) builder.emit(isExists ? GpuTlaIR.OP_OR : GpuTlaIR.OP_AND);
                }
                return;
            }
        }
        compileRuntimeQuantifier(node, isExists);
    }

    private void compileRuntimeQuantifier(final OpApplNode node, final boolean isExists) {
        compileRuntimeQuantifierBinding(node, isExists, 0, 0);
    }

    private void compileRuntimeQuantifierBinding(final OpApplNode node, final boolean isExists,
            final int boundIndex, final int formalIndex) {
        final FormalParamNode[][] formals = node.getBdedQuantSymbolLists();
        if (boundIndex >= formals.length) {
            compileExpr(node.getArgs()[0]);
            return;
        }

        final boolean tupleBound = node.isBdedQuantATuple()[boundIndex];
        final FormalParamNode[] boundFormals = formals[boundIndex];
        final int iterationSlot;
        if (tupleBound) {
            iterationSlot = locals.slot(new Object());
        } else {
            iterationSlot = locals.slot(boundFormals[formalIndex]);
        }

        compileExpr(node.getBdedQuantBounds()[boundIndex]);
        final int begin = builder.emitJump(isExists ? GpuTlaIR.OP_EXISTS_BEGIN : GpuTlaIR.OP_FORALL_BEGIN);
        final int bodyStart = builder.currentOffset();
        if (tupleBound) bindTupleFormals(iterationSlot, boundFormals);

        final boolean moreFormalsInBound = !tupleBound && formalIndex + 1 < boundFormals.length;
        if (moreFormalsInBound) {
            compileRuntimeQuantifierBinding(node, isExists, boundIndex, formalIndex + 1);
        } else {
            compileRuntimeQuantifierBinding(node, isExists, boundIndex + 1, 0);
        }

        final int next = builder.emitJump(isExists ? GpuTlaIR.OP_EXISTS_NEXT : GpuTlaIR.OP_FORALL_NEXT);
        final int end = builder.currentOffset();
        builder.patchOperand(begin, packLoop(iterationSlot, end));
        builder.patchOperand(next, bodyStart);
    }

    private boolean hasRuntimeBinding(final SemanticNode node) {
        if (node == null) return false;
        if (node.getKind() != ASTConstants.OpApplKind) {
            return node.getKind() == ASTConstants.LetInKind || node.getKind() == ASTConstants.AtNodeKind;
        }
        final OpApplNode appl = (OpApplNode) node;
        final SymbolNode op = appl.getOperator();
        if (op.getKind() == ASTConstants.VariableDeclKind) return varIndex.containsKey(op.getName());
        if (op.getKind() == ASTConstants.FormalParamKind) return locals.contains(op) || context.lookup(op) == null;
        if (op.getKind() == ASTConstants.UserDefinedOpKind) {
            final Object bound = context.lookup(op);
            if (bound instanceof LazyValue) return hasRuntimeBinding(((LazyValue) bound).expr);
            final OpDefNode def = (OpDefNode) op;
            return hasRuntimeBinding(def.getBody());
        }
        for (ExprOrOpArgNode arg : appl.getArgs()) if (hasRuntimeBinding(arg)) return true;
        return false;
    }

    private void compileShortCircuit(final ExprOrOpArgNode[] args, final boolean conjunction) {
        if (args.length == 0) {
            emitConstant(conjunction ? BoolValue.ValTrue : BoolValue.ValFalse);
            return;
        }
        if (conjunction) {
            final List<Integer> falseJumps = new ArrayList<Integer>();
            for (int i = 0; i < args.length - 1; i++) {
                compileExpr(args[i]);
                falseJumps.add(Integer.valueOf(builder.emitJump(GpuTlaIR.OP_JUMP_IF_FALSE)));
            }
            compileExpr(args[args.length - 1]);
            final int skipFalse = builder.emitJump(GpuTlaIR.OP_JUMP);
            final int falseOffset = builder.currentOffset();
            emitConstant(BoolValue.ValFalse);
            final int endOffset = builder.currentOffset();
            for (Integer jump : falseJumps) builder.patchOperand(jump.intValue(), falseOffset);
            builder.patchOperand(skipFalse, endOffset);
            return;
        }

        final List<Integer> endJumps = new ArrayList<Integer>();
        for (int i = 0; i < args.length - 1; i++) {
            compileExpr(args[i]);
            final int falsePath = builder.emitJump(GpuTlaIR.OP_JUMP_IF_FALSE);
            emitConstant(BoolValue.ValTrue);
            endJumps.add(Integer.valueOf(builder.emitJump(GpuTlaIR.OP_JUMP)));
            builder.patchOperand(falsePath, builder.currentOffset());
        }
        compileExpr(args[args.length - 1]);
        final int endOffset = builder.currentOffset();
        for (Integer jump : endJumps) builder.patchOperand(jump.intValue(), endOffset);
    }

    private void compileConstant(final SemanticNode expr) {
        if (expr instanceof ExprOrOpArgNode && ((ExprOrOpArgNode) expr).getLevel() > 0) {
            throw unsupported("unsupported non-constant expression for first GPU IR compiler: " + expr);
        }
        emitConstant(tool.eval(expr, context));
    }

    private void emitConstant(final IValue value) {
        builder.emit(GpuTlaIR.OP_LOAD_CONST, builder.addConstant(value));
    }

    private int varIndex(final UniqueString name) {
        final Integer index = varIndex.get(name);
        if (index == null) {
            throw unsupported("unknown state variable: " + name);
        }
        return index.intValue();
    }

    private static void requireArity(final ExprOrOpArgNode[] args, final int expected, final String op) {
        if (args.length != expected) {
            throw new IllegalArgumentException(op + " expects " + expected + " arguments, found " + args.length);
        }
    }

    private static UnsupportedOperationException unsupported(final String message) {
        return new UnsupportedOperationException("GPU TLA+ IR compiler: " + message);
    }
}
