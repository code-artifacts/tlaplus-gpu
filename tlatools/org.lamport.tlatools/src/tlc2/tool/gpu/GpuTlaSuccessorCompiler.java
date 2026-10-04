package tlc2.tool.gpu;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import tla2sany.semantic.ASTConstants;
import tla2sany.semantic.ExprOrOpArgNode;
import tla2sany.semantic.FormalParamNode;
import tla2sany.semantic.LetInNode;
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
import tlc2.value.impl.Enumerable;
import tlc2.value.impl.Value;
import tlc2.value.impl.ValueEnumeration;
import util.UniqueString;

public final class GpuTlaSuccessorCompiler implements ToolGlobals {
    private final ITool tool;
    private final Context context;
    private final OpDeclNode[] vars;
    private final String[] varNames;
    private final Map<UniqueString, Integer> varIndex = new HashMap<UniqueString, Integer>();
    private final GpuTlaSuccessorIR.Builder builder;
    private static final boolean DEBUG = Boolean.getBoolean("tlc.gpu.generic.debug.actions");

    private GpuTlaSuccessorCompiler(final ITool tool, final Context context, final OpDeclNode[] vars) {
        this(tool, context, vars, null);
    }

    private GpuTlaSuccessorCompiler(final ITool tool, final Context context, final OpDeclNode[] vars,
            final GpuTlaSuccessorIR.Builder builder) {
        this.tool = tool;
        this.context = context;
        this.vars = vars;
        this.varNames = new String[vars.length];
        for (int i = 0; i < vars.length; i++) {
            final UniqueString name = vars[i].getName();
            varNames[i] = name.toString();
            varIndex.put(name, Integer.valueOf(i));
        }
        this.builder = builder != null ? builder : new GpuTlaSuccessorIR.Builder(varNames);
    }

    public static GpuTlaSuccessorIR compileAction(final ITool tool, final TLCState prototype, final Action action) {
        return compileActions(tool, prototype, new Action[] { action });
    }

    public static GpuTlaSuccessorIR compileActions(final ITool tool, final TLCState prototype, final Action[] actions) {
        final OpDeclNode[] vars = prototype.getVars();
        final GpuTlaSuccessorIR.Builder builder = new GpuTlaSuccessorIR.Builder(variableNames(vars));
        for (Action action : actions) {
            if (DEBUG) {
                System.out.println("GPU generic compile action: " + action);
            }
            final GpuTlaSuccessorCompiler compiler = new GpuTlaSuccessorCompiler(tool, action.con, vars, builder);
            compiler.compilePredicateBranches(action.pred);
        }
        return builder.build();
    }

    public static GpuTlaSuccessorIR compile(final ITool tool, final Context context, final OpDeclNode[] vars,
            final SemanticNode nextPredicate) {
        final GpuTlaSuccessorCompiler compiler = new GpuTlaSuccessorCompiler(tool, context, vars);
        compiler.compilePredicateBranches(nextPredicate);
        return compiler.builder.build();
    }

    private void compilePredicateBranches(final SemanticNode nextPredicate) {
        final List<BranchPlan> initial = new ArrayList<BranchPlan>();
        initial.add(new BranchPlan());
        final List<BranchPlan> branches = expandPredicate(nextPredicate, context, initial);
        for (BranchPlan branch : branches) {
            final int branchIndex = builder.beginBranch();
            compileBranch(branch);
            builder.endBranch(branchIndex);
        }
    }

    private static String[] variableNames(final OpDeclNode[] vars) {
        final String[] names = new String[vars.length];
        for (int i = 0; i < vars.length; i++) {
            names[i] = vars[i].getName().toString();
        }
        return names;
    }

    /**
     * Normalizes an action predicate into disjunctive branch plans before any
     * GPU instructions are emitted. Named actions are inlined, conjunction
     * builds a Cartesian product, disjunction creates alternatives, and finite
     * bounded exists quantifiers instantiate formals in the TLC context.
     */
    private List<BranchPlan> expandPredicate(final SemanticNode node, final Context lexicalContext,
            final List<BranchPlan> input) {
        if (node == null || input.isEmpty()) return input;
        // Keep action-level LET expansion consistent with TLC successor enumeration.
        if (node.getKind() == ASTConstants.LetInKind) {
            return expandPredicate(((LetInNode) node).getBody(), lexicalContext, input);
        }
        if (node.getKind() != ASTConstants.OpApplKind) return appendAtom(input, node, lexicalContext);

        final OpApplNode appl = (OpApplNode) node;
        final SymbolNode op = appl.getOperator();
        final ExprOrOpArgNode[] args = appl.getArgs();
        if (op.getKind() == ASTConstants.UserDefinedOpKind
                && !GpuTlaIRCompiler.isDirectOperator(op)) {
            final OpDefNode def = (OpDefNode) op;
            return expandPredicate(def.getBody(), tool.getOpContext(def, args, lexicalContext, true), input);
        }

        final int opcode = BuiltInOPs.getOpCode(op.getName());
        if (opcode == OPCODE_lor || opcode == OPCODE_dl) {
            final List<BranchPlan> output = new ArrayList<BranchPlan>();
            for (ExprOrOpArgNode arg : args) {
                output.addAll(expandPredicate(arg, lexicalContext, copyPlans(input)));
            }
            return output;
        }
        if (opcode == OPCODE_land || opcode == OPCODE_cl) {
            List<BranchPlan> output = input;
            for (ExprOrOpArgNode arg : args) {
                output = expandPredicate(arg, lexicalContext, output);
                if (output.isEmpty()) break;
            }
            return output;
        }
        if (opcode == OPCODE_be) return expandBoundedExists(appl, lexicalContext, input);
        return appendAtom(input, node, lexicalContext);
    }

    private List<BranchPlan> expandBoundedExists(final OpApplNode node, final Context lexicalContext,
            final List<BranchPlan> input) {
        final FormalParamNode[][] formals = node.getBdedQuantSymbolLists();
        final boolean[] tupleFlags = node.isBdedQuantATuple();
        final tla2sany.semantic.ExprNode[] bounds = node.getBdedQuantBounds();
        if (formals.length != bounds.length || tupleFlags.length != bounds.length || node.getArgs().length != 1) {
            throw unsupported("malformed bounded existential: " + node);
        }
        if (bounds.length == 0) throw unsupported("bounded existential has no bounds: " + node);
        for (int i = 0; i < formals.length; i++) {
            if (tupleFlags[i] || formals[i].length == 0) {
                throw unsupported("tuple-bound existential is not yet representable in GPU successor IR: " + node);
            }
        }
        return expandBoundedExistsBinding(node, lexicalContext, input, 0, 0);
    }

    private List<BranchPlan> expandBoundedExistsBinding(final OpApplNode node,
            final Context bindingContext, final List<BranchPlan> input,
            final int boundIndex, final int formalIndex) {
        final FormalParamNode[][] formals = node.getBdedQuantSymbolLists();
        if (boundIndex >= formals.length) {
            return expandPredicate(node.getArgs()[0], bindingContext, input);
        }

        final tla2sany.semantic.ExprNode bound = node.getBdedQuantBounds()[boundIndex];
        final FormalParamNode formal = formals[boundIndex][formalIndex];
        final int nextBoundIndex = formalIndex + 1 < formals[boundIndex].length
                ? boundIndex : boundIndex + 1;
        final int nextFormalIndex = formalIndex + 1 < formals[boundIndex].length
                ? formalIndex + 1 : 0;

        if (dynamicDomain(bound, bindingContext)) {
            final List<BranchPlan> plans = copyPlans(input);
            for (BranchPlan plan : plans) {
                plan.dynamicBindings.add(new DynamicBinding(formal, bound, bindingContext));
            }
            return expandBoundedExistsBinding(node, bindingContext, plans,
                    nextBoundIndex, nextFormalIndex);
        }

        final Object domain = tool.eval(bound, bindingContext);
        if (!(domain instanceof Enumerable)) {
            throw unsupported("bounded existential domain is not finite/enumerable: " + bound);
        }
        final List<BranchPlan> output = new ArrayList<BranchPlan>();
        final ValueEnumeration values = ((Enumerable) domain).elements();
        Value value;
        while ((value = values.nextElement()) != null) {
            final Context nextContext = bindingContext.cons(formal, value);
            output.addAll(expandBoundedExistsBinding(node, nextContext, copyPlans(input),
                    nextBoundIndex, nextFormalIndex));
        }
        return output;
    }

    private boolean dynamicDomain(final SemanticNode node, final Context lexicalContext) {
        if (node == null || node.getKind() != ASTConstants.OpApplKind) return false;
        final OpApplNode appl = (OpApplNode) node;
        final SymbolNode op = appl.getOperator();
        if (op.getKind() == ASTConstants.VariableDeclKind) return varIndex.containsKey(op.getName());
        if (op.getKind() == ASTConstants.FormalParamKind) return lexicalContext.lookup(op) == null;
        if (op.getKind() == ASTConstants.UserDefinedOpKind
                && !GpuTlaIRCompiler.isDirectOperator(op)) {
            final OpDefNode def = (OpDefNode) op;
            return dynamicDomain(def.getBody(), tool.getOpContext(def, appl.getArgs(), lexicalContext, true));
        }
        for (ExprOrOpArgNode arg : appl.getArgs()) if (dynamicDomain(arg, lexicalContext)) return true;
        return false;
    }

    private static List<BranchPlan> appendAtom(final List<BranchPlan> input, final SemanticNode node,
            final Context lexicalContext) {
        for (BranchPlan plan : input) plan.atoms.add(new Atom(node, lexicalContext));
        return input;
    }

    private static List<BranchPlan> copyPlans(final List<BranchPlan> source) {
        final List<BranchPlan> copy = new ArrayList<BranchPlan>(source.size());
        for (BranchPlan plan : source) copy.add(new BranchPlan(plan));
        return copy;
    }

    private void compileBranch(final BranchPlan plan) {
        final boolean[] boundNextVars = new boolean[vars.length];
        final List<Atom> bindings = new ArrayList<Atom>();
        final List<Atom> preFilters = new ArrayList<Atom>();
        final List<Atom> postFilters = new ArrayList<Atom>();
        for (Atom atom : plan.atoms) {
            if (compileBindingAtom(atom, boundNextVars, false)) {
                bindings.add(atom);
            } else {
                (requiresNextState(atom.node, atom.context) ? postFilters : preFilters).add(atom);
            }
        }
        // The resident kernel evaluates operations in table order. Current-state
        // guards must run before allocations from EXCEPT, record, and set updates.
        for (DynamicBinding dynamic : plan.dynamicBindings) {
            final GpuTlaIR expr = GpuTlaIRCompiler.compile(tool, dynamic.context, vars, dynamic.bound, builder.locals());
            final int exprIndex = builder.addExpr(expr);
            if (DEBUG) System.out.println("GPU generic emit ENUM_LOCAL formal=" + dynamic.formal + " identity=" + System.identityHashCode(dynamic.formal));
            builder.addOp(GpuTlaSuccessorIR.OP_ENUM_LOCAL, builder.locals().slot(dynamic.formal), exprIndex);
        }
        for (Atom filter : preFilters) addFilter(filter.node, filter.context);
        final boolean[] emittedNextVars = new boolean[vars.length];
        for (Atom binding : bindings) compileBindingAtom(binding, emittedNextVars, true);
        for (Atom filter : postFilters) addFilter(filter.node, filter.context);
        for (int i = 0; i < boundNextVars.length; i++) {
            if (!boundNextVars[i]) {
                throw unsupported("action branch leaves next-state variable unconstrained: " + varNames[i]);
            }
        }
    }

    private boolean compileBindingAtom(final Atom atom, final boolean[] boundNextVars) {
        return compileBindingAtom(atom, boundNextVars, true);
    }

    private boolean compileBindingAtom(final Atom atom, final boolean[] boundNextVars, final boolean emit) {
        final SemanticNode node = atom.node;
        final Context lexicalContext = atom.context;
        if (node.getKind() != ASTConstants.OpApplKind) return false;
        final OpApplNode appl = (OpApplNode) node;
        final int opcode = BuiltInOPs.getOpCode(appl.getOperator().getName());
        final ExprOrOpArgNode[] args = appl.getArgs();
        if ((opcode == OPCODE_eq || opcode == OPCODE_in) && args.length == 2) {
            final Integer primeVar = primeVariableIndex(args[0]);
            if (primeVar != null) {
                final int index = primeVar.intValue();
                if (boundNextVars[index]) return false;
                if (emit) {
                    final GpuTlaIR expr = GpuTlaIRCompiler.compile(tool, lexicalContext, vars, args[1], builder.locals());
                    final int exprIndex = builder.addExpr(expr);
                    if (DEBUG) {
                        System.out.println("GPU generic emit " + (opcode == OPCODE_eq ? "ASSIGN" : "ENUM")
                                + " var=" + index + " expr=" + exprIndex + " node=" + node);
                    }
                    builder.addOp(opcode == OPCODE_eq ? GpuTlaSuccessorIR.OP_ASSIGN : GpuTlaSuccessorIR.OP_ENUM,
                            index, exprIndex);
                }
                boundNextVars[index] = true;
                return true;
            }
        }
        if (opcode == OPCODE_unchanged && args.length == 1) return markUnchanged(args[0], boundNextVars);
        return false;
    }

    private boolean markUnchanged(final SemanticNode node, final boolean[] boundNextVars) {
        if (node.getKind() != ASTConstants.OpApplKind) return false;
        final OpApplNode appl = (OpApplNode) node;
        if (appl.getOperator().getKind() == ASTConstants.VariableDeclKind && appl.getArgs().length == 0) {
            final Integer index = varIndex.get(appl.getOperator().getName());
            if (index == null) throw unsupported("unknown UNCHANGED variable: " + appl.getOperator().getName());
            if (boundNextVars[index.intValue()]) return false;
            boundNextVars[index.intValue()] = true;
            return true;
        }
        if (BuiltInOPs.getOpCode(appl.getOperator().getName()) != OPCODE_tup) return false;
        for (ExprOrOpArgNode arg : appl.getArgs()) {
            if (!markUnchanged(arg, boundNextVars)) return false;
        }
        return true;
    }

    private void addFilter(final SemanticNode predicate, final Context lexicalContext) {
        final GpuTlaIR expr = GpuTlaIRCompiler.compile(tool, lexicalContext, vars, predicate, builder.locals());
        final int exprIndex = builder.addExpr(expr);
        if (DEBUG) {
            System.out.println("GPU generic emit FILTER expr=" + exprIndex + " node=" + predicate);
        }
        builder.addOp(GpuTlaSuccessorIR.OP_FILTER, -1, exprIndex);
    }

    private boolean requiresNextState(final SemanticNode node, final Context lexicalContext) {
        if (node == null || node.getKind() != ASTConstants.OpApplKind) return false;
        final OpApplNode appl = (OpApplNode) node;
        final SymbolNode op = appl.getOperator();
        final int opcode = BuiltInOPs.getOpCode(op.getName());
        if (opcode == OPCODE_prime || opcode == OPCODE_unchanged) return true;
        if (op.getKind() == ASTConstants.UserDefinedOpKind
                && !GpuTlaIRCompiler.isDirectOperator(op)) {
            final OpDefNode def = (OpDefNode) op;
            return requiresNextState(def.getBody(), tool.getOpContext(def, appl.getArgs(), lexicalContext, true));
        }
        for (ExprOrOpArgNode arg : appl.getArgs()) {
            if (requiresNextState(arg, lexicalContext)) return true;
        }
        return false;
    }

    private Integer primeVariableIndex(final SemanticNode node) {
        if (node.getKind() != ASTConstants.OpApplKind) return null;
        final OpApplNode appl = (OpApplNode) node;
        if (BuiltInOPs.getOpCode(appl.getOperator().getName()) != OPCODE_prime) return null;
        final ExprOrOpArgNode[] args = appl.getArgs();
        if (args.length != 1 || args[0].getKind() != ASTConstants.OpApplKind) return null;
        final OpApplNode var = (OpApplNode) args[0];
        if (var.getOperator().getKind() != ASTConstants.VariableDeclKind) return null;
        final Integer index = varIndex.get(var.getOperator().getName());
        if (index == null) throw unsupported("unknown primed variable: " + var.getOperator().getName());
        return index;
    }

    private static final class Atom {
        private final SemanticNode node;
        private final Context context;

        private Atom(final SemanticNode node, final Context context) {
            this.node = node;
            this.context = context;
        }
    }

    private static final class DynamicBinding {
        private final FormalParamNode formal;
        private final SemanticNode bound;
        private final Context context;

        private DynamicBinding(final FormalParamNode formal, final SemanticNode bound, final Context context) {
            this.formal = formal;
            this.bound = bound;
            this.context = context;
        }
    }

    private static final class BranchPlan {
        private final List<Atom> atoms;
        private final List<DynamicBinding> dynamicBindings;

        private BranchPlan() {
            this.atoms = new ArrayList<Atom>();
            this.dynamicBindings = new ArrayList<DynamicBinding>();
        }

        private BranchPlan(final BranchPlan source) {
            this.atoms = new ArrayList<Atom>(source.atoms);
            this.dynamicBindings = new ArrayList<DynamicBinding>(source.dynamicBindings);
        }
    }

    private static UnsupportedOperationException unsupported(final String message) {
        return new UnsupportedOperationException("GPU TLA+ successor compiler: " + message);
    }
}
