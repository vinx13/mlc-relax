/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/tir/expr.h>

#include "../../../support/arena.h"
#include "../../../support/ordered_set.h"
#include "../../../support/utils.h"

namespace tvm {
namespace relax {

TVM_REGISTER_PASS_CONFIG_OPTION("relax.backend.use_cuda_graph", Bool);

struct RewritePlan {
  Function func;
  bool is_alloc;  // is alloc or capture
  const VarNode* launch_point;
  // bindings to remove
  std::unordered_set<const VarNode*> lifted_bindings;
  // Remapping of the boundary (input/output) variables in the original module
  // std::unordered_map<const VarNode*, int> output_index;
  std::vector<const VarNode*> outputs;
  std::vector<const VarNode*> inputs;
};

/*! \brief Builder of the lifted function for cuda graph capturing or allocations */
class FuncBuilder : public ExprMutator {
 public:
  /*!
   * \brief Add a binding to the new function
   * \param binding The binding to add
   */
  void AddBinding(const VarBindingNode* binding) { bindings_.push_back(binding); }

  /*!
   * \brief Mark a variable as the input of the new function.
   * \param var The variable to mark as input
   */
  void MarkInput(const VarNode* var) { inputs_.push_back(var); }
  /*!
   * \brief Mark a variable as the output of the new function. The variable must be the LHS of an
   * existing binding in the new function.
   * \param var The variable to mark as output
   */
  void MarkOutput(const VarNode* var) { outputs_.push_back(var); }

  /*! \brief Get the number of bindings in the new function */
  auto size() const { return bindings_.size(); }

  /*! \brief Build the new function */
  Function Build() {
    Array<Var> params;
    // Set up the parameters
    for (const auto* var : inputs_) {
      auto var_node = make_object<VarNode>(*var);
      auto new_var = Var(var_node);
      var_remap_[var->vid] = new_var;
      params.push_back(new_var);
    }
    // Emit the function body
    builder_->BeginBindingBlock();
    for (const auto* binding : bindings_) {
      VisitBinding_(binding);
    }
    // Set up the outputs
    Array<Expr> outputs;
    for (const auto* var : outputs_) {
      outputs.push_back(VisitExpr_(var));
    }
    auto output = builder_->Emit(Tuple(outputs));
    auto block = builder_->EndBlock();
    auto body = builder_->Normalize(SeqExpr({block}, output));
    auto func = Function(params, body, Downcast<StructInfo>(output->struct_info_.value()));
    return func;
  }

  support::OrderedSet<const VarNode*> inputs_;
  support::OrderedSet<const VarNode*> outputs_;
  std::vector<const VarBindingNode*> bindings_;
};

/*!
 * \brief The planner for rewriting the function to enable cuda graph capturing.
 *
 * To enable cuda graph capturing, we need to identify and lift the static regions of the original
 * function. The static region is a region where all the tensors are statically allocated and only
 * contains kernel launches (control flow is not allowed). This pass is expected to run after
 * StaticPlanBlockMemory. After StaticPlanBlockMemory, all the tensors that can be statically
 * allocated are allocated with `R.memory.alloc_storage` and `R.memory.alloc_tensor`, while other
 * tensors will be allocated via `R.builtin.alloc_tensor`.
 *
 * This pass is executed at the level of BindingBlock. It first identify all the
 * storage objects allocated with `R.memory.alloc_storage` within the BindingBlock, and then
 * identify the static regions by propagating starting from the storage objects.
 *
 * All the calls to `R.memory.alloc_storage` within the same BindingBlock are grouped into a single
 * new function. Each of the static regions are lifted to a new function.
 */
class CUDAGraphRewritePlanner : public ExprVisitor {
 public:
  CUDAGraphRewritePlanner(IRModule mod) : mod_(mod) {}
  std::vector<RewritePlan> Plan() {
    for (const auto& [gv, func] : mod_->functions) {
      if (func->IsInstance<FunctionNode>()) {
        VisitExpr(func);
      }
    }
    std::vector<RewritePlan> plans;

    auto region_to_plan = [&](FuncBuilder* region, bool is_alloc) -> RewritePlan {
      RewritePlan plan;
      plan.is_alloc = true;
      plan.func = region->Build();
      ICHECK(region->size());
      plan.launch_point = region->bindings_.front()->var.get();
      plan.is_alloc = is_alloc;
      for (const auto* binding : region->bindings_) {
        plan.lifted_bindings.insert(binding->var.get());
      }
      plan.inputs.assign(region->inputs_.begin(), region->inputs_.end());
      plan.outputs.assign(region->outputs_.begin(), region->outputs_.end());
      return plan;
    };

    for (auto* region : alloc_storages_) {
      plans.push_back(region_to_plan(region, true));
    }

    for (auto* region : regions_) {
      plans.push_back(region_to_plan(region, false));
    }
    return plans;
  }

  /*!
   *\brief Start a new static region. This method should be called when encountering a
   * CUDA kernel launch (calls to PrimFunc or ExternFunc) that only depends on static parameters.
   */
  void StartRegion() { current_.capture_builder = arena_.make<FuncBuilder>(); }

  /*!
   * \brief Finish a static region. This method should be called when non-static bindings or
   * unsupported operations are encountered.
   */
  void EndRegion() {
    if (current_.capture_builder && current_.capture_builder->size()) {
      regions_.emplace_back(current_.capture_builder);
    }
    current_.capture_builder = nullptr;
  }

  void VisitBindingBlock_(const BindingBlockNode* binding_block) final {
    Scope new_scope;
    std::swap(new_scope, current_);
    current_.alloc_storage_builder = arena_.make<FuncBuilder>();
    for (const auto& binding : binding_block->bindings) {
      VisitBinding(binding);
    }
    EndRegion();
    if (current_.alloc_storage_builder->outputs_.size()) {
      alloc_storages_.emplace_back(current_.alloc_storage_builder);
    }
    std::swap(new_scope, current_);
  }

  void VisitBinding_(const VarBindingNode* binding, const CallNode* call) final {
    static const auto& mem_alloc_storage_op = Op::Get("relax.memory.alloc_storage");
    // static const auto& mem_alloc_tensor_op = Op::Get("relax.memory.alloc_tensor");
    static const auto& mem_kill_storage_op = Op::Get("relax.memory.kill_storage");
    // static const auto& mem_kill_tensor_op = Op::Get("relax.memory.kill_tensor");
    static const auto& builtin_alloc_tensor_op = Op::Get("relax.builtin.alloc_tensor");
    static const auto& call_builtin_with_ctx_op = Op::Get("relax.call_builtin_with_ctx");

    if (call->op.same_as(mem_alloc_storage_op) && IsStaticAllocStorage(binding)) {
      AddPendingBinding(binding, /*is_alloc_storage=*/true);
      return;
    } else if (call->op.same_as(mem_kill_storage_op) || call->op.same_as(builtin_alloc_tensor_op)) {
      return;
    }

    bool is_kernel_launch = [&]() {
      if (const auto* gv = call->op.as<GlobalVarNode>();
          gv && mod_->Lookup(GetRef<GlobalVar>(gv))->IsInstance<tir::PrimFuncNode>()) {
        return true;
      }
      if (call->op.as<ExternFuncNode>()) {
        return true;
      }
      if (const auto* op = call->op.as<OpNode>()) {
        return !support::StartsWith(op->name, "relax.memory") &&
               !support::StartsWith(op->name, "relax.builtin") &&
               !GetRef<Op>(op).same_as(call_builtin_with_ctx_op);
      }
      return false;
    }();

    std::vector<const VarNode*> args;
    bool is_all_static = IsStatic(call->args, &args);

    if (is_all_static) {
      if (current_.capture_builder == nullptr && is_kernel_launch) {
        StartRegion();
      }
      AddPendingBinding(binding, /*is_alloc_storage=*/false);
      MarkAsFuncInput(args);
    } else {
      EndRegion();
    }

    MarkAsFuncOutput(args);
  }

  void MarkAsFuncInput(const std::vector<const VarNode*>& vars) {
    if (current_.capture_builder == nullptr) {
      return;
    }
    for (const VarNode* var : vars) {
      auto it = binding_to_region_.find(var);
      if (it == binding_to_region_.end() || it->second != current_.capture_builder) {
        current_.capture_builder->MarkInput(var);
      }
    }
  }

  void MarkAsFuncOutput(const std::vector<const VarNode*>& vars) {
    for (const VarNode* var : vars) {
      if (auto it = binding_to_region_.find(var);
          it != binding_to_region_.end() && it->second != current_.capture_builder) {
        it->second->MarkOutput(var);
      }
    }
  }

  void VisitBinding_(const VarBindingNode* binding, const VarNode* var) final {
    if (IsStatic(GetRef<Var>(var))) {
      AddPendingBinding(binding, false);
      MarkAsFuncInput({var});
    } else {
      EndRegion();
    }
    MarkAsFuncOutput({var});
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleNode* tuple) final {
    std::vector<const VarNode*> args;
    if (IsStatic(tuple->fields, &args)) {
      AddPendingBinding(binding, false);
      MarkAsFuncInput(args);
    } else {
      EndRegion();
    }
    MarkAsFuncOutput(args);
  }

  void VisitBinding_(const VarBindingNode* binding, const TupleGetItemNode* tuple_get_item) {
    const VarNode* tuple = tuple_get_item->tuple.as<VarNode>();
    ICHECK(tuple);
    if (IsStatic(tuple_get_item->tuple)) {
      AddPendingBinding(binding, false);
      MarkAsFuncInput({tuple});
    } else {
      EndRegion();
    }
    MarkAsFuncOutput({tuple});
  }

  bool IsStatic(const PrimExpr& expr,
                [[maybe_unused]] std::vector<const VarNode*>* vars_collector = nullptr) {
    return expr->IsInstance<tir::IntImmNode>() || expr->IsInstance<tir::FloatImmNode>();
  }

  bool IsStatic(const Expr& expr, std::vector<const VarNode*>* vars_collector = nullptr) {
    if (expr->IsInstance<ConstantNode>() || expr->IsInstance<DataTypeImmNode>()) {
      return true;
    }
    if (const auto* prim_value = expr.as<PrimValueNode>()) {
      return IsStatic(prim_value->value, vars_collector);
    }
    if (const auto* var = expr.as<VarNode>()) {
      if (vars_collector != nullptr) {
        vars_collector->push_back(var);
      }
      return static_bindings_.count(var);
    }

    if (const auto* shape = expr.as<ShapeExprNode>()) {
      return IsStatic(shape->values, vars_collector);
    }
    if (const auto* tuple = expr.as<TupleNode>()) {
      return IsStatic(tuple->fields, vars_collector);
    }

    // TODO: remove this after rebasing
    if (expr->IsInstance<ExternFuncNode>()) {
      return true;
    }
    return false;
  }

  template <typename T>
  bool IsStatic(const Array<T>& exprs, std::vector<const VarNode*>* vars_collector = nullptr) {
    return std::all_of(exprs.begin(), exprs.end(),
                       [&](const T& expr) { return IsStatic(expr, vars_collector); });
  }

 private:
  bool IsStaticAllocStorage(const VarBindingNode* binding) {
    // Check if the allocation has constant shape
    const auto* alloc_storage_call = binding->value.as<CallNode>();
    auto shape = Downcast<ShapeExpr>(alloc_storage_call->args[0]);
    return std::all_of(shape->values.begin(), shape->values.end(),
                       [](const PrimExpr& expr) { return expr.as<IntImmNode>() != nullptr; });
  }

  void AddPendingBinding(const VarBindingNode* binding, bool is_alloc_storage) {
    if (is_alloc_storage) {
      current_.alloc_storage_builder->AddBinding(binding);
      binding_to_region_[binding->var.get()] = current_.alloc_storage_builder;
    } else if (current_.capture_builder != nullptr) {
      current_.capture_builder->AddBinding(binding);
      binding_to_region_[binding->var.get()] = current_.capture_builder;
    }
    static_bindings_.emplace(binding->var.get(), GetRef<VarBinding>(binding));
  }

  struct Scope {
    FuncBuilder* alloc_storage_builder = nullptr;
    FuncBuilder* capture_builder = nullptr;
  };

  IRModule mod_;
  Scope current_;
  std::unordered_map<const VarNode*, VarBinding> static_bindings_;
  std::unordered_map<const VarNode*, FuncBuilder*> binding_to_region_;
  std::vector<FuncBuilder*> regions_;
  std::vector<FuncBuilder*> alloc_storages_;
  support::Arena arena_;
};

class CUDAGraphRewriter : public ExprMutator {
 public:
  CUDAGraphRewriter(const IRModule& mod) : ExprMutator(mod) {}

  IRModule Rewrite() {
    CUDAGraphRewritePlanner planner(builder_->GetContextIRModule());
    auto plans = planner.Plan();
    for (const auto& plan : plans) {
      subgraph_launches[plan.launch_point] = plan;
    }

    for (const auto& [gv, func] : builder_->GetContextIRModule()->functions) {
      if (func->IsInstance<FunctionNode>()) {
        auto new_func = Downcast<Function>(VisitExpr(func));
        if (!new_func.same_as(func)) {
          builder_->UpdateFunction(gv, new_func);
        }
      }
    }
    return builder_->GetContextIRModule();
  }

  void LaunchSubgraph(const VarBindingNode* op, const RewritePlan& plan) {
    static const auto& call_builtin_with_ctx_op = Op::Get("relax.call_builtin_with_ctx");
    static const auto& builtin_run_or_capture = ExternFunc("vm.builtin.cuda_graph.run_or_capture");
    static const auto& builtin_get_cached_alloc =
        ExternFunc("vm.builtin.cuda_graph.get_cached_alloc");

    Expr launch_subgraph;
    auto gv_func =
        builder_->AddFunction(plan.func, plan.is_alloc ? "cuda_graph_alloc" : "cuda_graph_capture");
    if (plan.is_alloc) {
      ICHECK(plan.inputs.empty());
      launch_subgraph = Call(call_builtin_with_ctx_op,
                             {builtin_get_cached_alloc,

                              Tuple({gv_func, PrimValue(Integer(index_alloc++))})

                             },
                             Attrs(), {plan.func->ret_struct_info});
    } else {
      Array<Expr> args;
      for (const auto& arg : plan.inputs) {
        args.push_back(VisitExpr_(arg));
      }
      launch_subgraph = Call(call_builtin_with_ctx_op,
                             {builtin_run_or_capture,
                              Tuple({gv_func, Tuple(args), PrimValue(Integer(index_capture++))})},
                             Attrs(), {plan.func->ret_struct_info});
    }
    Expr ret_value = builder_->Emit(launch_subgraph);
    for (int i = 0; i < static_cast<int>(plan.outputs.size()); ++i) {
      var_redef_[plan.outputs[i]] = TupleGetItem(ret_value, i);
    }

    for (const auto* binding : plan.lifted_bindings) {
      lifted_bindings_.insert(binding);
    }
  }

  void VisitBinding_(const VarBindingNode* op) final {
    if (subgraph_launches.count(op->var.get())) {
      LaunchSubgraph(op, subgraph_launches[op->var.get()]);
    }
    if (lifted_bindings_.count(op->var.get())) {
      // The binding is lifted to the subgraph and will be removed from the original function.
      return;
    }
    ExprMutator::VisitBinding_(op);
  }

  Expr VisitExpr_(const VarNode* var) {
    if (var_redef_.count(var) && !var_remap_.count(var->vid)) {
      auto new_var = builder_->Emit(var_redef_[var], var->name_hint());
      var_remap_[var->vid] = new_var;
      return new_var;
    }
    return ExprMutator::VisitExpr_(var);
  }

  std::unordered_map<const VarNode*, RewritePlan> subgraph_launches;
  std::unordered_map<const VarNode*, Expr> var_redef_;
  std::unordered_set<const VarNode*> lifted_bindings_;
  int index_alloc = 0;
  int index_capture = 0;
};

IRModule RewriteCUDAGraph(IRModule mod) {
  CUDAGraphRewriter rewriter(mod);
  mod = rewriter.Rewrite();
  return mod;
}

namespace transform {

Pass RewriteCUDAGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =  //
      [=](IRModule m, PassContext pc) { return ::tvm::relax::RewriteCUDAGraph(std::move(m)); };
  return CreateModulePass(pass_func, 0, "RewriteCUDAGraph", {});
}

TVM_REGISTER_GLOBAL("relax.transform.RewriteCUDAGraph").set_body_typed(RewriteCUDAGraph);

}  // namespace transform

}  // namespace relax
}  // namespace tvm
