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
 *
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/dead_code_elimination.cc
 * \brief Dead code elimination pass.
 * Currently it removes:
 *   1. Unused local VarBindings in a DataflowBlock.
 */

#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

namespace tvm {
namespace relax {

class DeadCodeEliminator : public ExprMutator {
 private:
  Expr VisitExpr_(const VarNode* op) final {
    used_vars_.insert(GetRef<Var>(op));
    return GetRef<Expr>(op);
  }

  Expr VisitExpr_(const DataflowVarNode* op) final {
    used_vars_.insert(GetRef<Var>(op));
    return GetRef<Expr>(op);
  }

  void VisitBinding_(const VarBindingNode* binding) {
    this->VisitExpr(binding->value);
  }
  
  void VisitBinding_(const MatchCastNode* binding) {
    this->VisitExpr(binding->value);
    this->VisitAndCheckStructInfoFieldUnchanged(binding->struct_info);
  }

  BindingBlock VisitBindingBlock_(const DataflowBlockNode* block) final {
    // reverse scan the data flow block to find the used vars
    used_vars_.clear();
    std::vector<Binding> new_bindings;
    for (auto rit = block->bindings.rbegin(); rit != block->bindings.rend(); rit++) {
      const Var& var = (*rit)->var;
      // only keep the used bindings or non dataflow var bindings
      if (used_vars_.count(var) || !var->IsInstance<DataflowVarNode>()) {
        new_bindings.push_back(*rit);
        // collect the used vars
        this->VisitBinding((*rit));
      }
    }
    // reverse the bindings
    std::reverse(new_bindings.begin(), new_bindings.end());
    if (new_bindings.size() == block->bindings.size()) {
      return GetRef<BindingBlock>(block);
    } else {
      auto n = make_object<DataflowBlockNode>(*block);
      n->bindings = std::move(new_bindings);
      return BindingBlock(n);
    }
  }

  BindingBlock VisitBindingBlock_(const BindingBlockNode* block) final {
    return GetRef<BindingBlock>(block);
  }

  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> used_vars_;
};

Expr DeadCodeElimination(const Function& func, const IRModule& mod) {
  DeadCodeEliminator eliminator;
  return eliminator(func);
}

namespace transform {

Pass DeadCodeElimination() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) {
        return Downcast<Function>(DeadCodeElimination(f, m));
      };
  return CreateFunctionPass(pass_func, 1, "DeadCodeElimination", {});
}

TVM_REGISTER_GLOBAL("relax.transform.DeadCodeElimination").set_body_typed(DeadCodeElimination);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
