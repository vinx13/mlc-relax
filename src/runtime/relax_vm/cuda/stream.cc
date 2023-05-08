#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/vm.h>

#include "../../cuda/cuda_common.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

TVM_REGISTER_GLOBAL("vm.builtin.get_cuda_stream").set_body_typed([]([[maybe_unused]] TVMArgValue vm) {
  return static_cast<void*>(CUDAThreadEntry::ThreadLocal()->stream);
});

TVM_REGISTER_GLOBAL("runtime.get_cuda_stream").set_body_typed([]() {
  return static_cast<void*>(CUDAThreadEntry::ThreadLocal()->stream);
});

} // namespace relax_vm
}  // namespace runtime
}  // namespace tvm