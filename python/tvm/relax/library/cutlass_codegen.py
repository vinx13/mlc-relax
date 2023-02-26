# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name,comparison-with-callable,unused-variable
"""codegen for cutlass"""
from typing import List, Dict, Any

from tvm.contrib.cutlass.build import (
    select_gemm_kernel,
    _get_cutlass_path,
    handle_conv2d,
    codegen_cutlass_c_source,
)
from tvm.contrib.cutlass.gen_gemm import CutlassGemmProfiler
from tvm.contrib.cutlass.gen_conv2d import CutlassConv2DProfiler
from .pattern import (
    MatchResult,
    dense_row_row_row_fp16,
    bias_row_fp16,
    bias_row_fp16_2,
    batch_bias_row_fp16,
    batch_bias_row_fp16_2,
    relu_fp16,
    batch_dense_row_row_row_fp16,
    batch_dense_2_row_row_row_fp16,
    conv2d_nhwc_fp16,
    padding_2d_nhwc_fp16,
    copy_4d_nhwc_fp16,
    bias_add_nhwc_fp16,
    bias_add_nhwc_fp16_2,
    elem_add_2d_fp16,
    elem_add_3d_fp16,
    elem_add_4d_fp16,
)

# list representing the anchor ops
# in the future more layouts/dtypes can be supported
DENSE_LIST = [dense_row_row_row_fp16]
DENSE_BIAS_LIST = [bias_row_fp16, bias_row_fp16_2]
BATCH_DENSE_LIST = [batch_dense_row_row_row_fp16, batch_dense_2_row_row_row_fp16]
BATCH_DENSE_BIAS_LIST = [batch_bias_row_fp16, batch_bias_row_fp16_2]
CONV2D_LIST = [conv2d_nhwc_fp16]

# attributes for anchor ops used in code generation
OP_PATTERN_ATTR_LIST = {
    dense_row_row_row_fp16: {
        "typea": "float16",
        "typeb": "float16",
        "typec": "float16",
        "layouta": "row",
        "layoutb": "row",
        "layoutc": "row",
    },
    batch_dense_row_row_row_fp16: {
        "typea": "float16",
        "typeb": "float16",
        "typec": "float16",
        "layouta": "row",
        "layoutb": "row",
        "layoutc": "row",
    },
    batch_dense_2_row_row_row_fp16: {
        "typea": "float16",
        "typeb": "float16",
        "typec": "float16",
        "layouta": "row",
        "layoutb": "row",
        "layoutc": "row",
    },
    conv2d_nhwc_fp16: {
        "data_dtype": "float16",
        "weight_dtype": "float16",
        "out_dtype": "float16",
        # in the future we can add layout here
    },
}


def dense_bias_relu(match_results, attr, get_code=True):
    if len(match_results) < 3:
        return None
    attr = dense_bias(match_results[:2], attr, get_code=False)
    if attr is None or match_results[2].pattern != relu_fp16:
        return None
    m_bias, n_bias = match_results[1].symbol_values
    m_relu, n_relu = match_results[2].symbol_values
    A_bias, B_bias, C_bias = match_results[1].matched_buffers
    A_relu, B_relu = match_results[2].matched_buffers
    if m_bias == m_relu and n_bias == n_relu and C_bias == A_relu:
        attr["op_type"] = "cutlass.dense_bias_relu"
        return [_get_graph_pattern_cutlass_code(attr=attr), 3] if get_code else attr
    return None


def dense_bias(match_results, attr, get_code=True):
    if len(match_results) < 2:
        return None
    attr = dense(match_results[:1], attr, get_code=False)
    if attr is None or match_results[1].pattern not in DENSE_BIAS_LIST:
        return None
    m_dense, n_dense, k_dense = match_results[0].symbol_values
    m_bias, n_bias = match_results[1].symbol_values
    A_dense, B_dense, C_dense = match_results[0].matched_buffers
    A_bias, B_bias, C_bias = match_results[1].matched_buffers
    if m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
        attr["op_type"] = "cutlass.dense_bias"
        return [_get_graph_pattern_cutlass_code(attr=attr), 2] if get_code else attr
    return None


def dense(match_results, attr, get_code=True):
    if len(match_results) < 1:
        return None
    if match_results[0].pattern in DENSE_LIST:
        # dense
        attr["op_type"] = "cutlass.dense"
        return [_get_graph_pattern_cutlass_code(attr=attr), 1] if get_code else attr
    return None


def batch_dense_bias(match_results, attr, get_code=True):
    if len(match_results) < 2:
        return None
    attr = batch_dense(match_results[:1], attr, get_code=False)
    if attr is None or match_results[1].pattern not in BATCH_DENSE_BIAS_LIST:
        return None
    m_dense, n_dense, k_dense, b_dense = match_results[0].symbol_values
    m_bias, n_bias, b_bias = match_results[1].symbol_values
    A_dense, B_dense, C_dense = match_results[0].matched_buffers
    A_bias, B_bias, C_bias = match_results[1].matched_buffers
    if b_dense == b_bias and m_dense == m_bias and n_dense == n_bias and C_dense == A_bias:
        attr["op_type"] = "cutlass.batch_matmul_bias"
        return [_get_graph_pattern_cutlass_code(attr=attr), 2] if get_code else attr
    return None


def batch_dense(match_results, attr, get_code=True):
    if len(match_results) < 1:
        return None
    if match_results[0].pattern in BATCH_DENSE_LIST:
        attr["op_type"] = "cutlass.batch_matmul"
        return [_get_graph_pattern_cutlass_code(attr=attr), 1] if get_code else attr
    return None


def conv2d_bias_residual_add(match_results, attr, get_code=True):
    if len(match_results) < 4:
        return None
    attr = conv2d_bias(match_results[:3], attr, get_code=False)
    if attr is None or match_results[3].pattern != elem_add_4d_fp16:
        return None
    N_bias, H_bias, W_bias, C_bias = match_results[2].symbol_values
    in1_bias, in2_bias, out_bias = match_results[2].matched_buffers
    N_add, H_add, W_add, C_add = match_results[3].symbol_values
    in1_add, in2_add, out_add = match_results[3].matched_buffers
    if (
        N_bias == N_add
        and H_bias == H_add
        and W_bias == W_add
        and C_bias == C_add
        and (out_bias == in1_add or out_bias == in2_add)
    ):
        attr["op_type"] = "cutlass.conv2d_bias_residual_add"
        return [_get_graph_pattern_cutlass_code(attr=attr), 4] if get_code else attr
    return None


def conv2d_bias(match_results, attr, get_code=True):
    if len(match_results) < 3:
        return None
    attr = conv2d(match_results[:2], attr, get_code=False)
    if attr is None or (match_results[2].pattern not in [bias_add_nhwc_fp16, bias_add_nhwc_fp16_2]):
        return None
    (N_conv, pH_conv, pW_conv, H_conv, W_conv, C_conv, O_conv,) = match_results[
        1
    ].symbol_values[:7]
    A_pad_conv, B_conv, out_conv = match_results[1].matched_buffers
    N_bias, H_bias, W_bias, C_bias = match_results[2].symbol_values
    A_bias, B_bias, out_bias = match_results[2].matched_buffers
    if (
        N_bias == N_conv
        and H_bias == H_conv
        and W_bias == W_conv
        and C_bias == O_conv
        and out_conv == A_bias
    ):
        attr["op_type"] = "cutlass.conv2d_bias"
        return [_get_graph_pattern_cutlass_code(attr=attr), 3] if get_code else attr
    return None


def conv2d(match_results, attr, get_code=True):
    if len(match_results) < 2:
        return None
    if (
        match_results[0].pattern in [padding_2d_nhwc_fp16, copy_4d_nhwc_fp16]
        and match_results[1].pattern == conv2d_nhwc_fp16
    ):
        if match_results[0].pattern == padding_2d_nhwc_fp16:
            (
                N_pad,
                H_pad,
                W_pad,
                C_pad,
                pH_pad,
                pW_pad,
                lH_pad,
                lW_pad,
                rH_pad,
                rW_pad,
            ) = match_results[0].symbol_values
        else:
            (
                N_pad,
                H_pad,
                W_pad,
                C_pad,
            ) = match_results[0].symbol_values
            pH_pad = rH_pad = H_pad
            pW_pad = rW_pad = W_pad
            lH_pad = lW_pad = 0
        (
            N_conv,
            pH_conv,
            pW_conv,
            H_conv,
            W_conv,
            C_conv,
            O_conv,
            KH_conv,
            KW_conv,
            stride_h_conv,
            stride_w_conv,
            dilation_h_conv,
            dilation_w_conv,
        ) = match_results[1].symbol_values
        A, A_pad = match_results[0].matched_buffers
        A_pad_conv, B_conv, out_conv = match_results[1].matched_buffers
        if (
            N_pad == N_conv
            and pH_pad == pH_conv
            and pW_pad == pW_conv
            and C_pad == C_conv
            and A_pad == A_pad_conv
        ):
            if (
                lH_pad == pH_pad - rH_pad
                and lW_pad == pW_pad - rW_pad
                and lH_pad + H_pad == rH_pad
                and lW_pad + W_pad == rW_pad
            ):
                padding = (lH_pad, lW_pad)
                strides = (stride_h_conv, stride_w_conv)
                dilation = (dilation_h_conv, dilation_w_conv)
                attr["padding"] = padding
                attr["strides"] = strides
                attr["dilation"] = dilation
                attr["op_type"] = "cutlass.conv2d"
                return [_get_graph_pattern_cutlass_code(attr=attr), 2] if get_code else attr
    return None


def cutlass_fcodegen(sm=80, bin_dir="./bin"):
    gemm_profiler = CutlassGemmProfiler(sm, _get_cutlass_path(), bin_dir)
    conv2d_profiler = CutlassConv2DProfiler(sm, _get_cutlass_path(), bin_dir)

    def cutlass_codegen_with_match_results(match_results: List[MatchResult]):
        """generate cutlass code with match results"""
        nonlocal gemm_profiler
        nonlocal conv2d_profiler

        assert len(match_results) > 0

        # add shape into attr
        if match_results[0].pattern in DENSE_LIST:
            A_dense, B_dense, C_dense = match_results[0].matched_buffers
            m, n, k = A_dense.shape[0], B_dense.shape[1], A_dense.shape[1]
            attr: Dict[Any, Any] = OP_PATTERN_ATTR_LIST[match_results[0].pattern]
            attr["m"] = m
            attr["n"] = n
            attr["k"] = k
        elif match_results[0].pattern in BATCH_DENSE_LIST:
            A_dense, B_dense, C_dense = match_results[0].matched_buffers
            _, m, k = A_dense.shape
            n = B_dense.shape[-1]
            attr = OP_PATTERN_ATTR_LIST[match_results[0].pattern]
            attr["m"] = m
            attr["n"] = n
            attr["k"] = k
        elif len(match_results) >= 1 and match_results[1].pattern in CONV2D_LIST:
            A_input = match_results[0].matched_buffers[0]
            A_conv2d, B_conv2d, C_conv2d = match_results[1].matched_buffers
            d = A_input.shape
            w = B_conv2d.shape
            out_shape = C_conv2d.shape
            attr = OP_PATTERN_ATTR_LIST[match_results[1].pattern]
            attr["d"] = d
            attr["w"] = w
            attr["out_shape"] = out_shape
        else:
            return ["", 0]

        # add profiler into attr
        attr["gemm_profiler"] = gemm_profiler
        attr["conv2d_profiler"] = conv2d_profiler

        cutlass_patterns = [
            # 4
            conv2d_bias_residual_add,
            # 3
            dense_bias_relu,
            conv2d_bias,
            # 2
            dense_bias,
            batch_dense_bias,
            conv2d,
            # 1
            dense,
            batch_dense,
        ]
        for pattern in cutlass_patterns:
            res = pattern(match_results, attr)
            if res is not None:
                return res

        return ["", 0]

    return cutlass_codegen_with_match_results


def _get_graph_pattern_cutlass_code(attr):
    pattern = attr["op_type"]
    gemm_profiler = attr["gemm_profiler"]
    conv2d_profiler = attr["conv2d_profiler"]
    if pattern.startswith("cutlass.dense"):
        # initialize arg list for codegen function
        m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type = (
            attr["m"],
            attr["n"],
            attr["k"],
            attr["typea"],
            attr["typeb"],
            attr["typec"],
            attr["layouta"],
            attr["layoutb"],
            attr["layoutc"],
            attr["op_type"],
        )
        return cutlass_codegen_gemm(
            m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type, gemm_profiler
        )
    elif pattern.startswith("cutlass.batch_matmul"):
        m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type = (
            attr["m"],
            attr["n"],
            attr["k"],
            attr["typea"],
            attr["typeb"],
            attr["typec"],
            attr["layouta"],
            attr["layoutb"],
            attr["layoutc"],
            attr["op_type"],
        )
        return cutlass_codegen_batch_gemm(
            m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type, gemm_profiler
        )
    elif pattern.startswith("cutlass.conv2d"):
        (
            d,
            w,
            out_shape,
            padding,
            strides,
            dilation,
            out_dtype,
            data_dtype,
            weight_dtype,
            op_type,
        ) = (
            attr["d"],
            attr["w"],
            attr["out_shape"],
            attr["padding"],
            attr["strides"],
            attr["dilation"],
            attr["out_dtype"],
            attr["data_dtype"],
            attr["weight_dtype"],
            attr["op_type"],
        )
        return cutlass_codegen_conv2d(
            d,
            w,
            out_shape,
            padding,
            strides,
            dilation,
            out_dtype,
            data_dtype,
            weight_dtype,
            op_type,
            conv2d_profiler,
        )
    else:
        raise ValueError("op not supported")


def _convert_dtype_str(dtype):
    if isinstance(dtype, list):
        arr = []
        for t in dtype:
            arr.append(_convert_dtype_str(t))
        return arr
    elif isinstance(dtype, str):
        if dtype == "float16":
            return "cutlass::half_t"
        elif dtype == "float32":
            return "float"
    raise ValueError("dtype not supported")


def _convert_layout_str(layout):
    if isinstance(layout, list):
        arr = []
        for l in layout:
            arr.append(_convert_layout_str(l))
        return arr
    elif isinstance(layout, str):
        if layout == "row":
            return "cutlass::layout::RowMajor"
        elif layout == "col":
            return "cutlass::layout::ColumnMajor"
    raise ValueError("layout not supported")


def _reverse_layout(layout):
    if isinstance(layout, list):
        arr = []
        for l in layout:
            arr.append(_reverse_layout(l))
        return arr
    elif isinstance(layout, str):
        if layout == "cutlass::layout::RowMajor":
            return "cutlass::layout::ColumnMajor"
        elif layout == "cutlass::layout::ColumnMajor":
            return "cutlass::layout::RowMajor"
    raise ValueError("layout not supported")


def _attr_to_list(attr, arg_names):
    arr = []
    for n in arg_names:
        if n in attr:
            arr.append(attr[n])
    return arr


def cutlass_codegen_gemm(
    m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type, gemm_profiler
):
    """cutlass codegen for gemm"""
    op_name, cutlass_op_def = select_gemm_kernel(
        gemm_profiler,
        op_type,
        m,
        k,
        n,
        typec,
        typea,
        typeb,
        layoutc,
        layouta,
        layoutb,
        False,
        False,
        False,
        True,
    )
    op_name = "Operation_" + op_name
    typea, typeb, typec = _convert_dtype_str([typea, typeb, typec])
    layouta, layoutb, layoutc = _convert_layout_str([layouta, layoutb, layoutc])
    r_layouta, r_layoutb, r_layoutc = _reverse_layout([layouta, layoutb, layoutc])

    if op_type in ["cutlass.dense_bias", "cutlass.dense_bias_relu"]:
        bias_param = "NDArray Bias, "
        bias_var_def = f"""
        {r_layoutc}::Stride::Index ld_bias(0);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);
        """
        bias_arg = "{bias, ld_bias}"
    else:
        bias_param = bias_var_def = ""
        bias_arg = "{c, ldc}"

    text = f"""
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {{

      using namespace tvm;
      using namespace tvm::runtime;

      void _GEMM(NDArray A, NDArray B, {bias_param}NDArray C) {{
        // A: [M, K], B: [K, N]
        CHECK_EQ(A->ndim, 2);
        CHECK_EQ(B->ndim, 2);
        CHECK_EQ(C->ndim, 2);
        CHECK_EQ(A->shape[1], B->shape[0]);
        int M = A->shape[0];
        int K = A->shape[1];
        int N = B->shape[1];
        CHECK_EQ(C->shape[0], M);
        CHECK_EQ(C->shape[1], N);
        // Define the GEMM operation
        {cutlass_op_def}
        {op_name} gemm_op;
        {typec} alpha(1.0);
        {typec} beta(0.0);
        {r_layouta}::Stride::Index lda(K);
        {r_layoutb}::Stride::Index ldb(N);
        {r_layoutc}::Stride::Index ldc(N);
        {bias_var_def}
        {typea}* a = reinterpret_cast<{typea}*>(A->data);
        {typeb}* b = reinterpret_cast<{typeb}*>(B->data);
        {typec}* c = reinterpret_cast<{typec}*>(C->data);
        cutlass::Status status = gemm_op({{
            {{M, N, K}},     //
            {{a, lda}},      //
            {{b, ldb}},      //
            {bias_arg},      //
            {{c, ldc}},      //
            {{alpha, beta}}  //
        }});
        CHECK(status == cutlass::Status::kSuccess);
      }}

      }}  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({{global_symbol}}, _GEMM);
      """
    return text


def cutlass_codegen_batch_gemm(
    m, n, k, typea, typeb, typec, layouta, layoutb, layoutc, op_type, gemm_profiler
):
    """cutlass codegen for batch gemm"""
    op_name, cutlass_op_def = select_gemm_kernel(
        gemm_profiler,
        op_type,
        m,
        k,
        n,
        typec,
        typea,
        typeb,
        layoutc,
        layouta,
        layoutb,
        False,
        True,
        False,
        True,
    )
    op_name = "Operation_" + op_name
    typea, typeb, typec = _convert_dtype_str([typea, typeb, typec])
    layouta, layoutb, layoutc = _convert_layout_str([layouta, layoutb, layoutc])
    r_layouta, r_layoutb, r_layoutc = _reverse_layout([layouta, layoutb, layoutc])

    if op_type in ["cutlass.batch_matmul_bias", "cutlass.batch_matmul_bias_relu"]:
        bias_param = "NDArray Bias, "
        bias_var_def = f"""
        {r_layoutc}::Stride::Index ld_bias(0);
        {r_layoutc}::Stride::Index batch_stride_bias(0);
        cutlass::half_t* bias = reinterpret_cast<cutlass::half_t*>(Bias->data);
        """
        bias_arg = """{bias, ld_bias},
        batch_stride_bias"""
    else:
        bias_param = bias_var_def = ""
        bias_arg = """{c, ldc},
        batch_stride_C"""

    text = f"""
      #define CUTLASS_ENABLE_CUBLAS 1
      #define CUTLASS_NAMESPACE cutlass
      #define CUTLASS_ENABLE_TENSOR_CORE_MMA 1
      #define NDEBUG

      #include <cutlass/cutlass.h>
      #include <cutlass/gemm/device/gemm.h>
      #include <cutlass/gemm/device/gemm_batched.h>
      #include <cutlass/layout/matrix.h>
      #include <cutlass/numeric_types.h>

      #include <fstream>
      #include <iostream>
      #include <sstream>
      #include <vector>

      #define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>

      #include <tvm/runtime/logging.h>
      #include <tvm/runtime/ndarray.h>
      #include <tvm/runtime/packed_func.h>

      namespace {{

      using namespace tvm;
      using namespace tvm::runtime;

      void _BHGEMM(NDArray A, NDArray B, {bias_param}NDArray C) {{
        // A: [Batch, M, K], B: [K, N], C: [Batch, M, N]
        CHECK_EQ(A->ndim, 3);
        int bdim = B->ndim;
        CHECK_EQ(C->ndim, 3);
        CHECK_EQ(A->shape[2], B->shape[bdim - 2]);
        int Batch = A->shape[0];
        int M = A->shape[1];
        int K = A->shape[2];
        int N = B->shape[bdim - 1];
        CHECK_EQ(C->shape[0], Batch);
        CHECK_EQ(C->shape[1], M);
        CHECK_EQ(C->shape[2], N);
        // Define the GEMM operation
        {cutlass_op_def}
        {op_name} gemm_op;
        {typec} alpha(1.0);
        {typec} beta(0.0);
        {r_layouta}::Stride::Index lda(K);
        {r_layouta}::Stride::Index batch_stride_A(M * K);
        {r_layoutb}::Stride::Index ldb(N);
        {r_layoutb}::Stride::Index batch_stride_B(bdim == 2 ? 0 : K * N);
        {r_layoutc}::Stride::Index ldc(N);
        {r_layoutc}::Stride::Index batch_stride_C(M * N);
        {bias_var_def}
        {typea}* a = reinterpret_cast<{typea}*>(A->data);
        {typeb}* b = reinterpret_cast<{typeb}*>(B->data);
        {typec}* c = reinterpret_cast<{typec}*>(C->data);
        cutlass::Status status = gemm_op({{
            {{M, N, K}},     //
            {{a, lda}},      //
            batch_stride_A,   //
            {{b, ldb}},      //
            batch_stride_B,   //
            {bias_arg},      //
            {{c, ldc}},      //
            batch_stride_C,   //
            {{alpha, beta}},  //
            Batch             //
        }});
        CHECK(status == cutlass::Status::kSuccess);
      }}

      }}  // namespace
      TVM_DLL_EXPORT_TYPED_FUNC({{global_symbol}}, _BHGEMM);
      """
    return text


def cutlass_codegen_conv2d(
    d,
    w,
    out_shape,
    padding,
    strides,
    dilation,
    out_dtype,
    data_dtype,
    weight_dtype,
    op_type,
    conv2d_profiler,
):
    """cutlass codegen for conv2d"""
    # cutlass backend only supports nhwc for now
    res = handle_conv2d(
        cutlass_profiler=conv2d_profiler,
        op_type=op_type,
        d_shape=d,
        w_shape=w,
        padding=padding,
        strides=strides,
        dilation=dilation,
        out_dtype=out_dtype,
        data_dtype=data_dtype,
        weight_dtype=weight_dtype,
        use_3xtf32=False,
        split_k_slices=[1],
        profile_all_alignments=True,
        find_first_valid=False,
        use_multiprocessing=True,
    )
    cutlass_op_def = res["cutlass_op_def"]
    op_name = res["cutlass_op_name"]

    pattern_name = op_type
    ext_func_id = "cutlass_kernel"
    if op_type in ["cutlass.conv2d"]:
        nargs = 2
    elif op_type in ["cutlass.conv2d_bias"]:
        nargs = 3
    elif op_type in ["cutlass.conv2d_bias_residual_add"]:
        nargs = 4
    func_args = ["inp" + str(i) for i in range(nargs)]
    output_types = [out_dtype]
    attribute_args = {
        "op_type": op_type,
        "arg0_dtype": data_dtype,
        "arg1_dtype": weight_dtype,
        "ret_dtype": out_dtype,
        "arg0_shape": d,
        "arg1_shape": w,
        "ret_shape": out_shape,
        "strides": strides,
        "padding": padding,
        "dilation": dilation,
        "cutlass_op_name": op_name,
        "cutlass_op_def": cutlass_op_def,
    }
    code = codegen_cutlass_c_source(
        pattern_name, ext_func_id, func_args, output_types, attribute_args
    )
    return code
