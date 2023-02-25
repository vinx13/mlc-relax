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

from __future__ import annotations
import tempfile

from tvm import relax, runtime
import tvm
import tvm.testing
from tvm import relax
import numpy as np
from tvm.relax.vm import build as relax_build
from tvm.relax.transform import LegalizeOps
from tvm.script.ir_builder import relax as R
from tvm.script.ir_builder import ir as I
from tvm.script.ir_builder import tir as T
from tvm.script.ir_builder import IRBuilder

from tvm.relax.library import get_cutlass_pattern, cutlass_fcodegen

A_TYPE = "float16"
B_TYPE = "float16"
C_TYPE = "float16"

target = "cuda"


def f_run(rt_mod: runtime.Module, device: runtime.ndarray.Device, *input):
    vm = relax.vm.VirtualMachine(exec=rt_mod, device=device)
    return vm["main"](*input)


def build(mod):
    mod = relax.transform.LegalizeOps()(mod)
    mod = relax.transform.AnnotateTIROpPattern()(mod)
    mod = relax.transform.FuseOps()(mod)
    mod = relax.transform.FuseTIR()(mod)
    mod = relax.transform.SplitCallTIRByPattern(get_cutlass_pattern(), cutlass_fcodegen())(mod)
    mod = relax.transform.RemoveUnusedFunctions()(mod)
    print(mod.script())
    f = tempfile.NamedTemporaryFile(suffix=".so", delete=True)
    executable = relax_build(mod, target)
    executable.mod.export_library(f.name, cc="nvcc")
    rt_mod = runtime.load_module(f.name)
    f.close()
    return rt_mod


def constructGEMM(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense():
    m, n, k = 128, 128, 128
    executable = build(constructGEMM(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructGEMM_bias(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense_bias():
    m, n, k = 128, 128, 128
    executable = build(constructGEMM_bias(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructGEMM_bias_relu(M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.nn.relu(D))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_dense_bias_relu():
    m, n, k = 128, 128, 128
    executable = build(constructGEMM_bias_relu(m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), np.maximum(A @ B + bias, 0), rtol=1e-2)


def constructBatchGEMM(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense():
    b, m, n, k = 2, 128, 128, 128
    executable = build(constructBatchGEMM(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM2(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((batch, K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense2():
    b, m, n, k = 2, 128, 128, 128
    executable = build(constructBatchGEMM2(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    result = f_run(executable, dev, A_tvm, B_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B, rtol=1e-2)


def constructBatchGEMM_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense_bias():
    b, m, n, k = 2, 128, 128, 128
    executable = build(constructBatchGEMM_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructBatchGEMM2_bias(batch, M, N, K):
    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                A = R.arg(
                    "A", relax.TensorStructInfo((batch, M, K), A_TYPE)
                )  # pylint: disable=invalid-name
                B = R.arg(
                    "B", relax.TensorStructInfo((batch, K, N), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, N), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(R.matmul(A, B, out_dtype=C_TYPE))
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    relax_mod = ib.get()
    return relax_mod


@tvm.testing.requires_cutlass
def test_cutlass_batch_dense2_bias():
    b, m, n, k = 2, 128, 128, 128
    executable = build(constructBatchGEMM2_bias(b, m, n, k))
    dev = tvm.cuda()
    A = np.random.rand(b, m, k).astype("float16") * 5
    B = np.random.rand(b, k, n).astype("float16") * 5
    bias = np.random.rand(1, n).astype("float16") * 20
    A_tvm = tvm.nd.array(A, dev)
    B_tvm = tvm.nd.array(B, dev)
    bias_tvm = tvm.nd.array(bias, dev)
    result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
    np.testing.assert_allclose(result.numpy(), A @ B + bias, rtol=1e-2)


def constructConv2D(N, C, H, W, KH, KW, O, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg(
                    "x", relax.TensorStructInfo((N, H, W, C), A_TYPE)
                )  # pylint: disable=invalid-name
                w = R.arg(
                    "w", relax.TensorStructInfo((O, KH, KW, C), B_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(
                        R.nn.conv2d(
                            x,
                            w,
                            strides=strides,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            data_layout="NHWC",
                            kernel_layout="OHWI",
                            out_layout="NHWC",
                            out_dtype=C_TYPE,
                        )
                    )
                    R.output(C)
                (C,) = df.output_vars
                R.func_ret_value(C)
    mod = ib.get()
    return mod


@tvm.testing.requires_cutlass
def test_cutlass_conv2d():
    import torch

    n, c, h, w = 1, 3, 224, 224
    kh, kw, o = 3, 3, 64
    counter = 0
    for strides in [(1, 1), (2, 2)]:
        for padding in [(0, 0), (3, 3)]:
            for dilation in [(1, 1), (4, 4)]:
                executable = build(
                    constructConv2D(n, c, h, w, kh, kw, o, strides, padding, dilation)
                )
                dev = tvm.cuda()
                np.random.seed(0)
                A = np.random.rand(n, h, w, c).astype("float16")
                B = np.random.rand(o, kh, kw, c).astype("float16")
                A_tvm = tvm.nd.array(A, dev)
                B_tvm = tvm.nd.array(B, dev)
                result = f_run(executable, dev, A_tvm, B_tvm)
                A_torch = torch.from_numpy(np.transpose(A, (0, 3, 1, 2))).to(
                    torch.float32
                )  # .cuda()
                B_torch = torch.from_numpy(np.transpose(B, (0, 3, 1, 2))).to(
                    torch.float32
                )  # .cuda()
                C_torch = torch.nn.functional.conv2d(
                    A_torch, B_torch, stride=strides, padding=padding, dilation=dilation
                )
                np.testing.assert_allclose(
                    np.transpose(result.numpy(), (0, 3, 1, 2)), C_torch.cpu().numpy(), rtol=1e-2
                )
                counter += 1


def constructConv2D_bias(N, C, H, W, KH, KW, O, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg(
                    "x", relax.TensorStructInfo((N, H, W, C), A_TYPE)
                )  # pylint: disable=invalid-name
                w = R.arg(
                    "w", relax.TensorStructInfo((O, KH, KW, C), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, 1, 1, O), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(
                        R.nn.conv2d(
                            x,
                            w,
                            strides=strides,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            data_layout="NHWC",
                            kernel_layout="OHWI",
                            out_layout="NHWC",
                            out_dtype=C_TYPE,
                        )
                    )
                    D = R.emit(R.add(C, bias))
                    R.output(D)
                (D,) = df.output_vars
                R.func_ret_value(D)
    mod = ib.get()
    return mod


@tvm.testing.requires_cutlass
def test_cutlass_conv2d_bias():
    import torch

    n, c, h, w = 1, 3, 224, 224
    kh, kw, o = 3, 3, 64
    counter = 0
    for strides in [(1, 1), (2, 2)]:
        for padding in [(0, 0), (3, 3)]:
            for dilation in [(1, 1), (4, 4)]:
                filename = "/tmp/" + "test_transform_cutlass_codegen" + str(counter) + ".so"
                executable = build(
                    constructConv2D_bias(n, c, h, w, kh, kw, o, strides, padding, dilation),
                )
                dev = tvm.cuda()
                np.random.seed(0)
                A = np.random.rand(n, h, w, c).astype("float16") * 5
                B = np.random.rand(o, kh, kw, c).astype("float16") * 5
                bias = np.random.rand(o).astype("float16") * 5
                A_tvm = tvm.nd.array(A, dev)
                B_tvm = tvm.nd.array(B, dev)
                bias_tvm = tvm.nd.array(bias.reshape(1, 1, 1, o), dev)
                result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm)
                A_torch = torch.from_numpy(np.transpose(A, (0, 3, 1, 2))).to(
                    torch.float32
                )  # .cuda()
                B_torch = torch.from_numpy(np.transpose(B, (0, 3, 1, 2))).to(
                    torch.float32
                )  # .cuda()
                bias_torch = torch.from_numpy(bias).to(torch.float32)  # .cuda()
                C_torch = torch.nn.functional.conv2d(
                    A_torch,
                    B_torch,
                    bias=bias_torch,
                    stride=strides,
                    padding=padding,
                    dilation=dilation,
                )
                np.testing.assert_allclose(
                    np.transpose(result.numpy(), (0, 3, 1, 2)), C_torch.cpu().numpy(), rtol=1e-2
                )
                counter += 1


def constructConv2D_bias_add(N, C, H, W, KH, KW, O, OH, OW, strides, padding, dilation):
    from tvm.script.ir_builder import IRBuilder
    from tvm.script.ir_builder import ir as I
    from tvm.script.ir_builder import relax as R
    from tvm.script.ir_builder import tir as T

    with IRBuilder() as ib:  # pylint: disable=invalid-name
        with I.ir_module() as frame:
            with R.function():
                R.func_name("main")
                x = R.arg(
                    "x", relax.TensorStructInfo((N, H, W, C), A_TYPE)
                )  # pylint: disable=invalid-name
                w = R.arg(
                    "w", relax.TensorStructInfo((O, KH, KW, C), B_TYPE)
                )  # pylint: disable=invalid-name
                bias = R.arg(
                    "bias", relax.TensorStructInfo((1, 1, 1, O), A_TYPE)
                )  # pylint: disable=invalid-name
                res = R.arg(
                    "res", relax.TensorStructInfo((N, OH, OW, O), A_TYPE)
                )  # pylint: disable=invalid-name
                with R.dataflow() as df:
                    C = R.emit(
                        R.nn.conv2d(
                            x,
                            w,
                            strides=strides,
                            padding=padding,
                            dilation=dilation,
                            groups=1,
                            data_layout="NHWC",
                            kernel_layout="OHWI",
                            out_layout="NHWC",
                            out_dtype=C_TYPE,
                        )
                    )
                    D = R.emit(R.add(C, bias))
                    E = R.emit(R.add(D, res))
                    R.output(E)
                (E,) = df.output_vars
                R.func_ret_value(E)
    mod = ib.get()
    return mod


@tvm.testing.requires_cutlass
def test_cutlass_conv2d_bias_add():
    import torch

    n, c, h, w = 1, 8, 224, 224
    kh, kw, o = 3, 3, 64
    counter = 0
    for strides in [(1, 1), (2, 2)]:
        for padding in [(0, 0), (3, 3)]:
            for dilation in [(1, 1), (4, 4)]:
                oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // strides[0] + 1
                ow = (w + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // strides[1] + 1
                executable = build(
                    constructConv2D_bias_add(
                        n, c, h, w, kh, kw, o, oh, ow, strides, padding, dilation
                    )
                )
                dev = tvm.cuda()
                np.random.seed(0)
                A = np.random.rand(n, h, w, c).astype("float16")
                B = np.random.rand(o, kh, kw, c).astype("float16")
                bias = np.random.rand(o).astype("float16")
                res = np.random.rand(n, oh, ow, o).astype("float16")
                A_tvm = tvm.nd.array(A, dev)
                B_tvm = tvm.nd.array(B, dev)
                bias_tvm = tvm.nd.array(bias.reshape(1, 1, 1, o), dev)
                res_tvm = tvm.nd.array(res, dev)
                result = f_run(executable, dev, A_tvm, B_tvm, bias_tvm, res_tvm)
                A_torch = torch.from_numpy(np.transpose(A, (0, 3, 1, 2))).to(
                    torch.float32
                )  # .cuda()
                B_torch = torch.from_numpy(np.transpose(B, (0, 3, 1, 2))).to(
                    torch.float32
                )  # .cuda()
                bias_torch = torch.from_numpy(bias).to(torch.float32)  # .cuda()
                res_torch = torch.from_numpy(np.transpose(res, (0, 3, 1, 2))).to(
                    torch.float32
                )  # .cuda()
                C_torch = torch.nn.functional.conv2d(
                    A_torch,
                    B_torch,
                    bias=bias_torch,
                    stride=strides,
                    padding=padding,
                    dilation=dilation,
                )
                D_torch = C_torch + res_torch
                np.testing.assert_allclose(
                    np.transpose(result.numpy(), (0, 3, 1, 2)),
                    D_torch.cpu().numpy(),
                    rtol=1e-2,
                )
                counter += 1


if __name__ == "__main__":
    test_cutlass_dense()
    test_cutlass_dense_bias()
    test_cutlass_dense_bias_relu()
    test_cutlass_batch_dense()
    test_cutlass_batch_dense2()
    test_cutlass_batch_dense_bias()
    test_cutlass_batch_dense2_bias()
    test_cutlass_conv2d()
    test_cutlass_conv2d_bias()
    test_cutlass_conv2d_bias_add()
