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

import tvm
import tvm.testing
from tvm.relax.transform import DeadCodeElimination
from tvm.script.parser import ir as I, relax as R


def verify(input, expected):
    tvm.ir.assert_structural_equal(DeadCodeElimination()(input), expected)


def test_simple():
    @tvm.script.ir_module
    class Input:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            # block 0
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                gv21: R.Tensor((2, 4, 26, 26), dtype="float32") = R.permute_dims(
                    gv2, axes=[0, 3, 1, 2]
                )
                gv22: R.Tensor((2, 4, 26, 26), dtype="float32") = R.add(gv21, bias)
                R.output(gv2)
            return gv2

    @I.ir_module
    class Expected:
        @R.function
        def main(
            x: R.Tensor((2, 3, 28, 28), dtype="float32"),
            w: R.Tensor((4, 3, 3, 3), dtype="float32"),
            bias: R.Tensor((26, 26), dtype="float32"),
        ) -> R.Tensor((2, 4, 26, 26), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((2, 28, 28, 3), dtype="float32") = R.permute_dims(x, axes=[0, 2, 3, 1])
                gv1: R.Tensor((4, 3, 3, 3), dtype="float32") = R.permute_dims(w, axes=[0, 2, 3, 1])
                gv2: R.Tensor((2, 26, 26, 4), dtype="float32") = R.nn.conv2d(
                    gv,
                    gv1,
                    strides=[1, 1],
                    padding=[0, 0, 0, 0],
                    dilation=[1, 1],
                    groups=1,
                    data_layout="NHWC",
                    kernel_layout="OHWI",
                    out_layout="NHWC",
                    out_dtype="float32",
                )
                R.output(gv2)
            return gv2

    verify(Input, Expected)


if __name__ == "__main__":
    tvm.testing.main()
