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
# pylint: disable=invalid-name
"""The Relax training APIs."""

from . import optimizer
from ...block_builder import BlockBuilder
from ...expr import Call, Expr
from .common import TEFunc, LegalizeFunc
from . import trainer
from . import utils
from . import loss

from .setup_trainer import SetupTrainer


def _argmax_argmin(te_func: TEFunc) -> LegalizeFunc:
    def argmax_argmin_call_te(bb: BlockBuilder, call: Call) -> Expr:
        return bb.call_te(
            te_func,
            call.args[0],
            None if call.attrs.axis is None else call.attrs.axis.value,
            call.attrs.keepdims,
        )

    return argmax_argmin_call_te


register_legalize("relax.argmax", _argmax_argmin(topi.argmax))
register_legalize("relax.argmin", _argmax_argmin(topi.argmin))
