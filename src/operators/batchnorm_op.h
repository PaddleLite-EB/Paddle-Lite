/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef BATCHNORM_OP

#pragma once

#include <string>
#include "framework/operator.h"
#include "operators/kernel/batchnorm_kernel.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {
using std::string;
template <typename DeviceType, typename T>
class BatchNormOp
    : public framework::OperatorWithKernel<DeviceType,
                                           BatchNormParam<DeviceType>,
                                           BatchNormKernel<DeviceType, T>> {
 public:
  BatchNormOp(const string &type, const VariableNameMap &inputs,
              const VariableNameMap &outputs,
              const framework::AttributeMap &attrs,
              std::shared_ptr<framework::Scope> scope)
      : framework::OperatorWithKernel<DeviceType, BatchNormParam<DeviceType>,
                                      BatchNormKernel<DeviceType, T>>(
            type, inputs, outputs, attrs, scope) {}

  void InferShape() const override;

 protected:
};

}  // namespace operators
}  // namespace paddle_mobile

#ifdef PADDLE_MOBILE_CPU
USE_OP_CPU(batch_norm);
#endif
#ifdef PADDLE_MOBILE_MALI_GPU
USE_OP_MALI_GPU(batch_norm);
#endif
#ifdef PADDLE_MOBILE_FPGA
#endif
#ifdef PADDLE_MOBILE_CL
USE_OP_CL(batch_norm);
#endif

#endif
