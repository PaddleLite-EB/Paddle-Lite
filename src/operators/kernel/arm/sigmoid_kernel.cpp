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

#ifdef SIGMOID_OP

#include "../sigmoid_kernel.h"
#include "../central-arm-func/sigmoid_arm_func.h"
#if __ARM_NEON
#include "../../math/math_func_neon.h"
#endif
#include <cmath>
namespace paddle_mobile {
namespace operators {

using framework::DDim;
using framework::Tensor;

template <>
bool SigmoidKernel<CPU, float>::Init(SigmoidParam *param) {
  return true;
}

template <>
void SigmoidKernel<CPU, float>::Compute(const SigmoidParam &param) const {
  SigmoidCompute<float>(param);
}

template class SigmoidKernel<CPU, float>;
}  // namespace operators
}  // namespace paddle_mobile

#endif
