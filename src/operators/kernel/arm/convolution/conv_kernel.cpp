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

#ifdef CONV_OP

#include "operators/kernel/conv_kernel.h"
#include "operators/kernel/arm/convolution/conv_common.h"
#include "operators/kernel/central-arm-func/conv_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ConvKernel<CPU, float>::Init(ConvParam<CPU> *param) {
  InitBaseConvKernel(param);
  return true;
}

template <>
void ConvKernel<CPU, float>::Compute(const ConvParam<CPU> &param) {
  switch (param.ExecMode()) {
    case ConvParam<CPU>::EXEC_GEMM_INT8:
      GemmConv<int8_t, int32_t>(param);
      break;
#ifndef __aarch64__
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3_INT8:
      DepthwiseConv3x3<int8_t, int32_t>(param);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE5x5_INT8:
      DepthwiseConv5x5<int8_t, int32_t>(param);
      break;
#endif  // __aarch64__
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S1P1_FLOAT:
      math::DepthwiseConv3x3s1p1(param.Input(), param.Filter(), param.Output(),
                                 nullptr, false, false);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S2P1_FLOAT:
      math::DepthwiseConv3x3s2p1v2(param.Input(), param.Filter(),
                                   param.Output(), nullptr, false, false);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3S2P0_FLOAT:
      math::DepthwiseConv3x3s2p0(param.Input(), param.Filter(), param.Output(),
                                 nullptr, false, false);
      break;
    case ConvParam<CPU>::EXEC_DEPTHWISE3x3_FLOAT:
      math::DepthwiseConv3x3(param.Input(), param.Strides(), param.Paddings(),
                             param.Filter(), nullptr, param.Output(), false);
      break;
#ifndef __aarch64__
    case ConvParam<CPU>::EXEC_DEPTHWISE5x5_FLOAT:
      DepthwiseConv5x5<float, float>(param);
      break;
    case ConvParam<CPU>::EXEC_WINOGRAD3X3_FLOAT:
      WinogradConv3x3<8, 3>(param);
      break;
#endif  // __aarch64__
    case ConvParam<CPU>::EXEC_GEMM_FLOAT:
      GemmConv<float, float>(param);
      break;
    default:
      PADDLE_MOBILE_THROW_EXCEPTION("Invalid convolution execute mode %d",
                                    param.ExecMode());
  }
}

template class ConvKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
