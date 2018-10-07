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

#include "op_param.h"
namespace paddle_mobile {
namespace operators {

#ifdef CONV_OP
template <>
Print &operator<<(Print &printer, const ConvParam<CPU> &conv_param) {
  printer << "parameter of conv: "
          << "\n";
  printer << "  stride: "
          << " (" << conv_param.Strides()[0] << conv_param.Strides()[1] << ") "
          << "\n";
  printer << "  paddings: "
          << " (" << conv_param.Paddings()[0] << conv_param.Paddings()[1]
          << ") "
          << "\n";
  printer << "  dilations: "
          << " (" << conv_param.Dilations()[0] << conv_param.Dilations()[1]
          << ") "
          << "\n";
  printer << "  groups: " << conv_param.Groups() << "\n";
  printer << "  input  dims: " << conv_param.Input()->dims() << "\n";
  printer << "  filter dims: " << conv_param.Filter()->dims() << "\n";
  printer << "  output dims: " << conv_param.Output()->dims();
  return printer;
}

template class ConvParam<CPU>;
template class ConvParam<FPGA>;
template class ConvParam<GPU_MALI>;
template class ConvParam<X86>;
#endif

template class ElementwiseAddParam<CPU>;
template class ElementwiseAddParam<FPGA>;
template class ElementwiseAddParam<GPU_MALI>;
template class ElementwiseAddParam<X86>;

#ifdef MUL_OP
template class MulParam<CPU>;
template class MulParam<FPGA>;
template class MulParam<GPU_MALI>;
template class MulParam<X86>;
#endif

#ifdef CONCAT_OP
template class ConcatParam<CPU>;
template class ConcatParam<FPGA>;
template class ConcatParam<GPU_MALI>;
template class ConcatParam<X86>;
#endif

#ifdef LRN_OP
template class LrnParam<CPU>;
template class LrnParam<FPGA>;
template class LrnParam<GPU_MALI>;
template class LrnParam<X86>;
#endif

#ifdef FUSION_CONVADD_OP

Print &operator<<(Print &printer, const FusionConvAddParam<CPU> &conv_param) {
  printer << "parameter of conv_add: "
          << "\n";
  printer << "  stride: "
          << " (" << conv_param.Strides()[0] << conv_param.Strides()[1] << ") "
          << "\n";
  printer << "  paddings: "
          << " (" << conv_param.Paddings()[0] << conv_param.Paddings()[1]
          << ") "
          << "\n";
  printer << "  dilations: "
          << " (" << conv_param.Dilations()[0] << conv_param.Dilations()[1]
          << ") "
          << "\n";
  printer << "  groups: " << conv_param.Groups() << "\n";
  printer << "  input  dims: " << conv_param.Input()->dims() << "\n";
  printer << "  filter dims: " << conv_param.Filter()->dims() << "\n";
  printer << "  bias dims: " << conv_param.Bias()->dims() << "\n";
  printer << "  output dims: " << conv_param.Output()->dims();
  return printer;
}

#endif

}  // namespace operators
}  // namespace paddle_mobile
