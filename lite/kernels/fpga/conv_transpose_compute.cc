/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/kernels/fpga/conv_transpose_compute.h"
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/backends/fpga/KD/pes/transposed_conv_pe.hpp"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using lite_api::ActivationType;

void Conv2DTransposeCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  param.output->mutable_data<float16>();
  int pad_h = (*param.paddings)[0];
  int pad_w = (*param.paddings)[2];

  zynqmp::ActivationType active_type = zynqmp::TYPE_NONE;
  float leaky_relu_factor = 0;

  switch (param.activation_param.active_type) {
    case ActivationType::kIndentity:
      active_type = zynqmp::TYPE_NONE;
      break;
    case ActivationType::kRelu:
      active_type = zynqmp::TYPE_RELU;
      break;
    case ActivationType::kRelu6:
      active_type = zynqmp::TYPE_RELU6;
      break;
    case ActivationType::kPRelu:
    case ActivationType::kLeakyRelu:
      active_type = zynqmp::TYPE_LEAKY_RELU;
      leaky_relu_factor = param.activation_param.Leaky_relu_alpha;
      break;
    case ActivationType::kSigmoid:
      active_type = zynqmp::TYPE_SIGMOID;
      break;
    case ActivationType::kTanh:
    case ActivationType::kSwish:
    case ActivationType::kExp:
    case ActivationType::kAbs:
    case ActivationType::kHardSwish:
    case ActivationType::kReciprocal:
    default:
      throw("not supported activation");
      break;
  }

  zynqmp::ConvParam& conv_param = transposed_conv_pe_.param();
  conv_param.input = param.x->ZynqTensor();
  conv_param.output = param.output->ZynqTensor();
  conv_param.filter = param.filter->ZynqTensor();
  conv_param.filter->setDataType(zynqmp::FP32);
  conv_param.groups = param.groups;
  conv_param.strides = param.strides;
  conv_param.paddings = std::vector<int>({pad_h, pad_w});
  conv_param.dilations = *param.dilations;
  fill_scale_bias_const(&conv_param);
  if (param.bias != nullptr) {
    conv_param.bias()->copyFrom(param.bias->ZynqTensor());
  }

  conv_param.activeParam.type = active_type;
  conv_param.activeParam.leaky_relu_factor = leaky_relu_factor;

  transposed_conv_pe_.init();
  transposed_conv_pe_.apply();
}

void Conv2DTransposeCompute::Run() {
  auto& param = this->Param<param_t>();
  transposed_conv_pe_.dispatch();

#ifdef FPGA_PRINT_TENSOR
  zynqmp::ConvParam& conv_param = transposed_conv_pe_.param();
  Debugger::get_instance().registerOutput("conv2d_transpose",
                                          conv_param.output);
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(conv2d_transpose,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::Conv2DTransposeCompute,
                     def)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
