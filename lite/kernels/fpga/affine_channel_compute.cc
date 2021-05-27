// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/fpga/affine_channel_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

void AffineChannelCompute::PrepareForRun() {
  zynqmp::ScaleParam& scale_param = pe_.param();
    auto& param = Param<operators::AffineChannelParam>();
  const lite::Tensor* scale = param.Scale;
  const lite::Tensor* bias = param.Bias;

  param.Out->mutable_data<float16>();

  scale_param.input = param.X->ZynqTensor();
  scale_param.output = param.Out->ZynqTensor();
  scale_param.activeParam.type = zynqmp::TYPE_NONE;

  int channel = scale_param.input->shape().channel();
  scale_param.scale = &scale_;
  scale_param.bias = &bias_;
  zynqmp::Shape shape(zynqmp::N, {channel});
  zynqmp::float16* scale_data =
      scale_.mutableData<zynqmp::float16>(zynqmp::FP16, shape);
  zynqmp::float16* bias_data =
      bias_.mutableData<zynqmp::float16>(zynqmp::FP16, shape);
  
  scale_.copyFrom(param.Scale->ZynqTensor());
  bias_.copyFrom(param.Bias->ZynqTensor());

  pe_.init();
  pe_.apply();
}

void AffineChannelCompute::Run() {
  pe_.dispatch();
  return;
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(affine_channel,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::AffineChannelCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .Finalize();
