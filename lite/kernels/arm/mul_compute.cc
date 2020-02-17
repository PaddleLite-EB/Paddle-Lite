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

#include "lite/kernels/arm/mul_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

using float16 = paddle::zynqmp::float16;

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void MulCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto& param = this->Param<param_t>();

  param.output->mutable_data<float>();

  // int channel = param.x->ZynqTensor()->shape().channel();

  zynqmp::Shape s(zynqmp::NCHW,
                  {1, 2 * param.x->ZynqTensor()->shape().channel(), 1, 1});

  half_intput_.mutableData<void>(paddle::zynqmp::FP16, s);
  half_output_.mutableData<void>(paddle::zynqmp::FP16,
                                 param.output->ZynqTensor()->shape());

  // // ====================================================
  zynqmp::FullyConnectedParam& fc_param = pe_.param();

  fc_param.input = &half_intput_;
  fc_param.output = &half_output_;
  fc_param.filter = param.y->ZynqTensor();

  fc_param.bias = &bias_;

  int channel = fc_param.filter->shape().channel();

  zynqmp::Shape bias_shape(zynqmp::N, {channel});

  float* bias_data =
      fc_param.bias->mutableData<float>(zynqmp::FP32, bias_shape);
  memset(bias_data, 0, channel * sizeof(float));
  bias_.flush();

  pe_.init();
  pe_.apply();
}

void MulCompute::Run() {
  auto& param = Param<param_t>();

  /*
  const auto* x_data = param.x->data<float>();
  const auto* y_data = param.y->data<float>();
  auto* o_data = param.output->mutable_data<float>();

  m_ = static_cast<int>(
      param.x->dims().Slice(0, param.x_num_col_dims).production());
  int x_w =
      static_cast<int>(param.x->dims()
                           .Slice(param.x_num_col_dims, param.x->dims().size())
                           .production());
  int y_h = static_cast<int>(
      param.y->dims().Slice(0, param.y_num_col_dims).production());
  n_ = static_cast<int>(param.y->dims()
                            .Slice(param.y_num_col_dims, param.y->dims().size())
                            .production());

  CHECK_EQ(x_w, y_h) << "x_w must be equal with y_h";
  k_ = x_w;
  auto& ctx = this->ctx_->template As<ARMContext>();
  if (n_ == 1) {
    lite::arm::math::sgemv(
        x_data, y_data, o_data, false, m_, k_, false, nullptr, false, &ctx);

  } else {
    constexpr bool is_tranposed_y = false;
    int hblock = lite::arm::math::get_hblock(&ctx);
    int m_round = hblock * ((m_ + hblock - 1) / hblock);
    ctx.ExtendWorkspace(m_round * k_ * sizeof(float));

    float* packed_x = static_cast<float*>(ctx.workspace_data<float>()) +
                      ctx.llc_size() / sizeof(float);
    lite::arm::math::prepackA(
        packed_x, x_data, 1.f, k_, 0, m_, 0, k_, false, &ctx);
    int ldb = n_;
    if (is_tranposed_y) {
      ldb = k_;
    }
    lite::arm::math::sgemm_prepack(is_tranposed_y,
                                   m_,
                                   n_,
                                   k_,
                                   packed_x,
                                   y_data,
                                   ldb,
                                   0.f,
                                   o_data,
                                   n_,
                                   nullptr,
                                   false,
                                   false,
                                   &ctx);
  }

*/

  // param.output->ZynqTensor()->saveToFile("cpu", true);

  param.x->ZynqTensor()->flush();
  param.x->ZynqTensor()->setDataType(zynqmp::FP32);
  param.x->ZynqTensor()->flush();
   param.x->ZynqTensor()->saveToFile("x", true);

  half_intput_.copyFrom(param.x->ZynqTensor());
  half_intput_.invalidate();
  half_intput_.saveToFile("half", true);
  pe_.dispatch();
  half_output_.flush();
   param.output->ZynqTensor()->copyFrom(&half_output_);

   //param.x->ZynqTensor()->saveToFile("x", true);
  half_output_.saveToFile("half",true);

  exit(-1);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    mul, kARM, kFloat, kNCHW, paddle::lite::kernels::arm::MulCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
