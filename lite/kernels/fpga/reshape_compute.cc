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

#include <vector>

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/kernels/fpga/reshape_compute.h"
#include "lite/operators/reshape_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void FlattenCompute::PrepareForRun() {
  auto& param = Param<operators::ReshapeParam>();
  auto output = param.output;
  output->mutable_data<float16>();
  auto* input = param.x->ZynqTensor();
  output->ZynqTensor()->maxIndex(true);

  zynqmp::BypassParam& bypass_param = bypass_pe_.param();
  bypass_param.input = input;
  bypass_param.output = output->ZynqTensor();
  bypass_pe_.init();
  bypass_pe_.apply();
  if (input->aligned() && input->shape().shouldAlign()) {
    cpu_pe_.reset(new zynqmp::CPUPE());
    zynqmp::CPUParam& cpu_param = cpu_pe_->param();
    cpu_param.outputs.push_back(output->ZynqTensor());
    cpu_pe_->init();
    cpu_pe_->apply();
  }
}

void FlattenCompute::Run() {
  auto& param = Param<operators::ReshapeParam>();
  auto x = param.x;
  auto output = param.output;

  bypass_pe_.dispatch();

  auto output_dims = output->dims();
  if (param.inplace) {
    output->ShareDataWith(*x);
  } else {
    // output->CopyDataFrom(*x);
    output->ZynqTensor()->writeMax(x->ZynqTensor()->max()[0]);
  }

  if (cpu_pe_) {
    cpu_pe_->dispatch();
    output->ZynqTensor()->invalidate();
    output->ZynqTensor()->setAligned(true);
    output->ZynqTensor()->unalignImage();
    output->ZynqTensor()->flush();
    output->ZynqTensor()->setAligned(false);
  }

  output->ZynqTensor()->flush();
  output->Resize(output_dims);

#ifdef FPGA_PRINT_TENSOR
  Debugger::get_instance().registerOutput("flatten", output->ZynqTensor());
#endif
}

void ReshapeCompute::PrepareForRun() {
  auto& param = Param<operators::ReshapeParam>();
  auto output = param.output;

  auto* input = param.x->ZynqTensor();

  if (input->dataType() == zynqmp::FP16) {
    output->mutable_data<float16>();
  } else {
    output->mutable_data<float>();
  }

  zynqmp::BypassParam& bypass_param = bypass_pe_.param();
  bypass_param.input = input;
  bypass_param.output = output->ZynqTensor();
  bypass_pe_.init();
  bypass_pe_.apply();
  // if (input->aligned() && input->shape().shouldAlign()) {
  cpu_pe_.reset(new zynqmp::CPUPE());
  zynqmp::CPUParam& cpu_param = cpu_pe_->param();
  cpu_param.outputs.push_back(output->ZynqTensor());
  cpu_pe_->init();
  cpu_pe_->apply();
}

void ReshapeCompute::Run() {
  auto& param = Param<operators::ReshapeParam>();
  auto x = param.x;
  auto output = param.output;

  x->ZynqTensor()->unalignImage();
  x->ZynqTensor()->flush();

  bypass_pe_.dispatch();

  if (cpu_pe_) {
    cpu_pe_->dispatch();
    output->ZynqTensor()->invalidate();
    // output->ZynqTensor()->setAligned(true);
    output->ZynqTensor()->unalignImage();
    output->ZynqTensor()->flush();
    output->ZynqTensor()->setAligned(false);
  }

  output->ZynqTensor()->flush();

#ifdef FPGA_PRINT_TENSOR
  Debugger::get_instance().registerOutput("reshape", output->ZynqTensor());
#endif
}

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reshape,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ReshapeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(reshape2,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::ReshapeCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::FlattenCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(flatten2,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::FlattenCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("Shape",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .BindOutput("XShape",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
