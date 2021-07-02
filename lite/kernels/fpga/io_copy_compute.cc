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

#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/target_wrapper.h"
#include "lite/core/type_system.h"
#include "lite/backends/fpga/KD/pes/bypass_pe.hpp"
#include "lite/backends/fpga/KD/pes/cpu_pe.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;
using BypassParam = zynqmp::BypassParam;
using CPUPE = zynqmp::CPUPE;
using BypassPE = zynqmp::BypassPE;

void copy_properties(operators::IoCopyParam& param) {  // NOLINT
  param.y->set_persistable(param.x->persistable());
  auto out_lod = param.y->mutable_lod();
  *out_lod = param.x->lod();
  param.y->ZynqTensor()->copyScaleFrom(param.x->ZynqTensor());
  param.y->ZynqTensor()->copyMaxFrom(param.x->ZynqTensor());
}

template <typename T>
void chw_to_hwc(
    T* hwc_data, T* chw_data, int num, int channel, int height, int width) {
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
          hwc_data[n * chw + h * wc + w * channel + c] = chw_data[index];
          index++;
        }
      }
    }
  }
}

template <typename T>
void hwc_to_chw(
    T* chw_data, T* hwc_data, int num, int channel, int height, int width) {
  // channel == 1 || width == 1 直接拷贝，优化性能
  if (channel == 1 || width == 1) {
    memcpy(chw_data, hwc_data, num * channel * height * width * sizeof(T));
    return;
  }
  int chw = channel * height * width;
  int wc = width * channel;
  int wh = width * height;
  int index = 0;
  for (int n = 0; n < num; n++) {
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          chw_data[n * chw + c * wh + h * width + w] = hwc_data[index];
          index++;
        }
      }
    }
  }
}

/*
 * This kernel copies a tensor from host to FPGA space.
 * host is fp32 or int32，bypass fp32->fp16 ,cpu dispatch
 */
class IoCopyHostCHWToFpgaHWCCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void PrepareForRun() {
    auto& param = Param<operators::IoCopyParam>();
    param.x->ZynqTensor()->syncToDevice();
    param.y->mutable_data<float16>();
    param.y->ZynqTensor()->setDataType(zynqmp::FP16);
    param.y->ZynqTensor()->setDataLocation(zynqmp::Device);
    param.y->ZynqTensor()->setAligned(true);

    chw_fp16_.mutableData<float16>(zynqmp::FP16,
                                   param.y->ZynqTensor()->shape());
    param.y->ZynqTensor()->mutableData<float16>(zynqmp::FP16,
                                                param.y->ZynqTensor()->shape());

    BypassParam& out_param = bypass_pe_.param();

    out_param.input = param.x->ZynqTensor();
    out_param.output = &chw_fp16_;
    bypass_pe_.init();
    bypass_pe_.apply();

    cpu_pe_.init();
    cpu_pe_.apply();
  }

  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kFPGA));
    param.x->ZynqTensor()->flush();

    if (param.x->ZynqTensor()->dataType() == zynqmp::INT32) {
      param.y->mutable_data<int>();
      param.y->ZynqTensor()->copyFrom(param.x->ZynqTensor());
      param.y->ZynqTensor()->flush();
      copy_properties(param);
      return;
    }

    // FP32 -> FP16
    bypass_pe_.dispatch();
    cpu_pe_.dispatch();
    chw_fp16_.invalidate();
    auto dims = param.y->ZynqTensor()->shape();
    float16* chw_fp16_data = chw_fp16_.data<float16>();
    float16* hwc_fp16_data = param.y->mutable_data<float16>();
    // chw->hwc
    chw_to_hwc<float16>(hwc_fp16_data,
                        chw_fp16_data,
                        dims.num(),
                        dims.channel(),
                        dims.height(),
                        dims.width());
    param.y->ZynqTensor()->alignImage();
    param.y->ZynqTensor()->copyScaleFrom(&chw_fp16_);
    copy_properties(param);
  }

  std::string doc() const override { return "Copy IO from HOST to FPGA"; }

  private:
    zynqmp::Tensor chw_fp16_;
    CPUPE cpu_pe_;
    BypassPE bypass_pe_;
};

/*
 * This kernel copies a tensor from FPGA to host space.
 * host is fp32 ，bypass fp16->fp32 ,cpu dispatch
 */
class IoCopyFpgaToHostCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void PrepareForRun() {
    auto& param = Param<operators::IoCopyParam>();
    param.x->ZynqTensor()->syncToDevice();
    param.y->mutable_data<float>();
    param.y->ZynqTensor()->setDataType(zynqmp::FP32);
    param.y->ZynqTensor()->setDataLocation(zynqmp::CPU);

    hwc_fp32_.mutableData<float>(zynqmp::FP32, param.y->ZynqTensor()->shape());
    param.y->ZynqTensor()->mutableData<float>(zynqmp::FP32,
                                              param.y->ZynqTensor()->shape());

    BypassParam& out_param = bypass_pe_.param();

    out_param.input = param.x->ZynqTensor();
    out_param.output = &hwc_fp32_;
    bypass_pe_.init();
    bypass_pe_.apply();

    cpu_pe_.init();
    cpu_pe_.apply();
  }

  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kFPGA));

    // FP32 -> FP16
    bypass_pe_.dispatch();
    cpu_pe_.dispatch();
    hwc_fp32_.invalidate();
    auto dims = param.y->ZynqTensor()->shape();
    // unalign
    hwc_fp32_.unalignImage();
    float* hwc_fp32_data = hwc_fp32_.data<float>();
    float* chw_fp32_data = param.y->mutable_data<float>();
    // chw->hwc
    hwc_to_chw<float>(chw_fp32_data,
                      hwc_fp32_data,
                      dims.num(),
                      dims.channel(),
                      dims.height(),
                      dims.width());

    param.y->ZynqTensor()->copyScaleFrom(&hwc_fp32_);
    copy_properties(param);
  }
  std::string doc() const override { return "Copy IO from FPGA to HOST"; }

  private:
    zynqmp::Tensor hwc_fp32_;
    CPUPE cpu_pe_;
    BypassPE bypass_pe_;
};

class IoCopyFpgaToHostCHWCompute
    : public KernelLite<TARGET(kFPGA), PRECISION(kAny), DATALAYOUT(kAny)> {
 public:
  void PrepareForRun() {
    auto& param = Param<operators::IoCopyParam>();
    param.x->ZynqTensor()->syncToDevice();
    param.y->mutable_data<float>();
    // param.y->ZynqTensor()->setDataType(zynqmp::FP16);
    // param.y->ZynqTensor()->setDataLocation(zynqmp::Device);

    // hwc_.Resize(param.y->dims());
    hwc_fp32_.mutableData<float>(zynqmp::FP32, param.y->ZynqTensor()->shape());
    // param.y->ZynqTensor()->mutableData<float>(zynqmp::FP32,
    // param.y->ZynqTensor()->shape());
    // param.y->ZynqTensor()->setDataType(zynqmp::FP32);

    hwc_fp32_.setDataLocation(zynqmp::CPU);
    param.y->ZynqTensor()->setDataLocation(zynqmp::CPU);

    BypassParam& bypass_param = bypass_pe_.param();
    bypass_param.input = param.x->ZynqTensor();
    bypass_param.output = &hwc_fp32_;
    bypass_pe_.init();
    bypass_pe_.apply();

    cpu_pe_.init();
    cpu_pe_.apply();
  }

  void Run() override {
    auto& param = Param<operators::IoCopyParam>();
    CHECK(param.x->target() == TARGET(kHost) ||
          param.x->target() == TARGET(kFPGA));

    param.x->ZynqTensor()->syncToDevice();
    // if (param.x->ZynqTensor()->dataType() == zynqmp::INT32) {
    //   param.y->mutable_data<int32_t>();
    //   param.y->ZynqTensor()->copyFrom(param.x->ZynqTensor());
    //   return;
    // }
    // FP16 -> FP32
    cpu_pe_.dispatch();
    bypass_pe_.dispatch();
    hwc_fp32_.invalidate();
    // unalign
    hwc_fp32_.unalignImage();

    auto dims = param.y->ZynqTensor()->shape();
    float* hwc_fp32_data = hwc_fp32_.data<float>();
    float* chw_fp32_data = param.y->mutable_data<float>();

    hwc_to_chw(chw_fp32_data,
               hwc_fp32_data,
               dims.num(),
               dims.channel(),
               dims.height(),
               dims.width());

    param.y->ZynqTensor()->copyScaleFrom(&hwc_fp32_);
    param.y->ZynqTensor()->flush();
    copy_properties(param);

    // param.x->ZynqTensor()->invalidate();
    // param.x->ZynqTensor()->flush();
  }
  std::string doc() const override { return "Copy IO from FPGA to HOST"; }

  private:
    zynqmp::Tensor hwc_fp32_;

    CPUPE cpu_pe_;
    BypassPE bypass_pe_;
};

}  // namespace fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyHostCHWToFpgaHWCCompute,
                     host_to_device)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyHostCHWToFpgaHWCCompute,
                     host_float_chw_to_device_fp16_hwc)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

// REGISTER_LITE_KERNEL(io_copy_once,
//                      kFPGA,
//                      kAny,
//                      kAny,
//                      paddle::lite::kernels::fpga::IoCopyHostCHWToFpgaHWCCompute,
//                      host_float_chw_to_device_fp16_hwc)
//     .BindInput("Input",
//                {LiteType::GetTensorTy(TARGET(kHost),
//                                       PRECISION(kFloat),
//                                       DATALAYOUT(kNCHW))})
//     .BindOutput("Out",
//                 {LiteType::GetTensorTy(TARGET(kFPGA),
//                                        PRECISION(kFP16),
//                                        DATALAYOUT(kNHWC))})
//     .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCHWCompute,
                     device_to_host_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kAny),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

// REGISTER_LITE_KERNEL(io_copy_once,
//                      kFPGA,
//                      kAny,
//                      kAny,
//                      paddle::lite::kernels::fpga::IoCopyFpgaToHostCHWCompute,
//                      device_to_host_chw)
//     .BindInput("Input",
//                {LiteType::GetTensorTy(TARGET(kFPGA),
//                                       PRECISION(kAny),
//                                       DATALAYOUT(kNHWC))})
//     .BindOutput("Out",
//                 {LiteType::GetTensorTy(TARGET(kARM),
//                                        PRECISION(kFloat),
//                                        DATALAYOUT(kNCHW))})
//     .Finalize();

REGISTER_LITE_KERNEL(io_copy,
                     kFPGA,
                     kAny,
                     kAny,
                     paddle::lite::kernels::fpga::IoCopyFpgaToHostCHWCompute,
                     device_to_host_hwc_chw)
    .BindInput("Input",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNHWC))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kARM),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();

// REGISTER_LITE_KERNEL(io_copy_once,
//                      kFPGA,
//                      kAny,
//                      kAny,
//                      paddle::lite::kernels::fpga::IoCopyFpgaToHostCHWCompute,
//                      device_to_host_hwc_chw)
//     .BindInput("Input",
//                {LiteType::GetTensorTy(TARGET(kFPGA),
//                                       PRECISION(kFloat),
//                                       DATALAYOUT(kNHWC))})
//     .BindOutput("Out",
//                 {LiteType::GetTensorTy(TARGET(kARM),
//                                        PRECISION(kFloat),
//                                        DATALAYOUT(kNCHW))})
//     .Finalize();
