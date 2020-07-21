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

#pragma once

#include <cstring>
#include <vector>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {
class NormPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  int config_bypass() {
    BypassArgs args;
    args.input_data_type =
        param_.input->dataType() == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    args.output_data_type =
        param_.output->dataType() == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    args.input_layout_type = LAYOUT_HWC;
    args.output_layout_type = LAYOUT_HWC;
    args.image.address = param_.input->data<void>();
    args.image.scale_address = param_.input->scale();
    args.image.channels = param_.input->shape().alignedElementCount();
    args.image.height = 1;
    args.image.width = 1;
    args.image.pad_height = 0;
    args.image.pad_width = 0;
    args.output.address = param_.output->data<void>();
    args.output.scale_address = param_.output->scale();
    args.output_idx = param_.output->scaleIndex(true);
    args.inplace.findmax_restart = true;
    args.inplace.active_param.type = TYPE_NONE;

    return perform_bypass(args);
  }

  void apply() {
    transaction_ = TransactionManager::get_instance().getTransaction();

    Tensor* input = param_.input;
    Tensor* output = param_.output;

    Shape& input_shape = input->shape();
    float16* tmp_out_data =
        tmp_out_.mutableData<float16>(FP16, output->shape());

    ImageInputArgs imageInputArgs = {
        .address = input->data<float16>(),
        .scale_address = input->scale(),
        .channels = static_cast<uint32_t>(input->shape().channel()),
        .width = static_cast<uint32_t>(input->shape().width()),
        .height = static_cast<uint32_t>(input->shape().height()),
        .pad_width = 0,
        .pad_height = 0,
    };

    ImageOutputArgs BypassOutput = {
        .address = tmp_out_.data<float16>(), .scale_address = tmp_out_.scale(),
    };

    BypassArgs bypassArgs = {
        .input_data_type = DATA_TYPE_FP16,
        .output_data_type = DATA_TYPE_FP16,
        .input_layout_type = LAYOUT_HWC,
        .output_layout_type = LAYOUT_HWC,
        .image = imageInputArgs,
        .output = BypassOutput,
    };

    bypassArgs.inplace.active_param.type = param_.activeParam.type;
    bypassArgs.inplace.normalize_param.channel = input->shape().channel();
    int height = input->shape().height();
    int width = input->shape().width();

    bypassArgs.inplace.normalize_param.hight_width = height * width;
    bypassArgs.inplace.normalize_param.enabled = 1;

    Action* bypass_action = new Action(perform_bypass(bypassArgs));
    bypass_action_.reset(bypass_action);
    transaction_->appendAction(bypass_action);

    NormalizeArgs args = {0};
    args.input_image_address = tmp_out_.data<float16>();
    args.image_channel = input->shape().channel();
    args.image_height = input->shape().height();
    args.image_width = input->shape().width();
    args.output_image_address = output->data<float16>();
    args.inplace.active_param.type = param_.activeParam.type;
    args.inplace.normalize_param.channel = 0;
    args.inplace.normalize_param.hight_width = 0;
    args.inplace.normalize_param.enabled = 0;

    // uint32_t action_id2 = compute_norm(args);
    Action* norm_action = new Action(compute_norm(args));
    norm_action_.reset(norm_action);
    transaction_->appendAction(norm_action);
  }

  void cpuCompute() {
    Tensor input_float;
    Tensor float_out;
    input_float.mutableData<float>(FP32, param_.input->shape());
    float_out.mutableData<float>(FP32, param_.output->shape());

    // param_.input->syncToDevice();
    input_float.copyFrom(param_.input);
    input_float.syncToCPU();
    // input_float.saveToFile("normalize_", true);

    int channel = input_float.shape().channel();
    int height = input_float.shape().height();
    int width = input_float.shape().width();
    int cw = channel * width;

    Tensor* input = &input_float;
    float* input_ptr = input->data<float>();
    float* out_ptr = float_out.data<float>();

    int loop = height * width;
#pragma omp parallel for
    for (int i = 0; i < loop; i++) {
      float sum = param_.epsilon;
      for (int c = 0; c < channel; c++) {
        float value = input_ptr[i * channel + c];
        sum += value * value;
      }
      float norm = sqrtf(sum);
#pragma omp parallel for
      for (int c = 0; c < channel; c++) {
        out_ptr[i * channel + c] = input_ptr[i * channel + c] / norm;
      }
    }
    float_out.flush();
    // float_out.saveToFile("normalize_", true);
    param_.output->copyFrom(&float_out);
    param_.output->flush();
  }

  bool dispatch() {
    std::cout << "Norm\n";
    // cpuCompute();
    // std::cout << "CPU normalize ---------------------" << std::endl;

    // param_.input->syncToDevice();
    // // param_.input->saveToFile("normalize_fpga_", true);
    // config_norm_param(norm_param_args_);
    // inplace_args_.normalize_enable = true;
    // config_inplace(inplace_args_);

    // perform_bypass(bypass_args_);
    // inplace_args_.normalize_enable = false;
    // config_inplace(inplace_args_);
    // compute_norm(norm_args_);
    // param_.output->saveToFile("normalize_fpga_", true);
    // std::cout << "FPGA normalize ---------------------" << std::endl;
    return true;
  }

  NormParam& param() { return param_; }

 private:
  NormParam param_;
  Tensor tmp_out_;

  std::shared_ptr<Transaction> transaction_;
  std::shared_ptr<Action> bypass_action_;
  std::shared_ptr<Action> norm_action_;
};
}  // namespace zynqmp
}  // namespace paddle
