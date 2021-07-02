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
#include "lite/backends/fpga/KD/pes/bypass_pe.hpp"
#include "lite/backends/fpga/KD/pes/cpu_pe.hpp"

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
    args.image.scale_address = param_.input->max();
    args.image.channels = param_.input->shape().alignedElementCount();
    args.image.height = 1;
    args.image.width = 1;
    args.image.pad_height = 0;
    args.image.pad_width = 0;
    args.output.address = param_.output->data<void>();
    args.output.scale_address = param_.output->max();
    args.output_idx = param_.output->maxIndex(true);
    args.inplace.findmax_restart = true;
    args.inplace.active_param.type = TYPE_NONE;

    return perform_bypass(args);
  }


  void apply() {
     if (cpu_compute_) {
      input_float_.mutableData<float>(FP32, param_.input->shape());
      float_out_.mutableData<float>(FP32, param_.output->shape());

      BypassParam& in_param = bypass_in_pe_.param();
      in_param.input = param_.input;
      in_param.output = &input_float_;
      bypass_in_pe_.init();
      bypass_in_pe_.apply();

      cpu_pe_.init();
      cpu_pe_.apply();

      BypassParam& out_param = bypass_out_pe_.param();
      out_param.input = &float_out_;
      out_param.output = param_.output;
      bypass_out_pe_.init();
      bypass_out_pe_.apply();
      // cpu_pe2_.init();
      // cpu_pe2_.apply();

    } else {

      Shape& input_shape = param_.input->shape();
      float16* mid_data =
          mid_out_.mutableData<float16>(FP16, param_.output->shape());

      bypass_args_.input_data_type = DATA_TYPE_FP16;
      bypass_args_.output_data_type = DATA_TYPE_FP16;
      bypass_args_.input_layout_type = LAYOUT_HWC;
      bypass_args_.output_layout_type = LAYOUT_HWC;
      bypass_args_.image.address = param_.input->data<void>();
      bypass_args_.image.scale_address = param_.input->max();
      bypass_args_.image.channels = input_shape.channel();
      bypass_args_.image.height = input_shape.height();
      bypass_args_.image.width = input_shape.width();
      bypass_args_.output.address = mid_out_.data<void>();
      bypass_args_.output.scale_address = mid_out_.max();

      bypass_args_.inplace.normalize_param.channel = input_shape.channel();
      bypass_args_.inplace.normalize_param.hight_width =
          input_shape.height() * input_shape.width();
      bypass_args_.inplace.normalize_param.enabled = true;

      Action* bypass_action = new Action(perform_bypass(bypass_args_));
      bypass_action_.reset(bypass_action);
      transaction_->appendAction(bypass_action);

      norm_args_.input_image_address = mid_data;
      norm_args_.image_width = input_shape.width();
      norm_args_.image_height = input_shape.height();
      norm_args_.image_channel = input_shape.channel();
      norm_args_.output_image_address = param_.output->data<float16>();
      norm_args_.output_scale_address =
          reinterpret_cast<uint32_t*>(param_.output->max());
      norm_args_.inplace = bypass_args_.inplace;
      norm_args_.inplace.normalize_param.enabled = false;

      Action* norm_action = new Action(compute_norm(norm_args_));
      norm_action_.reset(norm_action);
      transaction_->appendAction(norm_action);
    }

      // if (DLEngine::get_instance().isZU3()) {
      //   cpu_compute_ = true;
      // }
      // cpu_compute_ = true;
  }

  void cpu_compute() {
    input_float_.syncToCPU();

    int channel = input_float_.shape().channel();
    int height = input_float_.shape().height();
    int width = input_float_.shape().width();
    int cw = channel * width;

    Tensor* input = &input_float_;
    float* input_ptr = input->data<float>();
    float* out_ptr = float_out_.data<float>();

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
    float_out_.flush();
  }

  bool dispatch() {
    if (cpu_compute_) {
      bypass_in_pe_.dispatch();
      cpu_pe_.dispatch();
      // std::cout << "norm_pe bypass1 max " << std::endl;
      param_.input->readMax();
      input_float_.readMax();
      cpu_compute();
      bypass_out_pe_.dispatch();

      // cpu_pe2_.dispatch();
      // std::cout << "norm_pe bypass2 max " << std::endl;
      // param_.output->readMax();
    }

    return true;
  }

  NormParam& param() { return param_; }

 private:
  NormParam param_;
  Tensor mid_out_;
  BypassArgs bypass_args_;
  NormalizeArgs norm_args_ = {0};
  bool cpu_compute_ = true;

  Tensor input_float_;
  Tensor float_out_;

  CPUPE cpu_pe_;
  // CPUPE cpu_pe2_;
  BypassPE bypass_in_pe_;
  BypassPE bypass_out_pe_;

  std::shared_ptr<Transaction> transaction_;
  std::unique_ptr<Action> bypass_action_;
  std::unique_ptr<Action> norm_action_;
};

}  // namespace zynqmp
}  // namespace paddle
