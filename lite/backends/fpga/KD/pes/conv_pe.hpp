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

#include <arm_neon.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"
#include "lite/backends/fpga/KD/pes/cpu_pe.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"

#include "lite/backends/fpga/KD/dispatch/transaction_manager.hpp"

namespace paddle {
namespace zynqmp {

class ConvPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    split_axis = fill_split_arg(param_);
    split_channel = param_.groups != 1 && param_.splitParams().size() > 1;

    // ======================= dispatch =======================
    transaction_ = TransactionManager::get_instance().getTransaction();
    if (split_axis == 0) {
      for (int i = 0; i < param_.splitParams().size(); i++) {
        auto conv_param = param_.splitParams()[i];
        if (i == 0) {
          conv_param->args.inplace.findmax_restart = true;
          conv_param->args.output_idx = param_.output->scaleIndex(true);
        } else {
          conv_param->args.inplace.findmax_restart = false;
          conv_param->args.output_idx = 0;
        }
        conv_param->args.inplace.active_param.type = param_.activeParam.type;
        conv_param->args.inplace.active_param.leaky_relu_factor =
            float_to_half(param_.activeParam.leaky_relu_factor);
        int action_id = compute_fpga_conv_basic(conv_param->args);
        Action* action = new Action(action_id);
        actions_.push_back(action);
        transaction_->appendAction(action);
      }
    }

    // ======================= concat =======================
    if (split_axis == 0 && param_.splitParams().size() > 1) {
      ConcatParam& concat_param = concatPE_.param();
      for (auto conv_param : param_.splitParams()) {
        concat_param.inputs.push_back(&conv_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
      concatPE_.setMergeScale(false);
    }

    if (split_channel) {
      SplitParam& split_param = splitPE_.param();
      split_param.input = param_.input;
      for (auto conv_param : param_.splitParams()) {
        split_param.outputs.push_back(&conv_param->input);
      }
      splitPE_.init();
      splitPE_.apply();
    }

    if (DLEngine::get_instance().isZU3() &&
        param_.input->shape().dimSize() == 4 &&
        param_.input->shape().width() == 1 &&
        param_.input->shape().channel() >= 2048) {
      // use_cpu_ = true;
      cpu_pe_.reset(new CPUPE());
    }

    if (!use_cpu_) {
      // param_.filter->releaseData();
    }
  }

  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    // float16* data_out = output->data<float16>();
    float* out = float_output.mutableData<float>(FP32, output->shape());

    int out_channel = output->shape().channel();
    int in_channel = input->shape().channel();

    float* filter_data = param_.filter->data<float>();
    float* mi = new float[in_channel];
    for (int i = 0; i < out_channel; i++) {
      float* image = image_addr;
      float* filter_ptr = filter_data + i * in_channel;
      float* out_ptr = mi;
#pragma omp parallel for
      for (int j = 0; j < in_channel; j++) {
        float value = image_addr[j] * filter_ptr[j];
        mi[j] = value;
      }

      float sum = 0;
      for (int j = 0; j < in_channel; j++) {
        sum += mi[j];
      }
      out[i] = sum;
    }
    delete[] mi;
    float_output.flush();
    output->copyFrom(&float_output);
  }

  bool dispatch() {
    if (use_cpu_) {
      cpu_pe_->dispatch();
      cpu_compute();
      return true;
    }

    std::vector<BasicConvParam*>& params = param_.splitParams();
    // if (split_channel) {
    //   splitPE_.dispatch();
    // }

    size_t size = params.size();
    if (split_axis == 0 && size > 1) {
      // param_.output->readScale();
      float scale = param_.output->scale()[0];
      concatPE_.dispatch();
      // param_.output->writeScale(scale);
    }

    // if (split_axis == 1 && ret == 0 && size > 1) {
    //   // for (int n = 0; n < size - 1; n++) {
    //   ElementwiseAddParam& add_param = addPE_.param();
    //   add_param.inputs = {&params[0]->output, &params[1]->output};
    //   add_param.output = param_.output;
    //   addPE_.init();
    //   addPE_.apply();
    //   addPE_.dispatch();
    // }
    // return ret == 0;

    return true;
  }

  ConvParam& param() { return param_; }

  ~ConvPE() {
    for (int i = 0; i < actions_.size(); i++) {
      Action* action = actions_[i];
      delete action;
    }
    actions_.clear();
  }

 private:
  bool use_cpu_ = false;
  bool split_channel = false;
  ConvParam param_;
  ConcatPE concatPE_;
  SplitPE splitPE_;
  ElementwiseAddPE addPE_;
  int split_axis = 0;
  InplaceArgs inplace_ = {0};
  ActiveParamterArgs activeParamterArgs;

  // =================
  std::shared_ptr<Transaction> transaction_;
  std::vector<Action*> actions_;
  std::shared_ptr<CPUPE> cpu_pe_;
};

}  // namespace zynqmp
}  // namespace paddle
