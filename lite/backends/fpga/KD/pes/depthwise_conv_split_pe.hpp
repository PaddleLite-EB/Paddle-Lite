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

#include <vector>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"

namespace paddle {
namespace zynqmp {

class DepthwiseConvSplitPE : public PE {
 public:
  inline int gcd_(int a, int b) {
    while (b) {
      int temp = a;
      a = b;
      b = temp % b;
    }
    return a;
  }

  inline int lcm_(int a, int b) { return a * b / gcd_(a, b); }

  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    DepthwiseConvSplitParam& param = param_;
    Tensor* input = param.input;
    Tensor* output = param.output;
    int channel = output->shape().channel();

    dwconv_split_channel(param);

    if (param.splitParams().size() > 1) {
      std::cout << "dwconv_split size:" << param.splitParams().size() << std::endl;
      splitPE_.init();
      splitPE_.apply();

      SplitParam& split_param = splitPE_.param();
      split_param.input = param_.input;
      for (auto dwconv_param : param_.splitParams()) {
        dwconv_param->args.output_idx = param_.output->scaleIndex(true);
        dwconv_param->args.inplace.findmax_restart = i == 0;
        dwconv_param->args.inplace.active_param.type = param_.activeParam.type;
        dwconv_param->args.inplace.active_param.leaky_relu_factor =
            float_to_half(param_.activeParam.leaky_relu_factor);
        int action_id = compute_fpga_dwconv(dwconv_param->args);
        Action* action = new Action(action_id);
        actions_.push_back(action);
        transaction_->appendAction(action);

        split_param.outputs.push_back(&dwconv_param->input);
      }
      
      ConcatParam& concat_param = concatPE_.param();
      for (auto dwconv_param : param_.splitParams()) {
        concat_param.inputs.push_back(&dwconv_param->output);
      }
      concat_param.output = param_.output;
      concatPE_.init();
      concatPE_.apply();
    }
  }

  bool dispatch() {
    std::vector<BasicDWConvParam*>& params = param_.splitParams();

    if (params.size() > 1) {
      splitPE_.dispatch();
    }

    if (params.size() > 1) {
      concatPE_.dispatch();
    }

    return ret;
  }

  DepthwiseConvSplitParam& param() { return param_; }

  ~DepthwiseConvSplitPE() {
    for (auto dwconv_param : param_.splitParams()) {
      if (param_.splitParams().size() > 1) {
        delete dwconv_param->input;
        delete dwconv_param->output;
        delete dwconv_param;
      }
    }
    splitParams_.clear();

    for (int i = 0; i < actions_.size(); i++) {
      Action* action = actions_[i];
      delete action;
    }
    actions_.clear();
  }

 private:
  DepthwiseConvSplitParam param_;
  ConcatPE concatPE_;
  SplitPE splitPE_;
  std::shared_ptr<Transaction> transaction_;
  std::vector<Action*> actions_;
};

}  // namespace zynqmp
}  // namespace paddle
