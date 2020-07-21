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

#include <memory>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {

class ElementwiseAddPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* input0 = param_.inputs[0];
    Tensor* input1 = param_.inputs[1];
    Tensor* output = param_.output;
    EWAddArgs args = {0};
    args.const0 = 0x3c00;
    args.const1 = 0x3c00;  // =1
    args.image0.address = input0->data<float16>();
    args.image0.channels = input0->shape().channel();
    args.image0.scale_address = input0->scale();
    args.image0.height = input0->shape().height();
    args.image0.width = input0->shape().width();
    args.image0.pad_height = 0;
    args.image0.pad_width = 0;
    args.image1.address = input1->data<float16>();
    args.image1.channels = input1->shape().channel();
    args.image1.scale_address = input1->scale();
    args.image1.height = input1->shape().height();
    args.image1.width = input1->shape().width();
    args.image1.pad_height = 0;
    args.image1.pad_width = 0;
    args.output.scale_address = output->scale();
    args.output.address = output->data<float16>();
    args.output_idx = output->scaleIndex(true);

    args.inplace.findmax_restart = true;
    args.inplace.active_param.type = param_.activeParam.type;
    ;
    args.inplace.active_param.leaky_relu_factor =
        float_to_half(param_.activeParam.leaky_relu_factor);

    param_.ewargs = args;  // delete ewargs;

    transaction_ = TransactionManager::get_instance().getTransaction();
    Action* action = new Action(compute_fpga_ewadd(args));
    action_.reset(action);
    transaction_->appendAction(action);
  }

  bool dispatch() { return true; }

  ElementwiseAddParam& param() { return param_; }

 private:
  ElementwiseAddParam param_;
  InplaceArgs inplace_ = {0};

  std::shared_ptr<Transaction> transaction_;
  std::shared_ptr<Action> action_;
};

}  // namespace zynqmp
}  // namespace paddle
