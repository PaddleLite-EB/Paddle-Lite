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
#include "lite/backends/fpga/KD/pes/bypass_pe.hpp"
#include "lite/backends/fpga/KD/tensor_util.hpp"

namespace paddle {
namespace zynqmp {

class InputPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  int config_bypass() {
    DDataType in_type =
        param_.input->dataType() == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    DDataType out_type =
        param_.output->dataType() == FP32 ? DATA_TYPE_FP32 : DATA_TYPE_FP16;
    switch (param_.output->dataType()) {
      case INT32:
        in_type = DATA_TYPE_FP32;
        out_type = DATA_TYPE_FP32;  // hack;
        break;
    }

    BypassArgs args;
    args.input_data_type = in_type;
    args.output_data_type = out_type;
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
    args.inplace.normalize_param.enabled = false;

    return perform_bypass(args);
  }

  void apply() {
    transaction_ = TransactionManager::get_instance().getTransaction();

    Action* action = new Action(config_bypass());
    action_.reset(action);
    transaction_->appendAction(action);

    // TransactionManager::get_instance().endTransaction();

    // BypassParam& bypass_param = bypass_pe_.param();
    // bypass_param.input = param_.input;
    // bypass_param.output = param_.output;

    // bypass_pe_.init();
    // bypass_pe_.apply();
  }

  bool dispatch() {
    // we need to align image first;
    Tensor* input = param_.input;
    input->alignImage();
    input->flush();
    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
  // BypassPE bypass_pe_;

  std::shared_ptr<Transaction> transaction_;
  std::unique_ptr<Action> action_;
};
}  // namespace zynqmp
}  // namespace paddle
