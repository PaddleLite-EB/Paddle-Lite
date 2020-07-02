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

#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/backends/fpga/KD/tensor_util.hpp"

namespace paddle {
namespace zynqmp {

class OutputPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(false);
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
    args.output_idx =
        param_.output->scaleIndex(true);  // TODO(chonwhite) use default index;
    args.activeParam.type = TYPE_NONE;
    return perform_bypass(args);
  }

  void apply() {
    transaction_ = TransactionManager::get_instance().getTransaction();
    Action* action = new Action(config_bypass());
    action_.reset(action);
    transaction_->appendAction(action);
    TransactionManager::get_instance().endTransaction();
  }

  // void setLast(bool last) {
  //   last_ = last;
  // }

  bool dispatch() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;

    input->saveToFile("fetch_in", true);

    transaction_->startTraction();

    output->saveToFile("fetch_out", true);

    // cpu_copy(input, output);
    output->invalidate();
    if (input->aligned()) {
      // output->unalignImage();
    }

    lite::Debugger::get_instance().commit();

    return true;
  }

  OutputParam& param() { return param_; }

 private:
  // bool last_ = true;
  OutputParam param_;
  std::shared_ptr<Transaction> transaction_;
  std::shared_ptr<Action> action_;
};
}  // namespace zynqmp
}  // namespace paddle
