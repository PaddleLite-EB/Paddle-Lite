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

#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/bypass_pe.hpp"
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

    args.inplace.findmax_restart = true;
    args.inplace.active_param.type = TYPE_NONE;
    return perform_bypass(args);
  }

  void apply() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;

    if (param_.input->dataType() == FP16) {
      BypassParam& bypass_param = bypass_out_pe_.param();
      if (input->aligned()) {
        temp_tensor.setAligned(true);
        temp_tensor.mutableData<void>(input->dataType(), input->shape());

        bypass_param.input = &temp_tensor;
        bypass_param.output = output;
        bypass_out_pe_.init();
        bypass_out_pe_.apply();
      } else {
        bypass_param.input = input;
        bypass_param.output = output;
      }
    }

    transaction_ = TransactionManager::get_instance().getTransaction();
    if (param_.input->dataType() == FP16) {
      Action* action = new Action(config_bypass());
      action_.reset(action);
      transaction_->appendAction(action);
    }
    TransactionManager::get_instance().endTransaction();
  }

  bool dispatch() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    transaction_->startTraction();
    if (input->dataType() == FP32) {
      memcpy(output->mutableData<void>(),
             input->data<void>(),
             input->memorySize());
      output->flush();
    } else {
      if (input->aligned()) {
        temp_tensor.copyFrom(input);
        temp_tensor.unalignImage();
      }
      bypass_out_pe_.dispatch();
    }

    // lite::Debugger::get_instance().commit();

    return true;
  }

  OutputParam& param() { return param_; }

 private:
  Tensor temp_tensor;
  OutputParam param_;
  std::shared_ptr<Transaction> transaction_;
  std::shared_ptr<Action> action_;

  BypassPE bypass_out_pe_;
};
}  // namespace zynqmp
}  // namespace paddle
