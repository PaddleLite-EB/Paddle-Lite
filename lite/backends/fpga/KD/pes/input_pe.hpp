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

  // bool dispatch() {
  //   Tensor* input = param_.input;
  //   Tensor* output = param_.output;

  //   Tensor* src = input;
  //   input->flush();
  //   Tensor half_tensor;
  //   DataType dataType = input->dataType();
  //   switch (dataType) {
  //     case FP32:
  //       half_tensor.mutableData<void*>(DataType::FP16, input->shape());
  //       half_tensor.copyFrom(input, &fpga_lock);
  //       src = &half_tensor;
  //       output->mutableData<void>();
  //       src->alignImage(&fpga_lock);
  //       output->copyFrom(src, &fpga_lock);
  //       break;
  //     case FP16:
  //       input->setAligned(true);
  //       bypassPE_.param().input = input;
  //       bypassPE_.param().output = output;
  //       bypassPE_.init(&fpga_lock);
  //       bypassPE_.apply(&fpga_lock);
  //       bypassPE_.dispatch(&fpga_lock);
  //       break;
  //     default:
  //       output->mutableData<void>();
  //       src->alignImage(&fpga_lock);
  //       output->copyFrom(src, &fpga_lock);
  //       break;
  //   }
  // }

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

    tmp_tensor_.mutableData<void>(param_.input->dataType(),
                                  param_.input->shape());

    BypassArgs args;
    args.input_data_type = in_type;
    args.output_data_type = out_type;
    args.input_layout_type = LAYOUT_HWC;
    args.output_layout_type = LAYOUT_HWC;
    args.image.address = tmp_tensor_.data<void>();
    args.image.scale_address = tmp_tensor_.scale();
    args.image.channels = tmp_tensor_.shape().alignedElementCount();
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
    Tensor* input = param_.input;
    auto in_type = input->dataType();
    if (in_type != zynqmp::INT32) {  
      transaction_ = TransactionManager::get_instance().getTransaction();

      Action* action = new Action(config_bypass());
      action_.reset(action);
      transaction_->appendAction(action);
    }

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
    auto in_type = input->dataType();
    int count = input->shape().numel();
    if (in_type == zynqmp::INT32) {  
      int32_t* input_data = input->mutableData<int32_t>();
      int32_t* out_data = param_.output->mutableData<int32_t>(DataType::INT32, input->shape());
      for (int i = 0; i < count; ++i) {
        out_data[i] =  input_data[i];
      }
      param_.output->flush();
    } else {
      memcpy(tmp_tensor_.data<void>(),
             input->data<void>(),
             tmp_tensor_.memorySize());
      tmp_tensor_.alignImage();
      tmp_tensor_.flush();
    }
    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
  Tensor tmp_tensor_;
  // BypassPE bypass_pe_;

  std::shared_ptr<Transaction> transaction_;
  std::unique_ptr<Action> action_;
};

}  // namespace zynqmp
}  // namespace paddle
