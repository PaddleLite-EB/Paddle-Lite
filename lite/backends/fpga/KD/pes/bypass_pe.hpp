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
#include <utility>
#include <vector>
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/tensor_util.hpp"

namespace paddle {
namespace zynqmp {

class BypassPE : public PE {
 public:
  bool init() {
    Tensor *output = param_.output;

    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  std::vector<int> config_bypass() {
    std::vector<int> action_ids;
    int ret = -1;
    int size = param_.input->shape().alignedElementCount();
    const int max_size = 1 << 21;

    float times = 1.0 * size / max_size;
    int count = static_cast<int>(times);

    void *input_address = param_.input->data<void>();
    int type_size =
        param_.input->dataType() == FP32 ? sizeof(float) : sizeof(int16_t);

    void *output_address = param_.output->data<void>();
    int out_type_size =
        param_.output->dataType() == FP32 ? sizeof(float) : sizeof(int16_t);

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

    float scale = 0;
    for (int i = 0; i < count; ++i) {
      args.image.channels = max_size;
      args.image.address =
          reinterpret_cast<char *>(input_address + i * max_size * type_size);
      args.output.address = reinterpret_cast<char *>(
          output_address + i * max_size * out_type_size);
      args.inplace.findmax_restart = i == 0;
      ret = perform_bypass(args);
      action_ids.push_back(ret);
    }

    // TODO(chonwhite) size should include type_size;
    int remainder = size - max_size * count;
    if (remainder > 0) {
      args.image.channels = remainder;
      args.image.address = reinterpret_cast<char *>(
          input_address + count * max_size * type_size);
      args.output.address = reinterpret_cast<char *>(
          output_address + count * max_size * out_type_size);
      args.inplace.findmax_restart = false;
      ret = perform_bypass(args);
      action_ids.push_back(ret);
    }
    return action_ids;
  }

  void apply() {
    transaction_ = TransactionManager::get_instance().getTransaction();

    auto action_ids = config_bypass();

    for (auto action_id : action_ids) {
      Action *action = new Action(action_id);
      transaction_->appendAction(action);

      std::unique_ptr<Action> action_pointer(action);
      actions_.push_back(std::move(action_pointer));
    }
  }

  bool dispatch() { return true; }

  BypassParam &param() { return param_; }

 private:
  BypassParam param_;
  std::shared_ptr<Transaction> transaction_;
  std::vector<std::unique_ptr<Action>> actions_;
};
}  // namespace zynqmp
}  // namespace paddle
