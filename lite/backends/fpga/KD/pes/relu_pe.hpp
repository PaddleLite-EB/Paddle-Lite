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

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
namespace paddle {
namespace zynqmp {

class ReluPE : public PE {
 public:
  bool init(FPGALock* lock = nullptr) {
    FPGALock fpga_lock(lock);
    fpga_lock.lock();
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply(FPGALock* lock = nullptr) {
    FPGALock fpga_lock(lock);
    fpga_lock.lock();

    inplace_args_.relu_enable = true;
    inplace_args_.power_enable = false;
    inplace_args_.normalize_enable = false;
    
    Shape& input_shape = param_.input->shape();
    
    bypass_args_.input_data_type = DATA_TYPE_FP16;
    bypass_args_.output_data_type = DATA_TYPE_FP16;
    bypass_args_.input_layout_type = LAYOUT_HWC;
    bypass_args_.output_layout_type = LAYOUT_HWC;
    bypass_args_.image.address = param_.input->data<void>();
    bypass_args_.image.scale_address = param_.input->max();
    bypass_args_.image.channels = input_shape.channel();
    bypass_args_.image.height = input_shape.height();
    bypass_args_.image.width = input_shape.width();
    bypass_args_.output.address = param_.output->data<void>();
    bypass_args_.output.scale_address = param_.output->max();
  }

  bool dispatch(FPGALock* lock = nullptr) {
    //cpu compute, todo
    // param_.input->invalidate();
    // int16_t* input_data = param_.input->data<int16_t>();
    // float16* out_data = param_.output->data<float16>();
    // for (int i = 0; i < param_.input->shape().alignedElementCount(); i++) {
    //   int16_t v = param_.input->data<float16>()[i];
    //   if (v > 0) {
    //     out_data[i] = input_data[i];
    //   } else {
    //     out_data[i] = zero;
    //   }
    // }
    // param_.output->copyMaxFrom(param_.input);
    // param_.output->flush();

    //fpga compute through bypass
    FPGALock fpga_lock(lock);
    fpga_lock.lock();
      
    param_.input->syncToDevice();
    inplace_args_.relu_enable = true;
    config_inplace(inplace_args_);
    perform_bypass(bypass_args_);
    inplace_args_.relu_enable = false;
    config_inplace(inplace_args_);

    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
  InplaceArgs inplace_args_ = {0};
  BypassArgs bypass_args_;
  float16 zero = float_to_half(0.0f);
};

}  // namespace zynqmp
}  // namespace paddle
