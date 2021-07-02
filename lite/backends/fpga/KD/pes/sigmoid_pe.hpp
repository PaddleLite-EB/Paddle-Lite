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

#include <algorithm>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/cpu_pe.hpp"

namespace paddle {
namespace zynqmp {

using float16 = zynqmp::float16;

class SigmoidPE : public PE {
 public:
  bool init() {
    param_.output->setAligned(false);
    param_.output->setDataLocation(CPU);
    return true;
  }

  void apply() {
    cpu_pe_.init();
    cpu_pe_.apply();
  }

  bool dispatch() {
    cpu_pe_.dispatch();

    Tensor* input = param_.input;
    Tensor* output = param_.output;
    float16* in_data = input->data<float16>();
    auto out_data = param_.output->mutableData<float16>();
    int numel = param_.output->shape().numel();

    input->invalidate();
    float max = 0.0f;
    for (int i = 0; i < numel; i++) {
      float value = zynqmp::half_to_float(in_data[i]);
      value = 1 / (1 + exp(-value));
      out_data[i] = zynqmp::float_to_half(value);
      max = std::max(std::abs(value), max);
    }
    output->max()[0] = zynqmp::float_to_half(max);
    output->maxIndex(true);
    output->writeMax(output->max()[0]);
    // output->scale()[0] = max / 127.0;
    // output->scale()[1] = 127.0 / max;
    output->flush();
  }
  
  SigmoidParam& param() { return param_; }

 private:
  SigmoidParam param_;
  CPUPE cpu_pe_;
};
}  // namespace zynqmp
}  // namespace paddle
