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

#include <algorithm>

#include "lite/backends/fpga/KD/tensor_util.hpp"

namespace paddle {
namespace zynqmp {
float find_max(const Tensor& tensor) {
  float max = 0;
  Tensor& t = const_cast<Tensor&>(tensor);
  float* data = t.data<float>();
  for (int i = 0; i < t.shape().numel(); i++) {
    float value = data[i] > 0 ? data[i] : -data[i];
    max = std::max(value, max);
  }
  return max;
}

void cpu_copy(Tensor* src, Tensor* dst) {
  float max = 0;
  if (src->dataType() == FP32 && dst->dataType() == FP16) {
    float16* dst_data = dst->data<float16>();
    float* src_data = src->data<float>();
    for (int i = 0; i < src->shape().alignedElementCount(); i++) {
      dst_data[i] = float_to_half(src_data[i]);
      if (src_data[i] > max) {
        max = src_data[i];
      }
    }
  }

  if (src->dataType() == FP16 && dst->dataType() == FP32) {
    float* dst_data = dst->data<float>();
    float16* src_data = src->data<float16>();
    for (int i = 0; i < src->shape().alignedElementCount(); i++) {
      dst_data[i] = half_to_float(src_data[i]);

      if (dst_data[i] > max) {
        max = dst_data[i];
      }
    }
  }
  dst->scale()[0] = max / 127;
  dst->scale()[1] = 127 / max;
  dst->flush();
}

}  // namespace zynqmp
}  // namespace paddle
