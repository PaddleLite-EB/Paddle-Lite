/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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


#define FILTER_NUM_ALIGNMENT 32  //Filter number aligned to 32
#define FILTER_ELEMENT_ALIGNMENT 16  //Filter element number aligned to 16

namespace paddle_mobile {
  namespace fpga {
    namespace filter{
      void convert_to_hwc(float** data_in, int num, int channel, int height, int width);
      float find_max(float* data_in, int num);
      void quantize(float* data_in, int num);
      void align_element(float** data_in, int num, int chw);
      void align_num(float** data_in, int num_per_div_before_alignment, int num, int chw);
      void reorder(float** data_in, int num_after_alignment, int chw);
      void interleave(float** data_in, int num_after_alignment, int chw);

    }  // namespace filter
  }  // namespace fpga
}  // namespace paddle_mobile