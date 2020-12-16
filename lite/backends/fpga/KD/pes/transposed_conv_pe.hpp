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

// #include <arm_neon.h>
#include <algorithm>
#include <vector>

#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"
#include "lite/backends/fpga/KD/pes/transposed_conv_process.hpp"

namespace paddle {
namespace zynqmp {

class TransposedConvPE : public PE {
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
    int kernel_width = param_.filter->shape().width();
    int kernel_height = param_.filter->shape().height();
    int stride_width = param_.strides[0];
    int padding_width = param_.paddings[0];

    if (kernel_width % stride_width == 0) {
      sub_filter_ena_ = true;
    } else {
      sub_filter_ena_ = false;
    }
    // 使用pad input方案
    sub_filter_ena_ = false;

    ConvParam& conv_param = pe_.param();

    convert_cnhw_to_nchw(param_.filter, &filter_);
    inverse_filter(&filter_);

    if (sub_filter_ena_) {
      omit_size_ = deconv_get_omit(stride_width, kernel_width, padding_width);

      fill_sub_filters(&param_, &filter_, &tmp_output_);

      conv_param = const_cast<ConvParam&>(param_);
      conv_param.deconv = true;
      conv_param.activeParam.type = param_.activeParam.type;

    } else {
      // fill_transposed_split_arg(param_);
      Shape& input_shape = param_.input->shape();
      int padded_height = input_shape.height() +
                          (input_shape.height() - 1) * (param_.strides[0] - 1);
      int padded_width = input_shape.width() +
                         (input_shape.width() - 1) * (param_.strides[1] - 1);

      Shape padded_shape(NCHW,
                         {input_shape.num(),
                          input_shape.channel(),
                          padded_height,
                          padded_width});

      // int p = param_.kernelSize[0] - param_.paddings[0] - 1;
      // int p = kernel_width - param_.paddings[0] - 1;
      int ph = param_.filter->shape().height() - param_.paddings[0] - 1;
      int pw = param_.filter->shape().width() - param_.paddings[1] - 1;

      padded_input_.mutableData<float16>(FP16, padded_shape);
      conv_param.input = &padded_input_;
      conv_param.output = param_.output;
      conv_param.filter = &filter_;
      conv_param.strides = {1, 1};
      conv_param.paddings = {ph, pw};
      conv_param.kernelSize = {kernel_height, kernel_width};
      // conv_param.kernelSize = param_.kernelSize;
      conv_param.dilations = {1, 1};
      conv_param.deconv = false;
      conv_param.activeParam.type = param_.activeParam.type;
      conv_param.scale()->mutableData<float>(FP32, param_.scale()->shape());
      conv_param.scale()->copyFrom(param_.scale());
      conv_param.bias()->mutableData<float>(FP32, param_.bias()->shape());
      conv_param.bias()->copyFrom(param_.bias());
    }
    pe_.init(&fpga_lock);
    pe_.apply(&fpga_lock);
  }

  template <typename T>
  void pad_input() {
    param_.input->syncToCPU();
    T* input_data = param_.input->data<T>();
    // param_.input->saveToFile("pad_input", true);
    int channel = param_.input->shape().channel();
    int in_wc = param_.input->shape().width() * channel;

    int o_wc = padded_input_.shape().width() * channel;

    T* data = padded_input_.data<T>();
    int oh = param_.input->shape().height();
    int ow = param_.input->shape().width();

    memset(data, 0, padded_input_.memorySize());

    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        T* src = input_data + h * in_wc + w * channel;
        T* dst = data + (h)*param_.strides[0] * o_wc +
                 (w) * (param_.strides[1]) * channel;
        memcpy(dst, src, channel * sizeof(T));
      }
    }

    padded_input_.flush();
    // padded_input_.saveToFile("padded_input", true);
    padded_input_.copyMaxFrom(param_.input);
  }

  bool dispatch(FPGALock* lock = nullptr) {
    FPGALock fpga_lock(lock);
    fpga_lock.lock();
    // int ih = param_.input->shape().height();
    // int iw = param_.input->shape().width();
    // if (ih == 8 && iw == 8) {
    //   param_.input->readFromFile("29_ew_add_relu_1_512_8_8");
    //   std::cout << "29_ew_add_relu_1_512_8_8" << std::endl;
    // }
    if (sub_filter_ena_ == false) {
      pad_input<float16>();
    }

    bool vi = pe_.dispatch(&fpga_lock);

    if (sub_filter_ena_ == true && vi == true) {
      int off_addr = omit_size_ * param_.output->shape().width() * param_.output->shape().channel();

      param_.output->unalignImage();
      param_.output->setOffset(off_addr);

      float16 max_val = 0.0;

      for (auto conv_param : param_.splitParams()) {
        max_val = std::max(max_val, conv_param->output_max);
      }
      param_.output->max()[0] = max_val;
    }

    return vi;
  }

  ConvParam& param() { return param_; }

 private:
  ConvParam param_;
  ConvPE pe_;
  bool sub_filter_ena_;
  int omit_size_;
  Tensor padded_input_;
  Tensor filter_;
  Tensor tmp_output_;
  InplaceArgs inplace_ = {0};
  ActiveParamterArgs activeParamterArgs;
};

}  // namespace zynqmp
}  // namespace paddle
