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

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"

namespace paddle {
namespace zynqmp {

class DepthwiseConvPE : public PE {
 public:
  inline int gcd_(int a, int b) {
    while (b) {
      int temp = a;
      a = b;
      b = temp % b;
    }
    return a;
  }

  inline int lcm_(int a, int b) { return a * b / gcd_(a, b); }

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
    
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    int channel = output->shape().channel();

    int image_dynamic_range = (1 << 11) - 1; //int12 max value, pow(2,11)-1
    float16 dynamic_range_fp16 = float_to_half(image_dynamic_range * 1.0);
    float inv_dynamic_range = 1.0 / image_dynamic_range;

    int alignment = 16;

    if (channel % alignment != 0 || channel < alignment) {
        int c_lcm = lcm_(channel, alignment);
        align_repeat_ = c_lcm / (channel);
    }
    Shape shape(N, {2 * channel * align_repeat_});
    
    float16* b_data = scale_bias_.mutableData<float16>(FP16, shape);
    if (param_.bias()->dataType() == FP32) {
        float* new_bias_data = param_.bias()->data<float>();
        for (int i = 0; i < align_repeat_; i++) {
            for (int j = 0; j < channel; j++) {
                float16 value = float_to_half(new_bias_data[j]);
                b_data[i * channel + j] = value;
            }
        }
    } else {
        float16* new_bias_data = param_.bias()->data<float16>();
        for (int i = 0; i < align_repeat_; i++) {
            for (int j = 0; j < channel; j++) {
                b_data[i * channel + j] = new_bias_data[j];
            }
        }
    }

    if (param_.scale() == nullptr) {
        float16 one = float_to_half(1.0f);
        for (int i = 0; i < align_repeat_; i++) {
            for (int j = 0; j < channel; j++) {
                b_data[channel * align_repeat_ + i * channel + j] = one;
            }
        }     
    } else {
        if (param_.scale()->dataType() == FP32) {
            float* new_scale_data = param_.scale()->data<float>();
            for (int i = 0; i < align_repeat_; i++) {
                for (int j = 0; j < channel; j++) {
                    float16 value = float_to_half(new_scale_data[j]);
                    b_data[channel * align_repeat_ + i * channel + j] = value;
                }
            }     
        } else {
            float16* new_scale_data = param_.scale()->data<float16>();
            for (int i = 0; i < align_repeat_; i++) {
                for (int j = 0; j < channel; j++) {
                    b_data[channel * align_repeat_ + i * channel + j] = new_scale_data[j];
                }
            } 
        } 
    }

    scale_bias_.flush();

    int filter_dynamic_range = 0; 
    Tensor* null_scale = nullptr;

    // if (param_.scale()->dataType() == FP32) {
    if (param_.filter->shape().width() == 1 && param_.filter->shape().height() == 1) {
        filter_dynamic_range = (1 << 15) - 1; //int16 max value, pow(2,15)-1
    } else {
        int fix16_range = (1 << 15) - 1;
        int ext_range = (1 << 19) - 1;
        int max_area = int(ext_range / (param_.filter->shape().height() * param_.filter->shape().width()));
        filter_dynamic_range = std::min(max_area, fix16_range); //int12 max value, pow(2,11)-1
    }
    
    format_dw_filter(param_.filter, param_.quantizedFilter(), null_scale, filter_dynamic_range);

    // } else {
    // //TODO filter 全为1时，且channal为对齐时
    //     filter_dynamic_range = (1 << 15) - 1; //int16 max value, pow(2,15)-1

    //     float16* scale_data = param_.scale()->data<float16>();

    //     Tensor new_scale;
    //     Shape& scale_shape = param_.scale()->shape();
    //     float* new_scale_data = new_scale.mutableData<float>(FP32, scale_shape);  

    //     for (int i = 0; i < length; i++) {
    //         new_scale_data[i] = half_to_float(scale_data[i]);
    //     }
    //     format_dw_filter(param_.filter, param_.quantizedFilter(), &new_scale, with_scale, filter_dynamic_range);
  
    // }

    DWconvArgs args = {0};
    args.bias_address = b_data;
    args.filter_address = param_.quantizedFilter()->data<int16_t>();
    args.filter_scale_address = param_.quantizedFilter()->scale();
    args.kernel.width = param_.filter->shape().height();
    args.kernel.height = param_.filter->shape().width();
    args.kernel.stride_w = param_.strides[0];
    args.kernel.stride_h = param_.strides[1];
    args.image.address = input->data<void>();
    args.image.channels = input->shape().channel();
    args.image.height = input->shape().height();
    args.image.width = input->shape().width();
    args.image.pad_width = param_.paddings[0];
    args.image.pad_height = param_.paddings[1];
    args.image.scale_address = input->max();
    args.output.address = output->data<void>();
    args.output.scale_address = output->max();
    args.out_width = param_.output->shape().width();
    args.out_height = param_.output->shape().height();
    args.quant.dynamic_range = *(uint16_t *)&dynamic_range_fp16;
    args.quant.inv_dynamic_range = *(uint32_t *)&inv_dynamic_range;
    args.sub_conv_num = 1;
    args.dilation = (param_.dilations[0] <= 1) ? 1 : param_.dilations[0];
    param_.args = args;

    // inplace_.relu_enable = param_.relu.enabled;
    inplace_.power_enable = false;
    inplace_.normalize_enable = false;
  }

  // bool dispatch() {
  //   FPGALock* lock = nullptr;
  //   dispatch(lock);
  // }

  bool dispatch(FPGALock* lock = nullptr) {
    FPGALock fpga_lock(lock);
    fpga_lock.lock();
    param_.input->syncToDevice();
    if (param_.activeParam.type == TYPE_RELU) {
      inplace_.relu_enable = true;
    } else if (param_.activeParam.type == TYPE_RELU6) {
      inplace_.relu6_enable = true;
    } else if (param_.activeParam.type == TYPE_SIGMOID) {
      inplace_.sigmoid_enable = true;
    } else if (param_.activeParam.type == TYPE_LEAKY_RELU) {
      inplace_.leaky_relu_enable = true;
    }

    if (inplace_.relu_enable || inplace_.leaky_relu_enable ||
        inplace_.relu6_enable || inplace_.sigmoid_enable) {
      config_inplace(inplace_);
    }
    
    if (param_.re_assign == true) {
        float16* scale_data = scale_bias_.data<float16>();
        int channel = param_.output->shape().channel();
        for (int i = 0; i < align_repeat_; i++) {
            int offset = channel * align_repeat_ + i * channel;
            memcpy(scale_data + offset, param_.scale()->data<float16>(), param_.scale()->memorySize());
        } 
        scale_bias_.flush();
    }

    bool ret = compute_fpga_dwconv(param_.args) == 0;
    if (inplace_.relu_enable || inplace_.leaky_relu_enable ||
        inplace_.relu6_enable || inplace_.sigmoid_enable) {
      inplace_.relu_enable = false;
      inplace_.leaky_relu_enable = false;
      inplace_.relu6_enable = false;
      inplace_.sigmoid_enable = false;
      config_inplace(inplace_);
    }
    return ret;
  }

  DepthwiseConvParam& param() { return param_; }

 private:
  DepthwiseConvParam param_;
  Tensor scale_bias_;
  InplaceArgs inplace_ = {0};
  int align_repeat_ = 1;
};

}  // namespace zynqmp
}  // namespace paddle
