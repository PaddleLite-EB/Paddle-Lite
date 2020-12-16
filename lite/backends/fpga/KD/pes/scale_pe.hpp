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
#include "lite/backends/fpga/KD/pes/depthwise_conv_pe.hpp"
#include "lite/backends/fpga/KD/tensor.hpp"

namespace paddle {
namespace zynqmp {

class ScalePE : public PE {
 public:
  inline int gcd(int a, int b) {
    while (b) {
      int temp = a;
      a = b;
      b = temp % b;
    }
    return a;
  }

  inline int lcm(int a, int b) { return a * b / gcd(a, b); }
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
    Shape& input_shape = input->shape(); 
    DepthwiseConvParam& dw_param = dw_pe_.param();

    int channel = input_shape.channel();
    
    // TODO FPGA限制 H >2047, W >1023 , WC> 65536 ，需要使用CPU实现
    Shape shape(N, {channel});

    float* filter_data = filter.mutableData<float>(FP32, shape);
    std::fill_n(filter_data, input->shape().channel(), 1.0f);
  
    Tensor *scale = dw_param.scale();
    float16* scale_data = scale->mutableData<float16>(FP16, shape);

    Tensor *bias = dw_param.bias();
    float16* bias_data = bias->mutableData<float16>(FP16, shape);
    std::fill_n(bias_data, input->shape().channel(), 0);

    if (param_.scale->dataType() == FP32) {
        if (param_.bias != nullptr) {
            float* bias_data_float = param_.bias->data<float>();
            for (int j = 0; j < channel; j++) {
                bias_data[j] = float_to_half(bias_data_float[j]);
            }
        } else {
            float16 zero = float_to_half(0.0f);
            for (int j = 0; j < channel; j++) {
                bias_data[j] = zero;
            }
        }
        float* scale_data_float = param_.scale->data<float>();
        for (int j = 0; j < channel; j++) {
            scale_data[j] = float_to_half(scale_data_float[j]);
        }
    } else {
        if (param_.bias != nullptr) {
            bias_data = param_.bias->data<float16>();
        } else {
            float16 zero = float_to_half(0.0f);
            for (int j = 0; j < channel; j++) {
                bias_data[j] = zero;
            }
        }
        memcpy(scale_data, param_.scale->data<float16>(), param_.scale->memorySize());
    }

    scale->flush();
    bias->flush();
    filter.flush();

    dw_param.input = param_.input;
    dw_param.output = param_.output;
    dw_param.filter = &filter;
    dw_param.activeParam.type = param_.activeParam.type;
    
    dw_param.strides = {1, 1};
    dw_param.paddings = {0, 0};
    dw_param.kernelSize = {1, 1};
    dw_param.dilations = {1, 1};

    dw_pe_.init(&fpga_lock);
    dw_pe_.apply(&fpga_lock);
  }

  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    Tensor float_input;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    // input->syncToCPU();
    // input->invalidate();
    float_input.copyFrom(input);
    float16* data_out = output->data<float16>();

    float16* scale_data = param_.scale->data<float16>();

    int wh = input->shape().width() * input->shape().height();

    float16* in_data = input->data<float16>();
    float max = 0;

    for (int i = 0; i < wh; i++) {
      for (int c = 0; c < input->shape().channel(); c++) {
        int index = i * input->shape().channel() + c;
        float x = image_addr[index];
        float y = half_to_float(scale_data[c]);
        float value = x * y;
        // std::cout << " x = " << std::to_string(x) << " y = " <<
        // std::to_string(y) << " v = " << std::to_string(value) << std::endl;
        // float value = half_to_float(in_data[index]) * 19.3598f;
        data_out[index] = float_to_half(value);

        if (value < 0) {
          value = -value;
        }
        if (value > max) {
          max = value;
        }
      }
    }
    // exit(-1);
    output->flush();
    output->scale()[0] = max / 127.0f;
    output->scale()[1] = 127.0f / max;
    output->max()[0] = float_to_half(max);
  }

  bool dispatch(FPGALock* lock = nullptr) {
    FPGALock fpga_lock(lock);
    fpga_lock.lock();

    if (param_.re_assign == true) {
        DepthwiseConvParam& dw_param = dw_pe_.param();
        
        memcpy(dw_param.scale()->mutableData<float16>(), param_.scale->data<float16>(), param_.scale->memorySize());

        dw_param.re_assign = true;
    }
    
    param_.input->syncToDevice();
    return dw_pe_.dispatch(&fpga_lock);

    // cpu_compute();
    return true;
  }

  ScaleParam& param() { return param_; }

 private:
  ScaleParam param_;
  Tensor filter;
  DepthwiseConvPE dw_pe_;
};
}  // namespace zynqmp
}  // namespace paddle
