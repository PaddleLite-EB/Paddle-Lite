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

#include <stdio.h>
#include <string>
#include <vector>

#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"
#include "lite/backends/fpga/KD/tensor.hpp"

namespace paddle {
namespace zynqmp {

struct ActiveParam {
  enum ActiveType type = TYPE_NONE;
  float leaky_relu_factor = 0.0f;
};

struct PEParam {
  ActiveParam activeParam;
};

struct BypassParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* output = nullptr;
};

using InputParam = BypassParam;
using OutputParam = BypassParam;

struct BatchnormParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* output = nullptr;

  Tensor* bias = nullptr;
  Tensor* scale = nullptr;
  Tensor* mean = nullptr;
  Tensor* variance = nullptr;
  float epsilon = 0;
};

struct BasicConvParam {
  Tensor input;
  Tensor output;
  Tensor filter;
  Tensor scaleBias;
  ConvArgs args;
};

struct ConvParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* output = nullptr;
  Tensor* filter = nullptr;

  int groups = 1;
  std::vector<int> strides;
  std::vector<int> paddings;
  std::vector<int> kernelSize;
  std::vector<int> dilations;

  Tensor* scale() { return &scale_; }

  Tensor* bias() { return &bias_; }

  std::vector<BasicConvParam*>& splitParams() { return splitParams_; }

  ~ConvParam() {
    for (int i = 0; i < splitParams_.size(); i++) {
      BasicConvParam* basic_param = splitParams_[i];
      delete basic_param;
    }
    splitParams_.clear();
  }

 protected:
  std::vector<BasicConvParam*> splitParams_;
  Tensor scale_;
  Tensor bias_;
};

struct DepthwiseConvParam : ConvParam {
 public:
  Tensor* quantizedFilter() { return &quantizedFilter_; }

  DWconvArgs args;

 protected:
  Tensor quantizedFilter_;
};

enum PoolingType : int {
  MAX = 0,
  AVERAGE = 1,
};

struct PoolingParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* output = nullptr;

  PoolingType type = PoolingType::MAX;
  bool globalPooling = false;
  std::vector<int> kernelSize;
  std::vector<int> strides;
  std::vector<int> paddings;

  PoolingArgs poolingArgs = {0};
};

struct ConcatParam : PEParam {
 public:
  std::vector<Tensor*> inputs;
  Tensor* output;
  int axis = 0;
};

struct ElementwiseAddParam : PEParam {
 public:
  std::vector<Tensor*> inputs;
  Tensor* output = nullptr;
  int axis = 0;

  EWAddArgs ewargs;
};

struct ElementwiseMulParam : PEParam {
 public:
  Tensor* input_x = nullptr;
  Tensor* input_y = nullptr;
  Tensor* output = nullptr;
};

struct FullyConnectedParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* filter = nullptr;
  Tensor* bias = nullptr;
  Tensor* output = nullptr;

  Tensor* quantizedFilter() { return &quantizedFilter_; }

  Tensor* biasScale() { return &biasScale_; }

 protected:
  Tensor quantizedFilter_;
  Tensor biasScale_;
};

struct SoftmaxParam : PEParam {
 public:
  Tensor* input = nullptr;

  Tensor* output = nullptr;

 private:
  Tensor* floatInput = nullptr;
};

struct SplitParam : PEParam {
 public:
  Tensor* input = nullptr;
  std::vector<Tensor*> outputs;
  int axis = 1;
  int num = 1;
};

struct NormParam : PEParam {
 public:
  Tensor* input = nullptr;

  Tensor* output = nullptr;
  float epsilon = 0;

 private:
  Tensor* floatInput = nullptr;
};

struct PriorBoxParam : PEParam {
  Tensor* input = nullptr;
  Tensor* image = nullptr;
  Tensor* outputBoxes = nullptr;
  Tensor* outputVariances = nullptr;

  std::vector<float> minSizes;
  std::vector<float> maxSizes;
  std::vector<float> aspectRatios;
  std::vector<float> variances;

  bool minMaxAspectRatiosOrder;
  bool flip;
  bool clip;
  float stepW;
  float stepH;
  float offset;
};

struct YoloBoxParam : PEParam {
  Tensor* input = nullptr;
  Tensor* imgSize = nullptr;
  Tensor* outputBoxes = nullptr;
  Tensor* outputScores = nullptr;
  int downsampleRatio;
  std::vector<int> anchors;
  int classNum;
  float confThresh;
};

struct ScaleParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* output = nullptr;
  Tensor* scale = nullptr;
  Tensor* bias = nullptr;

  Tensor* alignedScale() { return &alignedScale_; }

  Tensor* alignedBias() { return &alignedBias_; }

  ScaleArgs args = {0};

 protected:
  Tensor alignedScale_;
  Tensor alignedBias_;
};

struct ResizeParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* output = nullptr;
};

struct CropParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* output = nullptr;
  int axis = 2;
  std::vector<int> offsets;
  std::vector<int> shape;
};

struct GRUParam : PEParam {
 public:
  Tensor* input = nullptr;
  Tensor* h0 = nullptr;
  Tensor* weight = nullptr;
  Tensor* bias = nullptr;

  Tensor* batch_gate = nullptr;
  Tensor* batch_reset_hidden_prev = nullptr;
  Tensor* batch_hidden = nullptr;
  Tensor* hidden = nullptr;

  std::string gate_activation = "sigmoid";
  std::string activation = "tanh";
  bool is_reverse = false;
  bool origin_mode = false;
};

struct CPUParam : PEParam {
 public:
  std::vector<Tensor*> inputs;
  std::vector<Tensor*> outputs;
};

}  // namespace zynqmp
}  // namespace paddle
