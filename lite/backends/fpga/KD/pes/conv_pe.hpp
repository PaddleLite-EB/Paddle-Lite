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

#include <arm_neon.h>
#include <algorithm>
#include <memory>
#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"
#include "lite/backends/fpga/KD/pes/cpu_pe.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"

#include "lite/backends/fpga/KD/dispatch/transaction_manager.hpp"

namespace paddle {
namespace zynqmp {

class ConvPE : public PE {
 public:
  void cpu_conv_half_hwc() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;

    Shape& input_shape = input->shape();
    Shape& out_shape = output->shape();

    int image_height = input_shape.height();
    int image_width = input_shape.width();
    int image_channels = input_shape.channel();
    int image_pad_h = param_.paddings[0];
    int image_pad_w = param_.paddings[0];
    int kernel_height = param_.filter->shape().height();
    int kernel_width = param_.filter->shape().width();
    int kernel_step_h = param_.strides[0];
    int kernel_step_w = param_.strides[1];
    int dilation_rate = 1;
    int out_channel = out_shape.channel();
    int pooled_height_ = out_shape.height();
    int pooled_width_ = out_shape.width();
    int filter_chw = image_channels * kernel_height * kernel_width;

    int kernel_rw = kernel_width + (dilation_rate - 1) * (kernel_width - 1);
    int kernel_rh = kernel_height + (dilation_rate - 1) * (kernel_height - 1);

    float* weight = param_.filter->data<float>();

    Tensor float_input;
    Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    input->syncToDevice();
    float_input.copyFrom(input);
    float_input.invalidate();
    float_input.saveToFile("fi", true);

    float* out = float_output.mutableData<float>(FP32, output->shape());

    for (int ph = 0; ph < pooled_height_; ph++) {
      for (int pw = 0; pw < pooled_width_; pw++) {
        int hstart = ph * kernel_step_h - image_pad_h;
        int wstart = pw * kernel_step_w - image_pad_w;
        int hend = std::min(hstart + kernel_rh, static_cast<int>(image_height));
        int wend = std::min(wstart + kernel_rw, static_cast<int>(image_width));

        int hstart_plus =
            dilation_rate *
                ceil(static_cast<float>(image_pad_h - ph * kernel_step_h)) /
                static_cast<float>(dilation_rate) -
            image_pad_h + ph * kernel_step_h;
        int wstart_plus =
            dilation_rate *
                ceil(static_cast<float>(image_pad_w - pw * kernel_step_w) /
                     static_cast<float>(dilation_rate)) -
            image_pad_w + pw * kernel_step_w;

        int hstart_ = hstart < 0 ? hstart_plus : hstart;
        int wstart_ = wstart < 0 ? wstart_plus : wstart;

        for (int oc = 0; oc < out_channel; oc++) {
          float sum = 0.0f;
          const int pool_index = (ph * pooled_width_ + pw) * out_channel + oc;
          for (int c = 0; c < image_channels; c++) {
            for (int h = hstart_; h < hend; h += dilation_rate) {
              int hi = 0;
              if (hstart < 0) {
                hi = (kernel_rh - (hend - h)) / dilation_rate;
              } else {
                hi = (h - hstart_) / dilation_rate;
              }

              for (int w = wstart_; w < wend; w += dilation_rate) {
                int wi = 0;
                if (wstart < 0) {
                  wi = (kernel_rw - (wend - w)) / dilation_rate;
                } else {
                  wi = (w - wstart_) / dilation_rate;
                }

                const int index = (h * image_width + w) * image_channels + c;
                int weight_index = oc * filter_chw +
                                   kernel_width * kernel_height * c +
                                   kernel_width * hi + wi;
                float value = image_addr[index] * weight[weight_index];
                sum += value;
              }
            }
          }
          float s = param_.scale()->data<float>()[oc];
          float b = param_.bias()->data<float>()[oc];
          out[pool_index] = sum * s + b;
        }
      }
    }
    float_output.flush();
    float_output.saveToFile("fo", true);
    output->copyFrom(&float_output);
    output->invalidate();
    output->saveToFile("out", true);
    // exit(-1);
  }

  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    // cpu_pe_.reset(new CPUPE());
    // cpu_pe_->init();
    return true;
  }

  void apply() {

    if (param_.deconv == false) {
      split_axis = fill_split_arg(param_);
      pack_channel = split_axis == 2 && param_.splitParams().size() > 1;
      split_cpu_concat = split_axis == 0 && param_.cpu_concat;
      split_channel = split_axis == 1;

      if (pack_channel || split_channel) {
        SplitParam& split_param = splitPE_.param();
        split_param.input = param_.input;
        for (auto conv_param : param_.splitParams()) {
          split_param.outputs.push_back(&conv_param->input);
        }
        splitPE_.init();
        splitPE_.apply();
      }

      // ======================= dispatch =======================
      transaction_ = TransactionManager::get_instance().getTransaction();
     
      for (int i = 0; i < param_.splitParams().size(); i++) {
        auto conv_param = param_.splitParams()[i];
        conv_param->args.output_idx = param_.output->scaleIndex(true);
        conv_param->args.inplace.findmax_restart = i == 0;
        conv_param->args.inplace.active_param.type = param_.activeParam.type;
        conv_param->args.inplace.active_param.leaky_relu_factor =
            float_to_half(param_.activeParam.leaky_relu_factor);
        int action_id = compute_fpga_conv_basic(conv_param->args);
        Action* action = new Action(action_id);
        actions_.push_back(action);
        transaction_->appendAction(action);
      }  

      // ======================= concat =======================
      if (pack_channel) {
        ConcatParam& concat_param = concatPE_.param();
        for (auto conv_param : param_.splitParams()) {
          concat_param.inputs.push_back(&conv_param->output);
        }
        concat_param.output = param_.output;
        concatPE_.init();
        concatPE_.apply();
        concatPE_.setMergeScale(false);  // currently don't need handle scale, find_max not restart
 
      } else if (split_cpu_concat) {
        ConcatParam& concat_param = concatPE_.param();

        BasicConvParam* first = param_.splitParams().front();
        concat_param.inputs.push_back(&(first->output));

        BasicConvParam* last = param_.splitParams().back();
        concat_param.inputs.push_back(&(last->output));

        concat_param.output = param_.output;
        concatPE_.init();
        concatPE_.apply();
        concatPE_.setMergeScale(false);  //???
      }

      std::vector<BasicConvParam*>& params = param_.splitParams();
      if (split_channel && params.size() > 1) {
        ElementwiseAddParam& add_param = addPE_.param();
        add_param.inputs = {&params[0]->output, &params[1]->output};
        add_param.output = param_.output;
        addPE_.init();
        addPE_.apply();
      }
    }

    if (!use_cpu_) {
      param_.filter->releaseData();
    }
    // cpu_pe_->apply();
  }

  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    // float16* data_out = output->data<float16>();
    float* out = float_output.mutableData<float>(FP32, output->shape());

    int out_channel = output->shape().channel();
    int in_channel = input->shape().channel();

    float* filter_data = param_.filter->data<float>();
    float* mi = new float[in_channel];
    for (int i = 0; i < out_channel; i++) {
      float* image = image_addr;
      float* filter_ptr = filter_data + i * in_channel;
      float* out_ptr = mi;
#pragma omp parallel for
      for (int j = 0; j < in_channel; j++) {
        float value = image_addr[j] * filter_ptr[j];
        mi[j] = value;
      }

      float sum = 0;
      for (int j = 0; j < in_channel; j++) {
        sum += mi[j];
      }
      out[i] = sum;
    }
    delete[] mi;
    float_output.flush();
    output->flush();
    output->copyFrom(&float_output);
    output->invalidate();
  }

  bool dispatch() {
    if (use_cpu_) {
      cpu_pe_->dispatch();
      cpu_compute();
      return true;
    }

    if ((pack_channel || split_channel) && !param_.deconv) {
      splitPE_.dispatch();
    }

    if ((pack_channel || split_cpu_concat) && !param_.deconv) {
      float scale = param_.output->scale()[0];
      concatPE_.dispatch();
    }
    
    std::vector<BasicConvParam*>& params = param_.splitParams();
    if (split_channel && params.size() > 1) {
      ElementwiseAddParam& add_param = addPE_.param();
      // add_param.inputs = {&params[0]->output, &params[1]->output};
      // add_param.output = param_.output;
      addPE_.dispatch();
    }

    return true;
  }

  ConvParam& param() { return param_; }

  ~ConvPE() {
    for (int i = 0; i < actions_.size(); i++) {
      Action* action = actions_[i];
      delete action;
    }
    actions_.clear();
  }

 private:
  bool use_cpu_ = false;
  bool pack_channel = false;
  bool split_cpu_concat = false;
  bool split_channel = false;
  ConvParam param_;
  ConcatPE concatPE_;
  SplitPE splitPE_;
  ElementwiseAddPE addPE_;
  int split_axis = 0;
  InplaceArgs inplace_ = {0};
  ActiveParamterArgs activeParamterArgs;

  // =================
  std::shared_ptr<Transaction> transaction_;
  std::vector<Action*> actions_;
  std::unique_ptr<CPUPE> cpu_pe_;
};

}  // namespace zynqmp
}  // namespace paddle
