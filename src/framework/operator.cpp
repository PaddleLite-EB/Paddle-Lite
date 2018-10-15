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

#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace framework {

template <typename Dtype>
vector<string> OperatorBase<Dtype>::GetOutKeys() const {
  auto it = op_input_output_key.find(type_);
  if (it == op_input_output_key.end()) {
    DLOG << type_ << " has no outputs";
    return {};
  }
  return it->second.second;
}

template <typename Dtype>
vector<string> OperatorBase<Dtype>::GetInputKeys() const {
  auto it = op_input_output_key.find(type_);
  if (it == op_input_output_key.end()) {
    DLOG << type_ << " has no outputs";
    return {};
  }
  return it->second.first;
}

template <typename Dtype>
OperatorBase<Dtype>::OperatorBase(const std::string &type,
                                  const VariableNameMap &inputs,
                                  const VariableNameMap &outputs,
                                  const AttributeMap &attrs,
                                  std::shared_ptr<Scope> scope)
    : type_(type),
      inputs_(inputs),
      outputs_(outputs),
      attrs_(attrs),
      scope_(scope) {
  CheckAllInputOutputSet();
}

template <typename Dtype>
void OperatorBase<Dtype>::CheckAllInputOutputSet() const {}

template <typename Dtype>
void OperatorBase<Dtype>::Run() {
  RunImpl();
#ifdef PADDLE_MOBILE_DEBUG
  DLOG << "-------------" << type_ << "----------------------------";
  vector<string> input_keys = GetInputKeys();
  for (const auto key : input_keys) {
    auto var_vec_in = inputs_.at(key);
    for (int i = 0; i < var_vec_in.size(); ++i) {
      auto vari = scope_->FindVar(var_vec_in[i]);
      if (vari->IsInitialized()) {
#ifdef PADDLE_MOBILE_CL
        if (type_ == "feed") {
          Tensor *tensor = vari->template GetMutable<framework::LoDTensor>();
          if (tensor) DLOG << type_ << " input- " << key << "=" << *tensor;
        } else {
          CLImage *cl_image = vari->template GetMutable<framework::CLImage>();
          //                        cl_command_queue commandQueue =
          //                        scope_->GetCLScpoe()->CommandQueue(); Tensor
          //                        *tmp ;
          //                        CLImageToTensor(cl_image,tmp,commandQueue);
          //                        tmp->Resize(cl_image->dims());
          const float *input = cl_image->data<float>();
          if (cl_image) {
            DLOG << type_ << " input- " << key << "=" << cl_image->dims();
            //              if(input)
            //              DLOG<<type_<<" input- "<<key<<"="<<*input;
          }
        }

#else
        Tensor *tensor = vari->template GetMutable<framework::LoDTensor>();
        if (tensor) DLOG << type_ << " input- " << key << "=" << *tensor;
#endif
      }
    }
  }
  for (const auto key : GetOutKeys()) {
    auto var_vec_out = outputs_.at(key);
    for (int i = 0; i < var_vec_out.size(); ++i) {
      auto vari = scope_->FindVar(var_vec_out[i]);
      if (vari->IsInitialized()) {
#ifdef PADDLE_MOBILE_CL
        if (type_ == "fetch") {
          Tensor *tensor = vari->template GetMutable<framework::LoDTensor>();
          if (tensor)
            DLOG << type_ << " output- " << key << "=" << tensor->dims();
        } else {
          CLImage *cl_image = vari->template GetMutable<framework::CLImage>();
          //          cl_command_queue commandQueue =
          //          scope_->GetCLScpoe()->CommandQueue(); Tensor *tmp ;
          //          CLImageToTensor(cl_image,tmp,commandQueue);
          //          tmp->Resize(cl_image->dims());
          if (cl_image) {
            const float *output = cl_image->data<float>();
            DLOG << type_ << " output- " << key << "=" << cl_image->dims();
            //                  if(output)
            //                  DLOG<<type_<<" output- "<<key<<"="<<*output;
          }
        }

#else
        Tensor *tensor = vari->template GetMutable<framework::LoDTensor>();
        if (tensor) DLOG << type_ << " output- " << key << "=" << *tensor;
#endif
      }
    }
  }
#endif
}

template class OperatorBase<CPU>;
template class OperatorBase<FPGA>;
template class OperatorBase<GPU_MALI>;
template class OperatorBase<GPU_CL>;

}  // namespace framework
}  // namespace paddle_mobile
