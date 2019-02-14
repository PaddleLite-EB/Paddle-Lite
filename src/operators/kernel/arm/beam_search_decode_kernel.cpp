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

#ifdef BEAM_SEARCH_DECODE_OP

#include "operators/kernel/beam_search_decode_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool BeamSearchDecodeKernel<CPU, float>::Init(
    BeamSearchDecodeParam<CPU> *param) {
  return true;
}

template <>
void BeamSearchDecodeKernel<CPU, float>::Compute(
    const BeamSearchDecodeParam<CPU> &param) {
  // TODO(hjchen2)
  DLOG << "BeamSearchDecodeKernel";
  param.sentence_scores_->Resize(framework::make_ddim({10}));
  param.sentence_scores_->mutable_data<float>();
  DLOG << "BeamSearchDecodeKernel";

  param.sentence_ids_->Resize(framework::make_ddim({10}));
  param.sentence_ids_->mutable_data<int64_t>();
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
