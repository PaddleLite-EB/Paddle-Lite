// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/backends/fpga/KD/pes/cpu_pe.hpp"
#include "lite/backends/fpga/KD/pes/bypass_pe.hpp"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class BoxCoderCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::BoxCoderParam;

  void PrepareForRun() override;
  void Run() override;

  virtual ~BoxCoderCompute() = default;
 private:
  std::unique_ptr<zynqmp::CPUPE> cpu_pe_;
  std::unique_ptr<zynqmp::BypassPE> bypass_pe_;
  Tensor target_box_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
