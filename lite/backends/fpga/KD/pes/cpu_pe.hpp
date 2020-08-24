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
#include <memory>
#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

#include "lite/backends/fpga/KD/dispatch/transaction_manager.hpp"

namespace paddle {
namespace zynqmp {

class CPUPE : public PE {
 public:
  bool init() {
    for (Tensor* t : param_.outputs) {
      t->scaleIndex(true);
    }
    return true;
  }

  void apply() {
    transaction_ = TransactionManager::get_instance().getTransaction();
    TransactionManager::get_instance().endTransaction();
  }

  bool dispatch() { transaction_->startTraction(); }

  CPUParam& param() { return param_; }

 private:
  CPUParam param_;

  std::shared_ptr<Transaction> transaction_;
};

}  // namespace zynqmp
}  // namespace paddle
