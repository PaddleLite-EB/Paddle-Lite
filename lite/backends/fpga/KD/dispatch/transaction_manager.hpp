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

#include "lite/backends/fpga/KD/dispatch/transaction.hpp"

#include <vector>

namespace paddle {
namespace zynqmp {

class TransactionManager {
 public:
  static TransactionManager& get_instance() {
    static TransactionManager s_instance;
    return s_instance;
  }

  Transaction* getTransaction() {
    if (currentTransaction_ == nullptr) {
      currentTransaction_ = new Transaction();
      transactions_.push_back(currentTransaction_);
    }
    return currentTransaction_;
  }

  void endTransaction() { currentTransaction_ = nullptr; }

 private:
  Transaction* currentTransaction_ = nullptr;
  std::vector<Transaction*> transactions_;
};
}  // namespace zynqmp
}  // namespace paddle
