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

#include <algorithm>
#include <vector>

#include "lite/backends/fpga/KD/dispatch/action.hpp"
#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"

#pragma once

namespace paddle {
namespace zynqmp {

class Transaction {
 public:
  void appendAction(Action* action) {
    if (!actions_.empty()) {
      Action* last = actions_.back();
      link_actions(last->id(), action->id());
    }
    actions_.push_back(action);
  }

  void startTraction() {
    if (actions_.size() > 0) {
      struct CnnCmdArgs args;
      Action* action = actions_[0];
      args.action_id = action->id();
      start_transaction(args);
    }
  }

 private:
  std::vector<Action*> actions_;
  int id_ = -1;
};
}  // namespace zynqmp
}  // namespace paddle
