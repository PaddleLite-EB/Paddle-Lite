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

#pragma once

#include <string>
#include <type_traits>
#include <vector>

#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_image.h"
#include "framework/cl/cl_scope.h"

namespace paddle_mobile {
namespace framework {

class CLHelper {
 public:
  CLHelper() = default;

  explicit CLHelper(CLScope *scope) : scope_(scope) {}

  void AddKernel(const std::string &kernel_name, const std::string &file_name) {
    auto kernel = scope_->GetKernel(kernel_name, file_name);
    kernels.emplace_back(std::move(kernel));
  }

  cl_kernel KernelAt(const int index) { return kernels[index].get(); }

  cl_command_queue CLCommandQueue() { return scope_->CommandQueue(); }

  cl_context CLContext() { return scope_->Context(); }

  std::vector<size_t> DefaultWorkSize(const CLImage &image) {
    // n c h w
    auto image_dim = image.dims();
    if (image_dim.size() == 4) {
      auto n = image_dim[0];
      auto h = image_dim[2];
      auto w = image_dim[3];

      auto image_width = image.ImageWidth();

      auto work_size_0 = image_width / w;

      auto work_size_1 = w;

      auto work_size_2 = n * h;

      return {work_size_0, work_size_1, work_size_2};
    }
    PADDLE_MOBILE_THROW_EXCEPTION("not support this dim, need imp");
  }

 private:
  CLScope *scope_;
  std::vector<std::unique_ptr<_cl_kernel, CLKernelDeleter>> kernels;
};

}  // namespace framework
}  // namespace paddle_mobile
