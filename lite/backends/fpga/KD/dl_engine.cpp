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
#include "lite/backends/fpga/KD/dl_engine.hpp"
// #include "lite/backends/fpga/KD/repo_version.h"
#include "lite/core/version.h"

namespace paddle {
namespace zynqmp {

DLEngine::DLEngine() {
  open_device();
  int ret = get_device_info(info_);
  filter::set_filter_capacity(info_.filter_cap);
  filter::set_colunm(info_.column);

  struct VersionArgs args = {.buffer = new char[21], .size = 21};
  ret = get_version(args);

  if (ret == 0) {
    char paddle_lite_version[] = "1.5.3";
    char* driver_version = reinterpret_cast<char*>(args.buffer);
    char dest[5];
    strncpy(dest, driver_version, 5);

    int ret = strcmp(dest, paddle_lite_version);
    if (ret == 0) {
      std::cout << "driver_version: " << std::string(dest) << std::endl;
      std::cout << "paddle_lite_version: " << std::string(paddle_lite_version)
                << std::endl;
    } else {
      std::cout << "driver_version(" << std::string(dest)
                << ") not match paddle_lite_version("
                << std::string(paddle_lite_version) << ") " << std::endl;
    }
    delete[] args.buffer;
  }

  std::cout << "[GitVersion] " << GIT_VERSION << std::endl;
  std::cout << "[GitDate] " << GIT_DATE << std::endl;
  std::cout << "[GitHash] " << GIT_HASH << std::endl;
  std::cout << "[BuildDate] " << __DATE__ << std::endl;
  std::cout << "[BuildTime] " << __TIME__ << std::endl;

  std::cout << "paddle_lite_commit:" << lite::paddlelite_commit() << std::endl;
  std::cout << "paddlelite_branch:" << lite::paddlelite_branch() << std::endl;
  std::cout << "paddlelite_tag:" << lite::paddlelite_tag() << std::endl;
  std::cout << "version:" << lite::version() << std::endl;

  // printf("version: %s\n", REPO_VERSION);
  // printf("date: %s\n", REPO_DATE);
  // printf("hash: %s\n", REPO_HASH);
}

}  // namespace zynqmp
}  // namespace paddle
