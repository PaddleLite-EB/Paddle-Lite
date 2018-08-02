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

#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <limits>

// memory management;

namespace paddle_mobile {
namespace fpga {

int open_device();
int close_device();

void* fpga_malloc(size_t size);
void fpga_free(void* ptr);
void fpga_copy(void* dst, const void* src, size_t num);

struct VersionArgs {
  void* buffer;
};

struct MemoryCopyArgs {
  void* src;
  void* dest;
  size_t size;
};

struct BNArgs {
  bool enabled;
  void* bias_address;
  void* scale_address;
};

/**
Conv and Pooling kernel
*/
struct KernelArgs {
  uint32_t width;
  uint32_t height;
  uint32_t stride_w;
  uint32_t stride_h;
};

struct ImageInputArgs {
  void* address;         // input featuremap virtual address
  float* scale_address;  // input scale address;
  uint32_t channels;
  uint32_t width;  // featuremap width
  uint32_t height;
  uint32_t pad_width;  // padding width;
  uint32_t pad_height;
};

struct ImageOutputArgs {
  void* address;         // output result address;
  float* scale_address;  // output scale address;
};

struct ConvArgs {
  bool relu_enabled;
  void* bias_address;
  void* filter_address;
  uint32_t filter_num;
  uint32_t group_num;

  struct BNArgs bn;
  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
};

struct PoolingArgs {
  struct KernelArgs kernel;
  struct ImageInputArgs image;  // input image;
  struct ImageOutputArgs output;
};

// elementwise add arguments
struct EWAddArgs {
  bool relu_enabled;

  float const0;  // output0 = const0 x input0 + const1 x input1;
  float const1;
  struct ImageInputArgs image0;
  struct ImageInputArgs image1;
  struct ImageOutputArgs output;
};

struct FpgaRegWriteArgs {
  uint64_t address;  //
  uint64_t value;
};

struct FpgaRegReadArgs {
  uint64_t address;
  uint64_t value;
};

#define IOCTL_FPGA_MAGIC 'FPGA'

#define IOCTL_VERSION _IOW(IOCTL_FPGA_MAGIC, 01, struct VersionArgs)
#define IOCTL_FPGA_REG_READ _IOW(IOCTL_FPGA_MAGIC, 02, struct FpgaRegReadArgs)
#define IOCTL_FPGA_REG_WRITE _IOW(IOCTL_FPGA_MAGIC, 03, struct FpgaRegWriteArgs)

#define IOCTL_SEPARATOR_0 10

#define IOCTL_MEM_COPY _IOW(IOCTL_FPGA_MAGIC, 11, struct MemoryCopyArgs)

#define IOCTL_SEPARATOR_1 20

#define IOCTL_CONFIG_CONV _IOW(IOCTL_FPGA_MAGIC, 21, struct ConvArgs)
#define IOCTL_CONFIG_POOLING _IOW(IOCTL_FPGA_MAGIC, 22, struct PoolingArgs)
#define IOCTL_CONFIG_EW _IOW(IOCTL_FPGA_MAGIC, 23, struct EWAddArgs)

enum FPGA_ERR_TYPE {
  ERR_IOCTL_CMD = -1,
  ERR_TIMEOUT = -2,
  ERR_COMPLETION_TIMEOUT = -3,
  ERR_INVALID_FPGA_ADDR = -4,
  ERR_NOMEM = -5,
  ERR_NO_RESERVE_MEM = -6,
  ERR_COPY_FROM_USER = -7,
  ERR_COPY_TO_USER = -8,
  ERR_DEL_TIMER = -9,
  ERR_ENABLE_MSI = -10,
  ERR_REGISTER_IRQ = -11,
  ERR_PCIE_REGISTER = -12,
  ERR_PCIE_PROBE = -13,
  ERR_REGISTER_BLOCK = -14,
  ERR_ALLOC_GENDISK = -15,
  ERR_INIT_QUEUE = -16,
  ERR_WAIT = -17,
  ERR_ECC_ERROR = -31,
  ERR_FPGA_FAIL_STOP = -64,
  ERR_FPGA_DEBUG_STOP = -113,
  DEV_TMP_UNAVAILABLE = -128
};

//============================== API =============================

int ComputeFpgaConv(struct ConvArgs args);
int ComputeFpgaPool(struct PoolingArgs args);
int ComputeFpgaEWAdd(struct EWAddArgs args);

}  // namespace fpga
}  // namespace paddle_mobile
