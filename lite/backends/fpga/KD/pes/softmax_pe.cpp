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

#include "lite/backends/fpga/KD/pes/softmax_pe.hpp"

#include <vector>

namespace paddle {
namespace zynqmp {

#ifndef c_exp_hi
#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1
#endif

// exp() computed for 4 float at once
inline float32x4_t exp_ps(float32x4_t x) {
  float32x4_t tmp, fx;

  float32x4_t one = vdupq_n_f32(1);
  x = vminq_f32(x, vdupq_n_f32(c_exp_hi));
  x = vmaxq_f32(x, vdupq_n_f32(c_exp_lo));

  // express exp(x) as exp(g + n*log(2))
  fx = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(c_cephes_LOG2EF));

  // perform a floorf
  tmp = vcvtq_f32_s32(vcvtq_s32_f32(fx));

  // if greater, substract 1
  uint32x4_t mask = vcgtq_f32(tmp, fx);
  mask = vandq_u32(mask, vreinterpretq_u32_f32(one));

  fx = vsubq_f32(tmp, vreinterpretq_f32_u32(mask));

  tmp = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C1));
  float32x4_t z = vmulq_f32(fx, vdupq_n_f32(c_cephes_exp_C2));
  x = vsubq_f32(x, tmp);
  x = vsubq_f32(x, z);

  static const float cephes_exp_p[6] = {c_cephes_exp_p0,
                                        c_cephes_exp_p1,
                                        c_cephes_exp_p2,
                                        c_cephes_exp_p3,
                                        c_cephes_exp_p4,
                                        c_cephes_exp_p5};
  float32x4_t y = vld1q_dup_f32(cephes_exp_p + 0);
  float32x4_t c1 = vld1q_dup_f32(cephes_exp_p + 1);
  float32x4_t c2 = vld1q_dup_f32(cephes_exp_p + 2);
  float32x4_t c3 = vld1q_dup_f32(cephes_exp_p + 3);
  float32x4_t c4 = vld1q_dup_f32(cephes_exp_p + 4);
  float32x4_t c5 = vld1q_dup_f32(cephes_exp_p + 5);

  y = vmulq_f32(y, x);
  z = vmulq_f32(x, x);

  y = vaddq_f32(y, c1);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c2);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c3);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c4);
  y = vmulq_f32(y, x);
  y = vaddq_f32(y, c5);

  y = vmulq_f32(y, z);
  y = vaddq_f32(y, x);
  y = vaddq_f32(y, one);

  // build 2^n
  int32x4_t mm;
  mm = vcvtq_s32_f32(fx);
  mm = vaddq_s32(mm, vdupq_n_s32(0x7f));
  mm = vshlq_n_s32(mm, 23);
  float32x4_t pow2n = vreinterpretq_f32_s32(mm);

  y = vmulq_f32(y, pow2n);
  return y;
}

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#ifndef __aarch64__
static inline float32_t vmaxvq_f32(const float32x4_t &r) {
  float32x2_t v = vmax_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpmax_f32(v, v), 0);
}

static inline float32_t vaddvq_f32(const float32x4_t &r) {
  float32x2_t v = vadd_f32(vget_high_f32(r), vget_low_f32(r));
  return vget_lane_f32(vpadd_f32(v, v), 0);
}
#endif  // __aarch64__
#endif  // __ARM_NEON__

static float find_max(const float *input, const int num_classes) {
  int remain = num_classes;
  float max = -std::numeric_limits<float>::max();
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  int loop = num_classes >> 3;
  remain = num_classes & 0x7;
  float32x4_t __max = vdupq_n_f32(max);
  for (int i = 0; i < loop; ++i, input += 8) {
    float32x4_t x0 = vld1q_f32(input);
    float32x4_t x1 = vld1q_f32(input + 4);
    __max = vmaxq_f32(x0, __max);
    __max = vmaxq_f32(x1, __max);
  }
  max = vmaxvq_f32(__max);
#endif
  for (int i = 0; i < remain; ++i) {
    max = std::max(max, input[i]);
  }
  return max;
}

static void softmax(Tensor *X, Tensor *Y) {
  std::vector<int> dims = X->shape().dims();
  int batch_size = X->shape().num();
  int num_classes = dims[X->shape().dimSize() - 1];
  int channels = X->shape().numel() / batch_size / num_classes;

  float *x = X->data<float>();
  float *y = Y->mutableData<float>();

#pragma omp parallel for collapse(2)
  for (int batch = 0; batch < batch_size; ++batch) {
    for (int channel = 0; channel < channels; ++channel) {
      size_t offset = (batch * channels + channel) * num_classes;
      const float *input = x + offset;
      float *output = y + offset;
      // find max
      float max = find_max(input, num_classes);

      // exp(x - max)
      int remain = num_classes;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      int loop = num_classes >> 3;
      remain = num_classes & 0x7;
      float32x4_t __max = vdupq_n_f32(max);
      for (int i = 0; i < loop; ++i, input += 8, output += 8) {
        float32x4_t x0 = vld1q_f32(input);
        float32x4_t x1 = vld1q_f32(input + 4);
        x0 = vsubq_f32(x0, __max);
        x1 = vsubq_f32(x1, __max);
        x0 = exp_ps(x0);
        x1 = exp_ps(x1);
        vst1q_f32(output, x0);
        vst1q_f32(output + 4, x1);
      }
#endif  // __ARM_NEON__
      for (int i = 0; i < remain; ++i) {
        output[i] = expf(input[i] - max);
      }

      // sum(exp(x - max))
      float sum = 0.f;
      output = y + offset;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      float32x4_t __sum = vdupq_n_f32(0.f);
      for (int i = 0; i < loop; ++i, output += 8) {
        float32x4_t x0 = vld1q_f32(output);
        float32x4_t x1 = vld1q_f32(output + 4);
        __sum = vaddq_f32(x0, __sum);
        __sum = vaddq_f32(x1, __sum);
      }
      sum += vaddvq_f32(__sum);
#endif  // __ARM_NEON__
      for (int i = 0; i < remain; ++i) {
        sum += output[i];
      }

      // exp(x - max) / sum
      float inv_sum = 1.f / sum;
      output = y + offset;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
      float32x4_t __inv_sum = vdupq_n_f32(inv_sum);
      for (int i = 0; i < loop; ++i, output += 8) {
        float32x4_t x0 = vld1q_f32(output);
        float32x4_t x1 = vld1q_f32(output + 4);
        x0 = vmulq_f32(x0, __inv_sum);
        x1 = vmulq_f32(x1, __inv_sum);
        vst1q_f32(output, x0);
        vst1q_f32(output + 4, x1);
      }
#endif
      for (int i = 0; i < remain; ++i) {
        output[i] *= inv_sum;
      }
    }
  }
}

bool SoftmaxPE::init() {
  Tensor *output = param_.output;
  output->setAligned(false);
  output->setDataLocation(CPU);
  return true;
}

bool SoftmaxPE::dispatch() {
  Tensor *input = param_.input;
  Tensor *output = param_.output;

  Tensor float_input;
  Tensor float_output;
  float_input.mutableData<float>(DataType::FP32, input->shape());
  input->syncToDevice();
  float_input.copyFrom(input);

  float *out_data =
      float_output.mutableData<float>(DataType::FP32, input->shape());

  softmax(&float_input, &float_output);
  float_output.flush();

  output->copyFrom(&float_output);
  output->flush();
  return true;
}

SoftmaxParam &SoftmaxPE::param() { return param_; }
}  // namespace zynqmp
}  // namespace paddle
