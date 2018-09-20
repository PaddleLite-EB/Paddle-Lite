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

#include "image.h"
#include <memory.h>
#include "api.h"

namespace paddle_mobile {
namespace fpga {
namespace image {

void convert_to_hwc(float **data_in, int channel, int height, int width) {
  float *tmp = *data_in;
  float *data_tmp =
      (float *)fpga_malloc(channel * height * width * sizeof(float));
  int64_t amount_per_row = width * channel;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < height; h++) {
      int64_t offset_height = h * amount_per_row;
      for (int w = 0; w < width; w++) {
        *(data_tmp + offset_height + w * channel + c) = *((*data_in)++);
      }
    }
  }
  *data_in = data_tmp;
  fpga_free(tmp);
}

void align_element_conv(float **data_in, int height, int cw) {
  int h = 0;
  int align_cw = align_to_x(cw, IMAGE_ALIGNMENT);
  if (align_cw != cw) {
    float *tmp = *data_in;
    float *data_tmp = (float *)fpga_malloc(height * align_cw * sizeof(float));

    memset(data_tmp, 0, height * align_cw * sizeof(float));

    for (h = 0; h < height; h++) {
      memcpy((void *)(data_tmp + h * align_cw), (void *)(*data_in + h * cw),
             cw * sizeof(float));
    }

    *data_in = data_tmp;
    fpga_free(tmp);
  }
}

void format_image(float **data_in, int channel, int height, int width) {
  convert_to_hwc(data_in, channel, height, width);
  align_element_conv(data_in, height, channel * width);
  fpga_flush(*data_in, align_to_x(channel * width, IMAGE_ALIGNMENT) * height *
                           sizeof(float));
}

void concat_images(int16_t **images_in, float **scales_in, void *image_out,
                   float *scale_out, int image_num, uint32_t *channel_num,
                   int height, int width) {
  int i = 0;
  int j = 0;
  int k = 0;
  int each_out_line_channel = 0;
  int align_each_out_area_cw = 0;
  int align_each_in_area_cw = 0;
  int align_each_out_area_cw_differ = 0;
  int tmp_channel = 0;
  scale_out[0] = 0.0;
  scale_out[1] = 0.0;
  for (i = 0; i < image_num; i++) {
    each_out_line_channel += channel_num[i];
    scale_out[0] = std::max(*scale_out, scales_in[i][0]);
    fpga_invalidate(images_in[i],
                    height *
                        align_to_x(channel_num[i] * width, IMAGE_ALIGNMENT) *
                        sizeof(int16_t));
  }
  scale_out[1] = 1 / scale_out[0];
  align_each_out_area_cw =
      align_to_x(each_out_line_channel * width, IMAGE_ALIGNMENT);
  align_each_out_area_cw_differ =
      align_each_out_area_cw - each_out_line_channel * width;

  for (k = 0; k < height; k++) {
    for (j = 0; j < width; j++) {
      for (i = 0; i < image_num; i++) {
        align_each_in_area_cw =
            align_to_x(channel_num[i] * width, IMAGE_ALIGNMENT);
        memcpy((int16_t *)image_out + tmp_channel +
                   k * align_each_out_area_cw_differ,
               images_in[i] + j * channel_num[i] + k * align_each_in_area_cw,
               channel_num[i] * sizeof(int16_t));

        tmp_channel += channel_num[i];
      }
    }
  }

  fpga_flush(image_out, height * align_each_out_area_cw * sizeof(int16_t));
}

}  // namespace image
}  // namespace fpga
}  // namespace paddle_mobile
