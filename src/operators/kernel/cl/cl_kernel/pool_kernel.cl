#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#define MIN_VALUE -FLT_MAX

__kernel void pool_max(
    __private const int in_height, __private const int in_width,
    __private const int out_height, __private const int out_width,
    __private const int pad_top, __private const int pad_left,
    __private const int stride_h, __private const int stride_w,
    __private const int ksize_h, __private const int ksize_w,
    __read_only image2d_t input, __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int start_h = max(out_h * stride_h - pad_top, 0);
  int end_h = min(start_h + ksize_h, in_height);

  int start_w = max(out_w * stride_w - pad_left, 0);
  int end_w = min(start_w + ksize_w, in_width);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  half4 max_value = (half4)(MIN_VALUE);
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      half4 tmp = read_imageh(input, sampler, (int2)(pos_in_x + x, pos_in_y + y));
      max_value = max(max_value, tmp);
    }
  }

  const int pos_out_x = mad24(out_c, out_width, out_w);
  write_imageh(output, (int2)(pos_out_x, out_nh), max_value);
}

__kernel void pool_avg(
    __private const int in_height, __private const int in_width,
    __private const int out_height, __private const int out_width,
    __private const int pad_top, __private const int pad_left,
    __private const int stride_h, __private const int stride_w,
    __private const int ksize_h, __private const int ksize_w,
    __read_only image2d_t input, __write_only image2d_t output) {
  const int out_c = get_global_id(0);
  const int out_w = get_global_id(1);
  const int out_nh = get_global_id(2);
  const int out_n = out_nh / out_height;
  const int out_h = out_nh % out_height;

  const sampler_t sampler =
      CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

  int start_h = max(out_h * stride_h - pad_top, 0);
  int end_h = min(start_h + ksize_h, in_height);

  int start_w = max(out_w * stride_w - pad_left, 0);
  int end_w = min(start_w + ksize_w, in_width);

  const int pos_in_x = out_c * in_width;
  const int pos_in_y = out_n * in_height;
  half4 sum = (half4)(0.0f);
  int num = 0;
  for (int y = start_h; y < end_h; ++y) {
    for (int x = start_w; x < end_w; ++x) {
      sum += read_imageh(input, sampler, (int2)(pos_in_x + x, pos_in_y + y));
      num++;
    }
  }
  half4 avg = sum / num;
  const int pos_out_x = mad24(out_c, out_width, out_w);
  write_imageh(output, (int2)(pos_out_x, out_nh), avg);
}