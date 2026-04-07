#include <ATen/native/mps/kernels/GridSampler.h>
#include <ATen/native/mps/kernels/SamplingHelpers.h>
#include <c10/metal/utils.h>
#include <metal_array>
#include <metal_stdlib>

using namespace metal;
using namespace c10::metal;

struct GridSamplerOffsets {
  int32_t output;
  int32_t input;
  int32_t grid;

  GridSamplerOffsets() : output(0), input(0), grid(0) {}
};

// Find offsets into the tensors that this thread will operate on,
// based on the thread ID.
static GridSamplerOffsets find_grid_sampler_offsets(
    constant int32_t* output_sizes,
    constant int32_t* output_strides,
    constant int32_t* input_strides,
    constant int32_t* grid_strides,
    int32_t sampler_dims,
    uint tid) {
  auto dims = sampler_dims + 2;
  auto output_idx = static_cast<int32_t>(tid);
  GridSamplerOffsets offsets;

  for (auto dim = dims - 1; dim >= 0; dim--) {
    auto dim_idx = output_idx % output_sizes[dim];
    output_idx = output_idx / output_sizes[dim];

    // Select the output element that this thread will calculate.
    // output shape:
    //   2 sampler dims: (N, C, Hout, Wout)
    //   3 sampler dims: (N, C, Dout, Hout, Wout)
    offsets.output += output_strides[dim] * dim_idx;

    // Select the batch and channel for the input.
    // input shape:
    //   2 sampler dims: (N, C, Hin, Win)
    //   3 sampler dims: (N, C, Din, Hin, Win)
    if (dim < 2) {
      offsets.input += input_strides[dim] * dim_idx;
    }

    // Select the grid coordinates for the output element.
    // grid shape:
    //   2 sampler dims: (N, Hout, Wout, 2)
    //   3 sampler dims: (N, Dout, Hout, Wout, 3)
    if (dim == 0) {
      offsets.grid += grid_strides[dim] * dim_idx;
    } else if (dim >= 2) {
      offsets.grid += grid_strides[dim - 1] * dim_idx;
    }
  }

  return offsets;
}

// Mod function which gives positive output when `a` is negative
static int32_t mod(int32_t a, int32_t b) {
  auto r = a % b;
  return r + (r < 0 ? b : 0);
}

// Sentinel index value to indicate zero padding
constant int32_t IDX_ZERO = -1;

// Unnormalize grid coordinate from [-1, 1] to pixel space
static float grid_sampler_unnormalize(
    float coord,
    int32_t size,
    bool align_corners) {
  if (align_corners) {
    return ((coord + 1) / 2) * (size - 1);
  } else {
    return ((coord + 1) * size - 1) / 2;
  }
}

// Clip coordinates for border padding
static float clip_coordinates(float in, int32_t clip_limit) {
  return ::metal::clamp(in, 0.0, clip_limit - 1.0);
}

// Reflect coordinates for reflection padding
template <typename T>
static T reflect_coordinates(T in, int32_t twice_low, int32_t twice_high) {
  if (twice_low == twice_high) {
    return 0;
  }
  auto min_val = static_cast<T>(twice_low) / 2;
  auto span = static_cast<T>(twice_high - twice_low) / 2;
  in = fabs(in - min_val);
  auto extra = fmod(in, span);
  int32_t flips = static_cast<int32_t>(floor(in / span));
  return (flips % 2 == 0) ? (extra + min_val) : (span - extra + min_val);
}

// Padding functors: each encapsulates the padding logic for integer indices
// (pad) and float source coordinates (compute_source).
struct PadZeros {
  static constant constexpr bool checks_bounds = true;

  static int32_t pad(int32_t idx, int32_t input_size, bool) {
    return (idx < 0 || idx >= input_size) ? IDX_ZERO : idx;
  }

  static float compute_source(float coord, int32_t size, bool align_corners) {
    return grid_sampler_unnormalize(coord, size, align_corners);
  }
};

struct PadBorder {
  static constant constexpr bool checks_bounds = false;

  static int32_t pad(int32_t idx, int32_t input_size, bool) {
    return clamp(idx, 0, input_size - 1);
  }

  static float compute_source(float coord, int32_t size, bool align_corners) {
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    return clip_coordinates(coord, size);
  }
};

struct PadReflection {
  static constant constexpr bool checks_bounds = false;

  static int32_t pad(int32_t idx, int32_t input_size, bool align_corners) {
    auto scale_length = align_corners ? (input_size - 1) : input_size;
    auto idx_mod = mod(idx, scale_length);
    auto idx_mod_reverse = (input_size - 1) - idx_mod;
    bool is_reverse = (abs(idx - idx_mod) / scale_length) % 2 == 1;
    return is_reverse ? idx_mod_reverse : idx_mod;
  }

  static float compute_source(float coord, int32_t size, bool align_corners) {
    coord = grid_sampler_unnormalize(coord, size, align_corners);
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    return clip_coordinates(coord, size);
  }
};

// 2D Bilinear interpolation
template <typename Pad, typename T>
static T interpolate_bilinear_2d(
    constant T* input,
    float ix,
    float iy,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = grid_sampler_unnormalize(ix, inp_W, align_corners);
  iy = grid_sampler_unnormalize(iy, inp_H, align_corners);

  int32_t ix_nw = static_cast<int32_t>(floor(ix));
  int32_t iy_nw = static_cast<int32_t>(floor(iy));
  int32_t ix_ne = ix_nw + 1;
  int32_t iy_ne = iy_nw;
  int32_t ix_sw = ix_nw;
  int32_t iy_sw = iy_nw + 1;
  int32_t ix_se = ix_nw + 1;
  int32_t iy_se = iy_nw + 1;

  const auto nw = (ix_se - ix) * (iy_se - iy);
  const auto ne = (ix - ix_sw) * (iy_sw - iy);
  const auto sw = (ix_ne - ix) * (iy - iy_ne);
  const auto se = (ix - ix_nw) * (iy - iy_nw);

  int32_t iy_nw_p = Pad::pad(iy_nw, inp_H, align_corners);
  int32_t ix_nw_p = Pad::pad(ix_nw, inp_W, align_corners);
  int32_t iy_ne_p = Pad::pad(iy_ne, inp_H, align_corners);
  int32_t ix_ne_p = Pad::pad(ix_ne, inp_W, align_corners);
  int32_t iy_sw_p = Pad::pad(iy_sw, inp_H, align_corners);
  int32_t ix_sw_p = Pad::pad(ix_sw, inp_W, align_corners);
  int32_t iy_se_p = Pad::pad(iy_se, inp_H, align_corners);
  int32_t ix_se_p = Pad::pad(ix_se, inp_W, align_corners);

  opmath_t<T> out_acc = 0;
  if (iy_nw_p != IDX_ZERO && ix_nw_p != IDX_ZERO) {
    out_acc += input[iy_nw_p * inp_sH + ix_nw_p * inp_sW] * nw;
  }
  if (iy_ne_p != IDX_ZERO && ix_ne_p != IDX_ZERO) {
    out_acc += input[iy_ne_p * inp_sH + ix_ne_p * inp_sW] * ne;
  }
  if (iy_sw_p != IDX_ZERO && ix_sw_p != IDX_ZERO) {
    out_acc += input[iy_sw_p * inp_sH + ix_sw_p * inp_sW] * sw;
  }
  if (iy_se_p != IDX_ZERO && ix_se_p != IDX_ZERO) {
    out_acc += input[iy_se_p * inp_sH + ix_se_p * inp_sW] * se;
  }

  return static_cast<T>(out_acc);
}

// 2D Nearest neighbor interpolation
template <typename Pad, typename T>
static T interpolate_nearest_2d(
    constant T* input,
    opmath_t<T> ix,
    opmath_t<T> iy,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = Pad::compute_source(ix, inp_W, align_corners);
  iy = Pad::compute_source(iy, inp_H, align_corners);

  int32_t ix_nearest = static_cast<int32_t>(rint(ix));
  int32_t iy_nearest = static_cast<int32_t>(rint(iy));

  if (Pad::checks_bounds) {
    if (ix_nearest < 0 || ix_nearest >= inp_W || iy_nearest < 0 ||
        iy_nearest >= inp_H) {
      return static_cast<T>(0);
    }
  }

  return input[iy_nearest * inp_sH + ix_nearest * inp_sW];
}

// Helper to get bounded value for bicubic interpolation
template <typename Pad, typename T>
static opmath_t<T> get_bicubic_value(
    constant T* input,
    int32_t y,
    int32_t x,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  int32_t y_p = Pad::pad(y, inp_H, align_corners);
  int32_t x_p = Pad::pad(x, inp_W, align_corners);
  if (y_p == IDX_ZERO || x_p == IDX_ZERO) {
    return 0;
  }
  return input[y_p * inp_sH + x_p * inp_sW];
}

// 2D Bicubic interpolation
template <typename Pad, typename T>
static T interpolate_bicubic_2d(
    constant T* input,
    float ix,
    float iy,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = grid_sampler_unnormalize(ix, inp_W, align_corners);
  iy = grid_sampler_unnormalize(iy, inp_H, align_corners);

  auto ix_nw = floor(ix);
  auto iy_nw = floor(iy);
  auto tx = ix - ix_nw;
  auto ty = iy - iy_nw;

  opmath_t<T> coefficients[4];
  int32_t ix_nw_i = static_cast<int32_t>(ix_nw);
  int32_t iy_nw_i = static_cast<int32_t>(iy_nw);

  for (int32_t i = 0; i < 4; ++i) {
    coefficients[i] = cubic_interp1d(
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i - 1,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i + 0,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i + 1,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        get_bicubic_value<Pad, T>(
            input,
            iy_nw_i - 1 + i,
            ix_nw_i + 2,
            inp_H,
            inp_W,
            inp_sH,
            inp_sW,
            align_corners),
        tx);
  }

  return static_cast<T>(cubic_interp1d(
      coefficients[0], coefficients[1], coefficients[2], coefficients[3], ty));
}

// Interpolation functors for the 2D kernel.
// Each wraps the respective interpolation function with the padding already
// baked in via the Pad template parameter.
template <typename Pad>
struct Bilinear2D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_bilinear_2d<Pad, T>(
        input, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
  }
};

template <typename Pad>
struct Nearest2D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_nearest_2d<Pad, T>(
        input, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
  }
};

template <typename Pad>
struct Bicubic2D {
  template <typename T>
  static T interpolate(
      constant T* input,
      float ix,
      float iy,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_bicubic_2d<Pad, T>(
        input, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
  }
};

// 2D grid sampler kernel
template <typename Interp, typename T>
kernel void grid_sampler_2d(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerParams<4>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto C = params.output_sizes[1];
  auto out_H = params.output_sizes[2];
  auto out_W = params.output_sizes[3];
  auto inp_H = params.input_sizes[2];
  auto inp_W = params.input_sizes[3];

  auto out_sN = params.output_strides[0];
  auto out_sC = params.output_strides[1];
  auto out_sH = params.output_strides[2];
  auto out_sW = params.output_strides[3];
  auto inp_sN = params.input_strides[0];
  auto inp_sC = params.input_strides[1];
  auto inp_sH = params.input_strides[2];
  auto inp_sW = params.input_strides[3];
  auto grid_sN = params.grid_strides[0];
  auto grid_sH = params.grid_strides[1];
  auto grid_sW = params.grid_strides[2];
  auto grid_sCoor = params.grid_strides[3];

  auto align_corners = params.align_corners;

  int32_t w = tid % out_W;
  int32_t h = (tid / out_W) % out_H;
  int32_t n = tid / (out_H * out_W);

  auto grid_ptr = grid + n * grid_sN + h * grid_sH + w * grid_sW;
  opmath_t<T> ix = static_cast<opmath_t<T>>(grid_ptr[0]);
  opmath_t<T> iy = static_cast<opmath_t<T>>(grid_ptr[grid_sCoor]);

  auto inp_ptr_N = input + n * inp_sN;
  auto out_ptr_NCHW = output + n * out_sN + h * out_sH + w * out_sW;

  for (int32_t c = 0; c < C; ++c) {
    auto result = Interp::template interpolate<T>(
        inp_ptr_N, ix, iy, inp_H, inp_W, inp_sH, inp_sW, align_corners);
    out_ptr_NCHW[c * out_sC] = result;
    inp_ptr_N += inp_sC;
  }
}

// 3D trilinear interpolation matching the 2D bilinear pattern.
// Takes pre-read grid coordinates as values, returns the interpolated value.
template <typename Pad, typename T>
static T interpolate_trilinear_3d(
    constant T* input,
    opmath_t<T> ix,
    opmath_t<T> iy,
    opmath_t<T> iz,
    int32_t inp_D,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sD,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = grid_sampler_unnormalize(ix, inp_W, align_corners);
  iy = grid_sampler_unnormalize(iy, inp_H, align_corners);
  iz = grid_sampler_unnormalize(iz, inp_D, align_corners);

  int32_t ix_l = static_cast<int32_t>(floor(ix));
  int32_t iy_l = static_cast<int32_t>(floor(iy));
  int32_t iz_l = static_cast<int32_t>(floor(iz));
  int32_t ix_r = ix_l + 1;
  int32_t iy_r = iy_l + 1;
  int32_t iz_r = iz_l + 1;

  opmath_t<T> sx = ix - ix_l;
  opmath_t<T> sy = iy - iy_l;
  opmath_t<T> sz = iz - iz_l;

  int32_t ix_l_p = Pad::pad(ix_l, inp_W, align_corners);
  int32_t ix_r_p = Pad::pad(ix_r, inp_W, align_corners);
  int32_t iy_l_p = Pad::pad(iy_l, inp_H, align_corners);
  int32_t iy_r_p = Pad::pad(iy_r, inp_H, align_corners);
  int32_t iz_l_p = Pad::pad(iz_l, inp_D, align_corners);
  int32_t iz_r_p = Pad::pad(iz_r, inp_D, align_corners);

  opmath_t<T> out_acc = 0;
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_l_p * inp_sH + ix_l_p * inp_sW] *
        (1 - sz) * (1 - sy) * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_l_p * inp_sH + ix_r_p * inp_sW] *
        (1 - sz) * (1 - sy) * sx;
  }
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_r_p * inp_sH + ix_l_p * inp_sW] *
        (1 - sz) * sy * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_l_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_l_p * inp_sD + iy_r_p * inp_sH + ix_r_p * inp_sW] *
        (1 - sz) * sy * sx;
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_l_p * inp_sH + ix_l_p * inp_sW] * sz *
        (1 - sy) * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_l_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_l_p * inp_sH + ix_r_p * inp_sW] * sz *
        (1 - sy) * sx;
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_l_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_r_p * inp_sH + ix_l_p * inp_sW] * sz *
        sy * (1 - sx);
  }
  if (!Pad::checks_bounds ||
      (iz_r_p != IDX_ZERO && iy_r_p != IDX_ZERO && ix_r_p != IDX_ZERO)) {
    out_acc += input[iz_r_p * inp_sD + iy_r_p * inp_sH + ix_r_p * inp_sW] * sz *
        sy * sx;
  }

  return static_cast<T>(out_acc);
}

// 3D nearest neighbor interpolation matching the 2D nearest pattern.
template <typename Pad, typename T>
static T interpolate_nearest_3d(
    constant T* input,
    opmath_t<T> ix,
    opmath_t<T> iy,
    opmath_t<T> iz,
    int32_t inp_D,
    int32_t inp_H,
    int32_t inp_W,
    int32_t inp_sD,
    int32_t inp_sH,
    int32_t inp_sW,
    bool align_corners) {
  ix = Pad::compute_source(ix, inp_W, align_corners);
  iy = Pad::compute_source(iy, inp_H, align_corners);
  iz = Pad::compute_source(iz, inp_D, align_corners);

  int32_t ix_nearest = static_cast<int32_t>(rint(ix));
  int32_t iy_nearest = static_cast<int32_t>(rint(iy));
  int32_t iz_nearest = static_cast<int32_t>(rint(iz));

  if (Pad::checks_bounds) {
    if (ix_nearest < 0 || ix_nearest >= inp_W || iy_nearest < 0 ||
        iy_nearest >= inp_H || iz_nearest < 0 || iz_nearest >= inp_D) {
      return static_cast<T>(0);
    }
  }

  return input[iz_nearest * inp_sD + iy_nearest * inp_sH + ix_nearest * inp_sW];
}

// Interpolation strategies for 3D kernel (matching 2D pattern).
template <typename Pad>
struct Bilinear3D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      opmath_t<T> iz,
      int32_t inp_D,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sD,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_trilinear_3d<Pad, T>(
        input,
        ix,
        iy,
        iz,
        inp_D,
        inp_H,
        inp_W,
        inp_sD,
        inp_sH,
        inp_sW,
        align_corners);
  }
};

template <typename Pad>
struct Nearest3D {
  template <typename T>
  static T interpolate(
      constant T* input,
      opmath_t<T> ix,
      opmath_t<T> iy,
      opmath_t<T> iz,
      int32_t inp_D,
      int32_t inp_H,
      int32_t inp_W,
      int32_t inp_sD,
      int32_t inp_sH,
      int32_t inp_sW,
      bool align_corners) {
    return interpolate_nearest_3d<Pad, T>(
        input,
        ix,
        iy,
        iz,
        inp_D,
        inp_H,
        inp_W,
        inp_sD,
        inp_sH,
        inp_sW,
        align_corners);
  }
};

// 3D grid sampler kernel: one thread per spatial position (n, d, h, w),
// loops over channels C. Grid coordinates are read once and reused.
template <typename Interp, typename T>
kernel void grid_sampler_3d(
    device T* output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    constant GridSamplerParams<5>& params [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
  auto C = params.output_sizes[1];
  auto out_D = params.output_sizes[2];
  auto out_H = params.output_sizes[3];
  auto out_W = params.output_sizes[4];

  auto out_sN = params.output_strides[0];
  auto out_sC = params.output_strides[1];
  auto out_sD = params.output_strides[2];
  auto out_sH = params.output_strides[3];
  auto out_sW = params.output_strides[4];
  auto inp_sN = params.input_strides[0];
  auto inp_sC = params.input_strides[1];
  auto inp_sD = params.input_strides[2];
  auto inp_sH = params.input_strides[3];
  auto inp_sW = params.input_strides[4];
  auto inp_D = params.input_sizes[2];
  auto inp_H = params.input_sizes[3];
  auto inp_W = params.input_sizes[4];

  auto grid_sN = params.grid_strides[0];
  auto grid_sD = params.grid_strides[1];
  auto grid_sH = params.grid_strides[2];
  auto grid_sW = params.grid_strides[3];
  auto grid_sCoor = params.grid_strides[4];

  auto align_corners = params.align_corners;

  int32_t w = tid % out_W;
  int32_t h = (tid / out_W) % out_H;
  int32_t d = (tid / (out_W * out_H)) % out_D;
  int32_t n = tid / (out_W * out_H * out_D);

  auto grid_ptr = grid + n * grid_sN + d * grid_sD + h * grid_sH + w * grid_sW;
  opmath_t<T> ix = static_cast<opmath_t<T>>(grid_ptr[0]);
  opmath_t<T> iy = static_cast<opmath_t<T>>(grid_ptr[grid_sCoor]);
  opmath_t<T> iz = static_cast<opmath_t<T>>(grid_ptr[2 * grid_sCoor]);

  auto inp_ptr_N = input + n * inp_sN;
  auto out_ptr = output + n * out_sN + d * out_sD + h * out_sH + w * out_sW;

  for (int32_t c = 0; c < C; ++c) {
    auto result = Interp::template interpolate<T>(
        inp_ptr_N,
        ix,
        iy,
        iz,
        inp_D,
        inp_H,
        inp_W,
        inp_sD,
        inp_sH,
        inp_sW,
        align_corners);
    out_ptr[c * out_sC] = result;
    inp_ptr_N += inp_sC;
  }
}

// Padding mode constants (must match GridSamplerPadding enum)
constant int32_t kPaddingZeros = 0;
constant int32_t kPaddingBorder = 1;
constant int32_t kPaddingReflection = 2;

// Original version from working branch - uses pointer for grad_in
// Uses opmath_t<T> for intermediate computations to avoid overflow with
// half/bfloat
template <typename T>
T grid_sampler_compute_source_index_set_grad(
    T coord,
    int32_t size,
    int32_t padding_mode,
    bool align_corners,
    thread T* grad_in) {
  using U = opmath_t<T>;
  U u_coord = static_cast<U>(coord);
  U u_grad_in = static_cast<U>(*grad_in);
  U u_size = static_cast<U>(size);

  // Unnormalize
  if (align_corners) {
    u_coord = ((u_coord + U(1.0)) / U(2.0)) * (u_size - U(1.0));
    u_grad_in = (u_size - U(1.0)) / U(2.0);
  } else {
    u_coord = ((u_coord + U(1.0)) * u_size - U(1.0)) / U(2.0);
    u_grad_in = u_size / U(2.0);
  }

  if (padding_mode == kPaddingBorder) {
    U grad_clip = U(1.0);
    if (u_coord < U(0.0)) {
      u_coord = U(0.0);
      grad_clip = U(0.0);
    } else if (u_coord > u_size - U(1.0)) {
      u_coord = u_size - U(1.0);
      grad_clip = U(0.0);
    }
    u_grad_in = u_grad_in * grad_clip;
  } else if (padding_mode == kPaddingReflection) {
    U grad_refl = U(1.0);
    U twice_low, twice_high;
    if (align_corners) {
      twice_low = U(0.0);
      twice_high = U(2 * (size - 1));
    } else {
      twice_low = U(-1.0);
      twice_high = U(2 * size - 1);
    }

    if (twice_low != twice_high) {
      U min_val = twice_low / U(2.0);
      U span = (twice_high - twice_low) / U(2.0);
      u_coord = u_coord - min_val;

      if (u_coord < U(0.0)) {
        u_coord = -u_coord;
        grad_refl = -grad_refl;
      }

      U extra = u_coord - span * floor(u_coord / span);
      int32_t flips = static_cast<int32_t>(floor(u_coord / span));

      if (flips % 2 == 0) {
        u_coord = extra + min_val;
      } else {
        u_coord = span - extra + min_val;
        grad_refl = -grad_refl;
      }
    } else {
      u_coord = U(0.0);
    }

    // Clip after reflection
    U grad_clip = U(1.0);
    if (u_coord < U(0.0)) {
      u_coord = U(0.0);
      grad_clip = U(0.0);
    } else if (u_coord > u_size - U(1.0)) {
      u_coord = u_size - U(1.0);
      grad_clip = U(0.0);
    }
    u_grad_in = u_grad_in * grad_refl * grad_clip;
  }

  coord = static_cast<T>(u_coord);
  *grad_in = static_cast<T>(u_grad_in);
  return coord;
}

inline bool within_bounds_2d(int32_t y, int32_t x, int32_t H, int32_t W) {
  return y >= 0 && y < H && x >= 0 && x < W;
}

inline bool within_bounds_3d(
    int32_t z,
    int32_t y,
    int32_t x,
    int32_t D,
    int32_t H,
    int32_t W) {
  return z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W;
}

template <typename T>
static void get_cubic_coefficients_grad(T coeffs[4], T t) {
  T A = static_cast<T>(-0.75);
  T x;
  x = -1 - t;
  coeffs[0] = (-3 * A * x - 10 * A) * x - 8 * A;
  x = -t;
  coeffs[1] = (-3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 1 - t;
  coeffs[2] = (3 * (A + 2) * x - 2 * (A + 3)) * x;
  x = 2 - t;
  coeffs[3] = (3 * A * x - 10 * A) * x + 8 * A;
}

template <typename T>
static T grid_sampler_unnormalize_set_grad(
    T coord,
    int32_t size,
    bool align_corners,
    thread T* grad_in) {
  if (align_corners) {
    *grad_in = static_cast<T>(size - 1) / T(2);
    return ((coord + T(1)) / T(2)) * (size - 1);
  } else {
    *grad_in = static_cast<T>(size) / T(2);
    return ((coord + T(1)) * size - T(1)) / T(2);
  }
}

static float compute_coordinates(
    float coord,
    int32_t size,
    int32_t padding_mode,
    bool align_corners) {
  if (padding_mode == kPaddingBorder) {
    coord = clip_coordinates(coord, size);
  } else if (padding_mode == kPaddingReflection) {
    if (align_corners) {
      coord = reflect_coordinates(coord, 0, 2 * (size - 1));
    } else {
      coord = reflect_coordinates(coord, -1, 2 * size - 1);
    }
    coord = clip_coordinates(coord, size);
  }
  return coord;
}

template <typename T>
static opmath_t<T> get_value_bounded_2d(
    constant T* data,
    float x,
    float y,
    int32_t W,
    int32_t H,
    ulong sW,
    ulong sH,
    int32_t padding_mode,
    bool align_corners) {
  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);
  int32_t ix = static_cast<int32_t>(x);
  int32_t iy = static_cast<int32_t>(y);
  if (within_bounds_2d(iy, ix, H, W)) {
    return static_cast<opmath_t<T>>(data[iy * sH + ix * sW]);
  }
  return static_cast<opmath_t<T>>(0);
}

static void add_value_bounded_2d(
    device atomic<float>* data,
    float x,
    float y,
    int32_t W,
    int32_t H,
    ulong sW,
    ulong sH,
    float delta,
    int32_t padding_mode,
    bool align_corners,
    ulong NC_offset) {
  x = compute_coordinates(x, W, padding_mode, align_corners);
  y = compute_coordinates(y, H, padding_mode, align_corners);
  int32_t ix = static_cast<int32_t>(x);
  int32_t iy = static_cast<int32_t>(y);
  if (within_bounds_2d(iy, ix, H, W)) {
    atomic_fetch_add_explicit(
        &data[NC_offset + iy * sH + ix * sW], delta, memory_order_relaxed);
  }
}

// 3D backward kernel for grad_input - matches original working version exactly
template <typename T>
kernel void grid_sampler_3d_backward_input(
    constant T* grad_output [[buffer(0)]],
    constant T* grid [[buffer(1)]],
    device atomic<float>* grad_input [[buffer(2)]],
    constant int& interpolation_mode [[buffer(3)]],
    constant int& padding_mode [[buffer(4)]],
    constant bool& align_corners [[buffer(5)]],
    constant ulong* input_sizes [[buffer(6)]],
    constant ulong* output_sizes [[buffer(7)]],
    constant ulong* grad_input_strides [[buffer(8)]],
    constant ulong* grid_strides [[buffer(9)]],
    constant ulong* grad_output_strides [[buffer(10)]],
    uint3 thread_index [[thread_position_in_grid]]) {
  const auto out_w = thread_index.x;
  const auto out_d_h_combined = thread_index.y;
  const auto n = thread_index.z;

  const auto out_d = out_d_h_combined / output_sizes[3];
  const auto out_h = out_d_h_combined % output_sizes[3];

  if (n >= input_sizes[0] || out_d >= output_sizes[2] ||
      out_h >= output_sizes[3] || out_w >= output_sizes[4]) {
    return;
  }

  const auto C = input_sizes[1];
  const auto inp_D = input_sizes[2];
  const auto inp_H = input_sizes[3];
  const auto inp_W = input_sizes[4];

  const auto grid_offset = n * grid_strides[0] + out_d * grid_strides[1] +
      out_h * grid_strides[2] + out_w * grid_strides[3];

  const float grid_x = grid[grid_offset];
  const float grid_y = grid[grid_offset + grid_strides[4]];
  const float grid_z = grid[grid_offset + 2 * grid_strides[4]];

  float gix_mult, giy_mult, giz_mult;
  float ix = grid_sampler_compute_source_index_set_grad(
      grid_x,
      static_cast<int32_t>(inp_W),
      padding_mode,
      align_corners,
      &gix_mult);
  float iy = grid_sampler_compute_source_index_set_grad(
      grid_y,
      static_cast<int32_t>(inp_H),
      padding_mode,
      align_corners,
      &giy_mult);
  float iz = grid_sampler_compute_source_index_set_grad(
      grid_z,
      static_cast<int32_t>(inp_D),
      padding_mode,
      align_corners,
      &giz_mult);

  if (interpolation_mode == 0) { // trilinear
    const int ix_tnw = static_cast<int>(floor(ix));
    const int iy_tnw = static_cast<int>(floor(iy));
    const int iz_tnw = static_cast<int>(floor(iz));

    const int ix_tne = ix_tnw + 1;
    const int iy_tne = iy_tnw;
    const int iz_tne = iz_tnw;
    const int ix_tsw = ix_tnw;
    const int iy_tsw = iy_tnw + 1;
    const int iz_tsw = iz_tnw;
    const int ix_tse = ix_tnw + 1;
    const int iy_tse = iy_tnw + 1;
    const int iz_tse = iz_tnw;
    const int ix_bnw = ix_tnw;
    const int iy_bnw = iy_tnw;
    const int iz_bnw = iz_tnw + 1;
    const int ix_bne = ix_tnw + 1;
    const int iy_bne = iy_tnw;
    const int iz_bne = iz_tnw + 1;
    const int ix_bsw = ix_tnw;
    const int iy_bsw = iy_tnw + 1;
    const int iz_bsw = iz_tnw + 1;
    const int ix_bse = ix_tnw + 1;
    const int iy_bse = iy_tnw + 1;
    const int iz_bse = iz_tnw + 1;

    const float tnw = (ix_bse - ix) * (iy_bse - iy) * (iz_bse - iz);
    const float tne = (ix - ix_bsw) * (iy_bsw - iy) * (iz_bsw - iz);
    const float tsw = (ix_bne - ix) * (iy - iy_bne) * (iz_bne - iz);
    const float tse = (ix - ix_bnw) * (iy - iy_bnw) * (iz_bnw - iz);
    const float bnw = (ix_tse - ix) * (iy_tse - iy) * (iz - iz_tse);
    const float bne = (ix - ix_tsw) * (iy_tsw - iy) * (iz - iz_tsw);
    const float bsw = (ix_tne - ix) * (iy - iy_tne) * (iz - iz_tne);
    const float bse = (ix - ix_tnw) * (iy - iy_tnw) * (iz - iz_tnw);

    for (ulong c = 0; c < C; c++) {
      const auto grad_out_offset = n * grad_output_strides[0] +
          c * grad_output_strides[1] + out_d * grad_output_strides[2] +
          out_h * grad_output_strides[3] + out_w * grad_output_strides[4];
      const float gOut = grad_output[grad_out_offset];
      const auto base_grad_input_offset =
          n * grad_input_strides[0] + c * grad_input_strides[1];

      if (within_bounds_3d(
              iz_tnw,
              iy_tnw,
              ix_tnw,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_tnw * grad_input_strides[2] +
                 iy_tnw * grad_input_strides[3] +
                 ix_tnw * grad_input_strides[4]],
            tnw * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_3d(
              iz_tne,
              iy_tne,
              ix_tne,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_tne * grad_input_strides[2] +
                 iy_tne * grad_input_strides[3] +
                 ix_tne * grad_input_strides[4]],
            tne * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_3d(
              iz_tsw,
              iy_tsw,
              ix_tsw,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_tsw * grad_input_strides[2] +
                 iy_tsw * grad_input_strides[3] +
                 ix_tsw * grad_input_strides[4]],
            tsw * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_3d(
              iz_tse,
              iy_tse,
              ix_tse,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_tse * grad_input_strides[2] +
                 iy_tse * grad_input_strides[3] +
                 ix_tse * grad_input_strides[4]],
            tse * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_3d(
              iz_bnw,
              iy_bnw,
              ix_bnw,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_bnw * grad_input_strides[2] +
                 iy_bnw * grad_input_strides[3] +
                 ix_bnw * grad_input_strides[4]],
            bnw * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_3d(
              iz_bne,
              iy_bne,
              ix_bne,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_bne * grad_input_strides[2] +
                 iy_bne * grad_input_strides[3] +
                 ix_bne * grad_input_strides[4]],
            bne * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_3d(
              iz_bsw,
              iy_bsw,
              ix_bsw,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_bsw * grad_input_strides[2] +
                 iy_bsw * grad_input_strides[3] +
                 ix_bsw * grad_input_strides[4]],
            bsw * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_3d(
              iz_bse,
              iy_bse,
              ix_bse,
              static_cast<int32_t>(inp_D),
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base_grad_input_offset + iz_bse * grad_input_strides[2] +
                 iy_bse * grad_input_strides[3] +
                 ix_bse * grad_input_strides[4]],
            bse * gOut,
            memory_order_relaxed);
      }
    }
  } else { // nearest
    int32_t ix_n = static_cast<int32_t>(rint(ix));
    int32_t iy_n = static_cast<int32_t>(rint(iy));
    int32_t iz_n = static_cast<int32_t>(rint(iz));

    if (padding_mode == kPaddingBorder) {
      ix_n = clamp(ix_n, 0, static_cast<int32_t>(inp_W - 1));
      iy_n = clamp(iy_n, 0, static_cast<int32_t>(inp_H - 1));
      iz_n = clamp(iz_n, 0, static_cast<int32_t>(inp_D - 1));
    } else if (padding_mode == kPaddingReflection) {
      if (align_corners) {
        ix_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(ix_n), 0, 2 * (inp_W - 1))));
        iy_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(iy_n), 0, 2 * (inp_H - 1))));
        iz_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(iz_n), 0, 2 * (inp_D - 1))));
      } else {
        ix_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(ix_n), -1, 2 * inp_W - 1)));
        iy_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(iy_n), -1, 2 * inp_H - 1)));
        iz_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(iz_n), -1, 2 * inp_D - 1)));
      }
      ix_n = clamp(ix_n, 0, static_cast<int32_t>(inp_W - 1));
      iy_n = clamp(iy_n, 0, static_cast<int32_t>(inp_H - 1));
      iz_n = clamp(iz_n, 0, static_cast<int32_t>(inp_D - 1));
    }

    bool in_bounds = padding_mode != kPaddingZeros ||
        within_bounds_3d(iz_n,
                         iy_n,
                         ix_n,
                         static_cast<int32_t>(inp_D),
                         static_cast<int32_t>(inp_H),
                         static_cast<int32_t>(inp_W));

    if (in_bounds) {
      const auto base_offset = n * grad_input_strides[0] +
          iz_n * grad_input_strides[2] + iy_n * grad_input_strides[3] +
          ix_n * grad_input_strides[4];

      for (ulong c = 0; c < C; c++) {
        const auto grad_out_offset = n * grad_output_strides[0] +
            c * grad_output_strides[1] + out_d * grad_output_strides[2] +
            out_h * grad_output_strides[3] + out_w * grad_output_strides[4];
        const float gOut = grad_output[grad_out_offset];
        atomic_fetch_add_explicit(
            &grad_input[base_offset + c * grad_input_strides[1]],
            gOut,
            memory_order_relaxed);
      }
    }
  }
}

// 3D backward kernel for grad_grid - matches original working version
template <typename T>
kernel void grid_sampler_3d_backward_grid(
    constant T* grad_output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    device T* grad_grid [[buffer(3)]],
    constant int& interpolation_mode [[buffer(4)]],
    constant int& padding_mode [[buffer(5)]],
    constant bool& align_corners [[buffer(6)]],
    constant ulong* input_sizes [[buffer(7)]],
    constant ulong* output_sizes [[buffer(8)]],
    constant ulong* input_strides [[buffer(9)]],
    constant ulong* grad_grid_strides [[buffer(10)]],
    constant ulong* grid_strides [[buffer(11)]],
    constant ulong* grad_output_strides [[buffer(12)]],
    uint3 thread_index [[thread_position_in_grid]]) {
  const auto out_w = thread_index.x;
  const auto out_d_h_combined = thread_index.y;
  const auto n = thread_index.z;

  const auto out_d = out_d_h_combined / output_sizes[3];
  const auto out_h = out_d_h_combined % output_sizes[3];

  if (n >= input_sizes[0] || out_d >= output_sizes[2] ||
      out_h >= output_sizes[3] || out_w >= output_sizes[4]) {
    return;
  }

  const auto C = input_sizes[1];
  const auto inp_D = input_sizes[2];
  const auto inp_H = input_sizes[3];
  const auto inp_W = input_sizes[4];

  const auto grid_offset = n * grid_strides[0] + out_d * grid_strides[1] +
      out_h * grid_strides[2] + out_w * grid_strides[3];

  const opmath_t<T> grid_x = grid[grid_offset];
  const opmath_t<T> grid_y = grid[grid_offset + grid_strides[4]];
  const opmath_t<T> grid_z = grid[grid_offset + 2 * grid_strides[4]];

  opmath_t<T> gix_mult, giy_mult, giz_mult;
  opmath_t<T> ix = grid_sampler_compute_source_index_set_grad(
      grid_x,
      static_cast<int32_t>(inp_W),
      padding_mode,
      align_corners,
      &gix_mult);
  opmath_t<T> iy = grid_sampler_compute_source_index_set_grad(
      grid_y,
      static_cast<int32_t>(inp_H),
      padding_mode,
      align_corners,
      &giy_mult);
  opmath_t<T> iz = grid_sampler_compute_source_index_set_grad(
      grid_z,
      static_cast<int32_t>(inp_D),
      padding_mode,
      align_corners,
      &giz_mult);

  const int32_t ix_tnw = static_cast<int32_t>(floor(ix));
  const int32_t iy_tnw = static_cast<int32_t>(floor(iy));
  const int32_t iz_tnw = static_cast<int32_t>(floor(iz));

  const int32_t ix_tne = ix_tnw + 1;
  const int32_t iy_tne = iy_tnw;
  const int32_t iz_tne = iz_tnw;
  const int32_t ix_tsw = ix_tnw;
  const int32_t iy_tsw = iy_tnw + 1;
  const int32_t iz_tsw = iz_tnw;
  const int32_t ix_tse = ix_tnw + 1;
  const int32_t iy_tse = iy_tnw + 1;
  const int32_t iz_tse = iz_tnw;
  const int32_t ix_bnw = ix_tnw;
  const int32_t iy_bnw = iy_tnw;
  const int32_t iz_bnw = iz_tnw + 1;
  const int32_t ix_bne = ix_tnw + 1;
  const int32_t iy_bne = iy_tnw;
  const int32_t iz_bne = iz_tnw + 1;
  const int32_t ix_bsw = ix_tnw;
  const int32_t iy_bsw = iy_tnw + 1;
  const int32_t iz_bsw = iz_tnw + 1;
  const int32_t ix_bse = ix_tnw + 1;
  const int32_t iy_bse = iy_tnw + 1;
  const int32_t iz_bse = iz_tnw + 1;

  const auto grad_grid_base_offset = n * grad_grid_strides[0] +
      out_d * grad_grid_strides[1] + out_h * grad_grid_strides[2] +
      out_w * grad_grid_strides[3];

  opmath_t<T> gix = 0, giy = 0, giz = 0;

  for (ulong c = 0; c < C; c++) {
    const auto grad_out_offset = n * grad_output_strides[0] +
        c * grad_output_strides[1] + out_d * grad_output_strides[2] +
        out_h * grad_output_strides[3] + out_w * grad_output_strides[4];
    const opmath_t<T> gOut = grad_output[grad_out_offset];

    const auto input_base_offset = n * input_strides[0] + c * input_strides[1];

    if (within_bounds_3d(
            static_cast<int32_t>(iz_tnw),
            static_cast<int32_t>(iy_tnw),
            static_cast<int32_t>(ix_tnw),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> tnw_val = input
          [input_base_offset + iz_tnw * input_strides[2] +
           iy_tnw * input_strides[3] + ix_tnw * input_strides[4]];
      gix -= tnw_val * (iy_bse - iy) * (iz_bse - iz) * gOut;
      giy -= tnw_val * (ix_bse - ix) * (iz_bse - iz) * gOut;
      giz -= tnw_val * (ix_bse - ix) * (iy_bse - iy) * gOut;
    }
    if (within_bounds_3d(
            static_cast<int32_t>(iz_tne),
            static_cast<int32_t>(iy_tne),
            static_cast<int32_t>(ix_tne),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> tne_val = input
          [input_base_offset + iz_tne * input_strides[2] +
           iy_tne * input_strides[3] + ix_tne * input_strides[4]];
      gix += tne_val * (iy_bsw - iy) * (iz_bsw - iz) * gOut;
      giy -= tne_val * (ix - ix_bsw) * (iz_bsw - iz) * gOut;
      giz -= tne_val * (ix - ix_bsw) * (iy_bsw - iy) * gOut;
    }
    if (within_bounds_3d(
            static_cast<int32_t>(iz_tsw),
            static_cast<int32_t>(iy_tsw),
            static_cast<int32_t>(ix_tsw),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> tsw_val = input
          [input_base_offset + iz_tsw * input_strides[2] +
           iy_tsw * input_strides[3] + ix_tsw * input_strides[4]];
      gix -= tsw_val * (iy - iy_bne) * (iz_bne - iz) * gOut;
      giy += tsw_val * (ix_bne - ix) * (iz_bne - iz) * gOut;
      giz -= tsw_val * (ix_bne - ix) * (iy - iy_bne) * gOut;
    }
    if (within_bounds_3d(
            static_cast<int32_t>(iz_tse),
            static_cast<int32_t>(iy_tse),
            static_cast<int32_t>(ix_tse),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> tse_val = input
          [input_base_offset + iz_tse * input_strides[2] +
           iy_tse * input_strides[3] + ix_tse * input_strides[4]];
      gix += tse_val * (iy - iy_bnw) * (iz_bnw - iz) * gOut;
      giy += tse_val * (ix - ix_bnw) * (iz_bnw - iz) * gOut;
      giz -= tse_val * (ix - ix_bnw) * (iy - iy_bnw) * gOut;
    }
    if (within_bounds_3d(
            static_cast<int32_t>(iz_bnw),
            static_cast<int32_t>(iy_bnw),
            static_cast<int32_t>(ix_bnw),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> bnw_val = input
          [input_base_offset + iz_bnw * input_strides[2] +
           iy_bnw * input_strides[3] + ix_bnw * input_strides[4]];
      gix -= bnw_val * (iy_tse - iy) * (iz - iz_tse) * gOut;
      giy -= bnw_val * (ix_tse - ix) * (iz - iz_tse) * gOut;
      giz += bnw_val * (ix_tse - ix) * (iy_tse - iy) * gOut;
    }
    if (within_bounds_3d(
            static_cast<int32_t>(iz_bne),
            static_cast<int32_t>(iy_bne),
            static_cast<int32_t>(ix_bne),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> bne_val = input
          [input_base_offset + iz_bne * input_strides[2] +
           iy_bne * input_strides[3] + ix_bne * input_strides[4]];
      gix += bne_val * (iy_tsw - iy) * (iz - iz_tsw) * gOut;
      giy -= bne_val * (ix - ix_tsw) * (iz - iz_tsw) * gOut;
      giz += bne_val * (ix - ix_tsw) * (iy_tsw - iy) * gOut;
    }
    if (within_bounds_3d(
            static_cast<int32_t>(iz_bsw),
            static_cast<int32_t>(iy_bsw),
            static_cast<int32_t>(ix_bsw),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> bsw_val = input
          [input_base_offset + iz_bsw * input_strides[2] +
           iy_bsw * input_strides[3] + ix_bsw * input_strides[4]];
      gix -= bsw_val * (iy - iy_tne) * (iz - iz_tne) * gOut;
      giy += bsw_val * (ix_tne - ix) * (iz - iz_tne) * gOut;
      giz += bsw_val * (ix_tne - ix) * (iy - iy_tne) * gOut;
    }
    if (within_bounds_3d(
            static_cast<int32_t>(iz_bse),
            static_cast<int32_t>(iy_bse),
            static_cast<int32_t>(ix_bse),
            static_cast<int32_t>(inp_D),
            static_cast<int32_t>(inp_H),
            static_cast<int32_t>(inp_W))) {
      const opmath_t<T> bse_val = input
          [input_base_offset + iz_bse * input_strides[2] +
           iy_bse * input_strides[3] + ix_bse * input_strides[4]];
      gix += bse_val * (iy - iy_tnw) * (iz - iz_tnw) * gOut;
      giy += bse_val * (ix - ix_tnw) * (iz - iz_tnw) * gOut;
      giz += bse_val * (ix - ix_tnw) * (iy - iy_tnw) * gOut;
    }
  }

  grad_grid[grad_grid_base_offset] = static_cast<T>(gix_mult * gix);
  grad_grid[grad_grid_base_offset + grid_strides[4]] =
      static_cast<T>(giy_mult * giy);
  grad_grid[grad_grid_base_offset + 2 * grid_strides[4]] =
      static_cast<T>(giz_mult * giz);
}

// 2D backward kernel for grad_input
template <typename T>
kernel void grid_sampler_2d_backward_input(
    constant T* grad_output [[buffer(0)]],
    constant T* grid [[buffer(1)]],
    device atomic<float>* grad_input [[buffer(2)]],
    constant int& interpolation_mode [[buffer(3)]],
    constant int& padding_mode [[buffer(4)]],
    constant bool& align_corners [[buffer(5)]],
    constant ulong* input_sizes [[buffer(6)]],
    constant ulong* output_sizes [[buffer(7)]],
    constant ulong* grad_input_strides [[buffer(8)]],
    constant ulong* grid_strides [[buffer(9)]],
    constant ulong* grad_output_strides [[buffer(10)]],
    uint3 thread_index [[thread_position_in_grid]]) {
  const auto out_w = thread_index.x;
  const auto out_h = thread_index.y;
  const auto n = thread_index.z;

  if (n >= input_sizes[0] || out_h >= output_sizes[2] ||
      out_w >= output_sizes[3]) {
    return;
  }

  const auto C = input_sizes[1];
  const auto inp_H = input_sizes[2];
  const auto inp_W = input_sizes[3];

  const auto grid_offset =
      n * grid_strides[0] + out_h * grid_strides[1] + out_w * grid_strides[2];
  const float grid_x = grid[grid_offset];
  const float grid_y = grid[grid_offset + grid_strides[3]];

  if (interpolation_mode == 0) { // bilinear
    float gix_mult, giy_mult;
    float ix = grid_sampler_compute_source_index_set_grad(
        grid_x,
        static_cast<int32_t>(inp_W),
        padding_mode,
        align_corners,
        &gix_mult);
    float iy = grid_sampler_compute_source_index_set_grad(
        grid_y,
        static_cast<int32_t>(inp_H),
        padding_mode,
        align_corners,
        &giy_mult);

    const int ix_nw = static_cast<int>(floor(ix));
    const int iy_nw = static_cast<int>(floor(iy));
    const int ix_ne = ix_nw + 1;
    const int iy_ne = iy_nw;
    const int ix_sw = ix_nw;
    const int iy_sw = iy_nw + 1;
    const int ix_se = ix_nw + 1;
    const int iy_se = iy_nw + 1;

    const float nw = (ix_se - ix) * (iy_se - iy);
    const float ne = (ix - ix_sw) * (iy_sw - iy);
    const float sw = (ix_ne - ix) * (iy - iy_ne);
    const float se = (ix - ix_nw) * (iy - iy_nw);

    for (ulong c = 0; c < C; c++) {
      const auto grad_out_offset = n * grad_output_strides[0] +
          c * grad_output_strides[1] + out_h * grad_output_strides[2] +
          out_w * grad_output_strides[3];
      const float gOut = grad_output[grad_out_offset];
      const auto base = n * grad_input_strides[0] + c * grad_input_strides[1];

      if (within_bounds_2d(
              iy_nw,
              ix_nw,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base + iy_nw * grad_input_strides[2] +
                 ix_nw * grad_input_strides[3]],
            nw * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_2d(
              iy_ne,
              ix_ne,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base + iy_ne * grad_input_strides[2] +
                 ix_ne * grad_input_strides[3]],
            ne * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_2d(
              iy_sw,
              ix_sw,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base + iy_sw * grad_input_strides[2] +
                 ix_sw * grad_input_strides[3]],
            sw * gOut,
            memory_order_relaxed);
      }
      if (within_bounds_2d(
              iy_se,
              ix_se,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        atomic_fetch_add_explicit(
            &grad_input
                [base + iy_se * grad_input_strides[2] +
                 ix_se * grad_input_strides[3]],
            se * gOut,
            memory_order_relaxed);
      }
    }
  } else if (interpolation_mode == 1) { // nearest
    float gix_mult, giy_mult;
    float ix = grid_sampler_compute_source_index_set_grad(
        grid_x,
        static_cast<int32_t>(inp_W),
        padding_mode,
        align_corners,
        &gix_mult);
    float iy = grid_sampler_compute_source_index_set_grad(
        grid_y,
        static_cast<int32_t>(inp_H),
        padding_mode,
        align_corners,
        &giy_mult);

    int32_t ix_n = static_cast<int32_t>(rint(ix));
    int32_t iy_n = static_cast<int32_t>(rint(iy));

    if (padding_mode == kPaddingBorder) {
      ix_n = clamp(ix_n, 0, static_cast<int32_t>(inp_W - 1));
      iy_n = clamp(iy_n, 0, static_cast<int32_t>(inp_H - 1));
    } else if (padding_mode == kPaddingReflection) {
      if (align_corners) {
        ix_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(ix_n), 0, 2 * (inp_W - 1))));
        iy_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(iy_n), 0, 2 * (inp_H - 1))));
      } else {
        ix_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(ix_n), -1, 2 * inp_W - 1)));
        iy_n = static_cast<int32_t>(rint(
            reflect_coordinates(static_cast<float>(iy_n), -1, 2 * inp_H - 1)));
      }
      ix_n = clamp(ix_n, 0, static_cast<int32_t>(inp_W - 1));
      iy_n = clamp(iy_n, 0, static_cast<int32_t>(inp_H - 1));
    }

    bool in_bounds = padding_mode != kPaddingZeros ||
        within_bounds_2d(iy_n,
                         ix_n,
                         static_cast<int32_t>(inp_H),
                         static_cast<int32_t>(inp_W));

    if (in_bounds) {
      const auto base_offset = n * grad_input_strides[0] +
          iy_n * grad_input_strides[2] + ix_n * grad_input_strides[3];

      for (ulong c = 0; c < C; c++) {
        const auto grad_out_offset = n * grad_output_strides[0] +
            c * grad_output_strides[1] + out_h * grad_output_strides[2] +
            out_w * grad_output_strides[3];
        const float gOut = grad_output[grad_out_offset];
        atomic_fetch_add_explicit(
            &grad_input[base_offset + c * grad_input_strides[1]],
            gOut,
            memory_order_relaxed);
      }
    }
  } else { // bicubic
    float gix_mult, giy_mult;
    float ix = grid_sampler_unnormalize_set_grad(
        grid_x, static_cast<int32_t>(inp_W), align_corners, &gix_mult);
    float iy = grid_sampler_unnormalize_set_grad(
        grid_y, static_cast<int32_t>(inp_H), align_corners, &giy_mult);

    float ix_nw = floor(ix);
    float iy_nw = floor(iy);
    float tx = ix - ix_nw;
    float ty = iy - iy_nw;

    float x_coeffs[4], y_coeffs[4];
    get_cubic_coefficients(x_coeffs, tx);
    get_cubic_coefficients(y_coeffs, ty);

    for (ulong c = 0; c < C; c++) {
      const auto grad_out_offset = n * grad_output_strides[0] +
          c * grad_output_strides[1] + out_h * grad_output_strides[2] +
          out_w * grad_output_strides[3];
      const float gOut = grad_output[grad_out_offset];
      const auto NC_offset =
          n * grad_input_strides[0] + c * grad_input_strides[1];

      for (int32_t j = 0; j < 4; ++j) {
        for (int32_t i = 0; i < 4; ++i) {
          add_value_bounded_2d(
              grad_input,
              ix_nw - 1 + i,
              iy_nw - 1 + j,
              static_cast<int32_t>(inp_W),
              static_cast<int32_t>(inp_H),
              grad_input_strides[3],
              grad_input_strides[2],
              gOut * x_coeffs[i] * y_coeffs[j],
              padding_mode,
              align_corners,
              NC_offset);
        }
      }
    }
  }
}

// 2D backward kernel for grad_grid
template <typename T>
kernel void grid_sampler_2d_backward_grid(
    constant T* grad_output [[buffer(0)]],
    constant T* input [[buffer(1)]],
    constant T* grid [[buffer(2)]],
    device T* grad_grid [[buffer(3)]],
    constant int& interpolation_mode [[buffer(4)]],
    constant int& padding_mode [[buffer(5)]],
    constant bool& align_corners [[buffer(6)]],
    constant ulong* input_sizes [[buffer(7)]],
    constant ulong* output_sizes [[buffer(8)]],
    constant ulong* input_strides [[buffer(9)]],
    constant ulong* grad_grid_strides [[buffer(10)]],
    constant ulong* grid_strides [[buffer(11)]],
    constant ulong* grad_output_strides [[buffer(12)]],
    uint3 thread_index [[thread_position_in_grid]]) {
  const auto out_w = thread_index.x;
  const auto out_h = thread_index.y;
  const auto n = thread_index.z;

  if (n >= input_sizes[0] || out_h >= output_sizes[2] ||
      out_w >= output_sizes[3]) {
    return;
  }

  const auto C = input_sizes[1];
  const auto inp_H = input_sizes[2];
  const auto inp_W = input_sizes[3];

  const auto grid_offset =
      n * grid_strides[0] + out_h * grid_strides[1] + out_w * grid_strides[2];

  const opmath_t<T> grid_x = grid[grid_offset];
  const opmath_t<T> grid_y = grid[grid_offset + grid_strides[3]];

  const auto grad_grid_base_offset = n * grad_grid_strides[0] +
      out_h * grad_grid_strides[1] + out_w * grad_grid_strides[2];

  if (interpolation_mode == 0) { // bilinear
    opmath_t<T> gix_mult, giy_mult;
    opmath_t<T> ix = grid_sampler_compute_source_index_set_grad(
        grid_x,
        static_cast<int32_t>(inp_W),
        padding_mode,
        align_corners,
        &gix_mult);
    opmath_t<T> iy = grid_sampler_compute_source_index_set_grad(
        grid_y,
        static_cast<int32_t>(inp_H),
        padding_mode,
        align_corners,
        &giy_mult);

    const int32_t ix_nw = static_cast<int32_t>(floor(ix));
    const int32_t iy_nw = static_cast<int32_t>(floor(iy));
    const int32_t ix_ne = ix_nw + 1;
    const int32_t iy_ne = iy_nw;
    const int32_t ix_sw = ix_nw;
    const int32_t iy_sw = iy_nw + 1;
    const int32_t ix_se = ix_nw + 1;
    const int32_t iy_se = iy_nw + 1;

    opmath_t<T> gix = 0, giy = 0;

    for (ulong c = 0; c < C; c++) {
      const auto grad_out_offset = n * grad_output_strides[0] +
          c * grad_output_strides[1] + out_h * grad_output_strides[2] +
          out_w * grad_output_strides[3];
      const opmath_t<T> gOut = grad_output[grad_out_offset];
      const auto input_base_offset =
          n * input_strides[0] + c * input_strides[1];

      if (within_bounds_2d(
              iy_nw,
              ix_nw,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        const opmath_t<T> nw_val = input
            [input_base_offset + iy_nw * input_strides[2] +
             ix_nw * input_strides[3]];
        gix -= nw_val * (iy_se - iy) * gOut;
        giy -= nw_val * (ix_se - ix) * gOut;
      }
      if (within_bounds_2d(
              iy_ne,
              ix_ne,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        const opmath_t<T> ne_val = input
            [input_base_offset + iy_ne * input_strides[2] +
             ix_ne * input_strides[3]];
        gix += ne_val * (iy_sw - iy) * gOut;
        giy -= ne_val * (ix - ix_sw) * gOut;
      }
      if (within_bounds_2d(
              iy_sw,
              ix_sw,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        const opmath_t<T> sw_val = input
            [input_base_offset + iy_sw * input_strides[2] +
             ix_sw * input_strides[3]];
        gix -= sw_val * (iy - iy_ne) * gOut;
        giy += sw_val * (ix_ne - ix) * gOut;
      }
      if (within_bounds_2d(
              iy_se,
              ix_se,
              static_cast<int32_t>(inp_H),
              static_cast<int32_t>(inp_W))) {
        const opmath_t<T> se_val = input
            [input_base_offset + iy_se * input_strides[2] +
             ix_se * input_strides[3]];
        gix += se_val * (iy - iy_nw) * gOut;
        giy += se_val * (ix - ix_nw) * gOut;
      }
    }

    grad_grid[grad_grid_base_offset] = static_cast<T>(gix_mult * gix);
    grad_grid[grad_grid_base_offset + grad_grid_strides[3]] =
        static_cast<T>(giy_mult * giy);
  } else if (interpolation_mode == 1) { // nearest
    grad_grid[grad_grid_base_offset] = static_cast<T>(0);
    grad_grid[grad_grid_base_offset + grad_grid_strides[3]] = static_cast<T>(0);
  } else { // bicubic
    opmath_t<T> gix_mult, giy_mult;
    opmath_t<T> ix = grid_sampler_unnormalize_set_grad(
        grid_x, static_cast<int32_t>(inp_W), align_corners, &gix_mult);
    opmath_t<T> iy = grid_sampler_unnormalize_set_grad(
        grid_y, static_cast<int32_t>(inp_H), align_corners, &giy_mult);

    opmath_t<T> ix_nw = floor(ix);
    opmath_t<T> iy_nw = floor(iy);
    opmath_t<T> tx = ix - ix_nw;
    opmath_t<T> ty = iy - iy_nw;

    opmath_t<T> x_coeffs[4], y_coeffs[4];
    opmath_t<T> x_coeffs_grad[4], y_coeffs_grad[4];
    get_cubic_coefficients(x_coeffs, tx);
    get_cubic_coefficients(y_coeffs, ty);
    get_cubic_coefficients_grad(x_coeffs_grad, tx);
    get_cubic_coefficients_grad(y_coeffs_grad, ty);

    opmath_t<T> gix = 0, giy = 0;

    for (ulong c = 0; c < C; c++) {
      const auto grad_out_offset = n * grad_output_strides[0] +
          c * grad_output_strides[1] + out_h * grad_output_strides[2] +
          out_w * grad_output_strides[3];
      const opmath_t<T> gOut = grad_output[grad_out_offset];
      const auto inp_ptr_offset = n * input_strides[0] + c * input_strides[1];
      constant T* inp_ptr_NC = input + inp_ptr_offset;

      for (int32_t j = 0; j < 4; ++j) {
        for (int32_t i = 0; i < 4; ++i) {
          opmath_t<T> val = get_value_bounded_2d<T>(
              inp_ptr_NC,
              ix_nw - 1 + i,
              iy_nw - 1 + j,
              static_cast<int32_t>(inp_W),
              static_cast<int32_t>(inp_H),
              input_strides[3],
              input_strides[2],
              padding_mode,
              align_corners);
          gix -= val * x_coeffs_grad[i] * y_coeffs[j] * gOut;
          giy -= val * y_coeffs_grad[j] * x_coeffs[i] * gOut;
        }
      }
    }

    grad_grid[grad_grid_base_offset] = static_cast<T>(gix_mult * gix);
    grad_grid[grad_grid_base_offset + grad_grid_strides[3]] =
        static_cast<T>(giy_mult * giy);
  }
}

#define REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PAD, PNAME)      \
  template [[host_name("grid_sampler_2d_" INAME "_" PNAME "_" #DTYPE)]] \
  kernel void grid_sampler_2d<INTERP<PAD>, DTYPE>(                      \
      device DTYPE * output [[buffer(0)]],                              \
      constant DTYPE * input [[buffer(1)]],                             \
      constant DTYPE * grid [[buffer(2)]],                              \
      constant GridSamplerParams<4> & params [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, INTERP, INAME)         \
  REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PadZeros, "zeros")   \
  REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PadBorder, "border") \
  REGISTER_GRID_SAMPLER_2D(DTYPE, INTERP, INAME, PadReflection, "reflection")

#define REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PAD, PNAME)      \
  template [[host_name("grid_sampler_3d_" INAME "_" PNAME "_" #DTYPE)]] \
  kernel void grid_sampler_3d<INTERP<PAD>, DTYPE>(                      \
      device DTYPE * output [[buffer(0)]],                              \
      constant DTYPE * input [[buffer(1)]],                             \
      constant DTYPE * grid [[buffer(2)]],                              \
      constant GridSamplerParams<5> & params [[buffer(3)]],             \
      uint tid [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_3D_INTERP(DTYPE, INTERP, INAME)         \
  REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PadZeros, "zeros")   \
  REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PadBorder, "border") \
  REGISTER_GRID_SAMPLER_3D(DTYPE, INTERP, INAME, PadReflection, "reflection")

#define REGISTER_GRID_SAMPLER_BACKWARD(DTYPE)                      \
  template [[host_name("grid_sampler_3d_backward_input_" #DTYPE)]] \
  kernel void grid_sampler_3d_backward_input<DTYPE>(               \
      constant DTYPE * grad_output [[buffer(0)]],                  \
      constant DTYPE * grid [[buffer(1)]],                         \
      device atomic<float> * grad_input [[buffer(2)]],             \
      constant int& interpolation_mode [[buffer(3)]],              \
      constant int& padding_mode [[buffer(4)]],                    \
      constant bool& align_corners [[buffer(5)]],                  \
      constant ulong* input_sizes [[buffer(6)]],                   \
      constant ulong* output_sizes [[buffer(7)]],                  \
      constant ulong* grad_input_strides [[buffer(8)]],            \
      constant ulong* grid_strides [[buffer(9)]],                  \
      constant ulong* grad_output_strides [[buffer(10)]],          \
      uint3 thread_index [[thread_position_in_grid]]);             \
                                                                   \
  template [[host_name("grid_sampler_3d_backward_grid_" #DTYPE)]]  \
  kernel void grid_sampler_3d_backward_grid<DTYPE>(                \
      constant DTYPE * grad_output [[buffer(0)]],                  \
      constant DTYPE * input [[buffer(1)]],                        \
      constant DTYPE * grid [[buffer(2)]],                         \
      device DTYPE * grad_grid [[buffer(3)]],                      \
      constant int& interpolation_mode [[buffer(4)]],              \
      constant int& padding_mode [[buffer(5)]],                    \
      constant bool& align_corners [[buffer(6)]],                  \
      constant ulong* input_sizes [[buffer(7)]],                   \
      constant ulong* output_sizes [[buffer(8)]],                  \
      constant ulong* input_strides [[buffer(9)]],                 \
      constant ulong* grad_grid_strides [[buffer(10)]],            \
      constant ulong* grid_strides [[buffer(11)]],                 \
      constant ulong* grad_output_strides [[buffer(12)]],          \
      uint3 thread_index [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_2D_BACKWARD(DTYPE)                   \
  template [[host_name("grid_sampler_2d_backward_input_" #DTYPE)]] \
  kernel void grid_sampler_2d_backward_input<DTYPE>(               \
      constant DTYPE * grad_output [[buffer(0)]],                  \
      constant DTYPE * grid [[buffer(1)]],                         \
      device atomic<float> * grad_input [[buffer(2)]],             \
      constant int& interpolation_mode [[buffer(3)]],              \
      constant int& padding_mode [[buffer(4)]],                    \
      constant bool& align_corners [[buffer(5)]],                  \
      constant ulong* input_sizes [[buffer(6)]],                   \
      constant ulong* output_sizes [[buffer(7)]],                  \
      constant ulong* grad_input_strides [[buffer(8)]],            \
      constant ulong* grid_strides [[buffer(9)]],                  \
      constant ulong* grad_output_strides [[buffer(10)]],          \
      uint3 thread_index [[thread_position_in_grid]]);             \
                                                                   \
  template [[host_name("grid_sampler_2d_backward_grid_" #DTYPE)]]  \
  kernel void grid_sampler_2d_backward_grid<DTYPE>(                \
      constant DTYPE * grad_output [[buffer(0)]],                  \
      constant DTYPE * input [[buffer(1)]],                        \
      constant DTYPE * grid [[buffer(2)]],                         \
      device DTYPE * grad_grid [[buffer(3)]],                      \
      constant int& interpolation_mode [[buffer(4)]],              \
      constant int& padding_mode [[buffer(5)]],                    \
      constant bool& align_corners [[buffer(6)]],                  \
      constant ulong* input_sizes [[buffer(7)]],                   \
      constant ulong* output_sizes [[buffer(8)]],                  \
      constant ulong* input_strides [[buffer(9)]],                 \
      constant ulong* grad_grid_strides [[buffer(10)]],            \
      constant ulong* grid_strides [[buffer(11)]],                 \
      constant ulong* grad_output_strides [[buffer(12)]],          \
      uint3 thread_index [[thread_position_in_grid]]);

#define REGISTER_GRID_SAMPLER_OPS(DTYPE)                         \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Bilinear2D, "bilinear") \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Nearest2D, "nearest")   \
  REGISTER_GRID_SAMPLER_2D_INTERP(DTYPE, Bicubic2D, "bicubic")   \
  REGISTER_GRID_SAMPLER_3D_INTERP(DTYPE, Bilinear3D, "bilinear") \
  REGISTER_GRID_SAMPLER_3D_INTERP(DTYPE, Nearest3D, "nearest")   \
  REGISTER_GRID_SAMPLER_2D_BACKWARD(DTYPE)                       \
  REGISTER_GRID_SAMPLER_BACKWARD(DTYPE)

REGISTER_GRID_SAMPLER_OPS(float);
REGISTER_GRID_SAMPLER_OPS(half);
REGISTER_GRID_SAMPLER_OPS(bfloat);
