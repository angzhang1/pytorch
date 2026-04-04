#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/GridSamplerUtils.h>
#include <ATen/native/Pool.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/kernels/GridSampler.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/grid_sampler_2d.h>
#include <ATen/ops/grid_sampler_2d_native.h>
#include <ATen/ops/grid_sampler_3d_backward_native.h>
#include <ATen/ops/grid_sampler_3d_native.h>
#endif

namespace at::native {

#ifndef PYTORCH_JIT_COMPILE_SHADERS
static auto& lib = mps::MetalShaderLibrary::getBundledLibrary();
#else
#include <ATen/native/mps/GridSampler_metallib.h>
#endif

namespace mps {

static const char* interp_to_string(GridSamplerInterpolation mode) {
  switch (mode) {
    case GridSamplerInterpolation::Bilinear:
      return "bilinear";
    case GridSamplerInterpolation::Nearest:
      return "nearest";
    case GridSamplerInterpolation::Bicubic:
      return "bicubic";
  }
  TORCH_CHECK(false, "Unrecognised interpolation mode: ", mode);
  return "";
}

static const char* padding_to_string(GridSamplerPadding mode) {
  switch (mode) {
    case GridSamplerPadding::Zeros:
      return "zeros";
    case GridSamplerPadding::Border:
      return "border";
    case GridSamplerPadding::Reflection:
      return "reflection";
  }
  TORCH_CHECK(false, "Unrecognised padding mode: ", mode);
  return "";
}

static void grid_sampler_2d_mps_impl(Tensor& output,
                                     const Tensor& input,
                                     const Tensor& grid,
                                     int64_t _interpolation_mode,
                                     int64_t _padding_mode,
                                     bool align_corners) {
  using namespace mps;
  check_grid_sampler_common(input, grid);
  check_grid_sampler_2d(input, grid);

  TORCH_CHECK(input.scalar_type() == grid.scalar_type(),
              "expected input and grid to have the same type, but got ",
              input.scalar_type(),
              " and ",
              grid.scalar_type());

  TORCH_CHECK_NOT_IMPLEMENTED(!c10::isComplexType(input.scalar_type()),
                              "grid_sampler_2d is not supported for complex on MPS");

  auto interpolation_mode = static_cast<GridSamplerInterpolation>(_interpolation_mode);
  auto padding_mode = static_cast<GridSamplerPadding>(_padding_mode);

  auto dims = input.dim();

  GridSamplerParams<4> params;
  params.sampler_dims = 2;
  params.align_corners = align_corners;

  for (const auto dim : c10::irange(dims)) {
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = safe_downcast<int32_t, int64_t>(output.stride(dim));
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = safe_downcast<int32_t, int64_t>(input.stride(dim));
    params.grid_sizes[dim] = safe_downcast<int32_t, int64_t>(grid.size(dim));
    params.grid_strides[dim] = safe_downcast<int32_t, int64_t>(grid.stride(dim));
  }

  auto N = output.size(0);
  auto out_H = output.size(2);
  auto out_W = output.size(3);
  auto num_threads = N * out_H * out_W;

  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("grid_sampler_2d_{}_{}_{}",
                                                         interp_to_string(interpolation_mode),
                                                         padding_to_string(padding_mode),
                                                         scalarToMetalTypeString(input)));

      getMPSProfiler().beginProfileKernel(pso, "grid_sampler_2d", {input, grid});
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, output, input, grid, params);

      mtl_dispatch1DJob(computeEncoder, pso, num_threads);
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

static void grid_sampler_3d_mps_impl(Tensor& output,
                                     const Tensor& input,
                                     const Tensor& grid,
                                     int64_t _interpolation_mode,
                                     int64_t _padding_mode,
                                     bool align_corners,
                                     int32_t sampler_dims,
                                     const std::string& op_name) {
  check_grid_sampler_common(input, grid);
  switch (sampler_dims) {
    case 2:
      check_grid_sampler_2d(input, grid);
      break;
    case 3:
      check_grid_sampler_3d(input, grid, _interpolation_mode);
      break;
    default:
      TORCH_INTERNAL_ASSERT(false, "Only 2D and 3D sampling are supported, but got: ", sampler_dims);
  }
  TORCH_CHECK(input.scalar_type() == grid.scalar_type(),
              "expected input and grid to have the same type, but got ",
              input.scalar_type(),
              " and ",
              grid.scalar_type());

  auto interpolation_mode = static_cast<GridSamplerInterpolation>(_interpolation_mode);
  auto padding_mode = static_cast<GridSamplerPadding>(_padding_mode);

  switch (interpolation_mode) {
    case GridSamplerInterpolation::Bilinear:
    case GridSamplerInterpolation::Nearest:
      break;
    case GridSamplerInterpolation::Bicubic:
      TORCH_CHECK(false, op_name, ": Unsupported Bicubic interpolation");
      break;
    default:
      TORCH_CHECK(false, op_name, ": Unrecognised interpolation mode: ", _interpolation_mode);
  }

  auto input_size = input.sizes();
  auto grid_size = grid.sizes();
  output.resize_({input_size[0], input_size[1], grid_size[1], grid_size[2], grid_size[3]}, MemoryFormat::Contiguous);

  auto dims = input.dim();

  GridSamplerParams<5> params;
  params.sampler_dims = sampler_dims;
  params.align_corners = align_corners;

  for (const auto dim : c10::irange(dims)) {
    params.output_sizes[dim] = safe_downcast<int32_t, int64_t>(output.size(dim));
    params.output_strides[dim] = safe_downcast<int32_t, int64_t>(output.stride(dim));
    params.input_sizes[dim] = safe_downcast<int32_t, int64_t>(input.size(dim));
    params.input_strides[dim] = safe_downcast<int32_t, int64_t>(input.stride(dim));
    params.grid_sizes[dim] = safe_downcast<int32_t, int64_t>(grid.size(dim));
    params.grid_strides[dim] = safe_downcast<int32_t, int64_t>(grid.stride(dim));
  }

  auto num_threads = input_size[0] * grid_size[1] * grid_size[2] * grid_size[3];
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      auto pso = lib.getPipelineStateForFunc(fmt::format("grid_sampler_3d_{}_{}_{}",
                                                         interp_to_string(interpolation_mode),
                                                         padding_to_string(padding_mode),
                                                         scalarToMetalTypeString(input)));

      getMPSProfiler().beginProfileKernel(pso, op_name, {input, grid});
      [computeEncoder setComputePipelineState:pso];
      mtl_setArgs(computeEncoder, output, input, grid, params);

      mtl_dispatch1DJob(computeEncoder, pso, num_threads);
      getMPSProfiler().endProfileKernel(pso);
    }
  });
}

} // namespace mps

Tensor grid_sampler_2d_mps(const Tensor& input,
                           const Tensor& grid,
                           int64_t interpolation_mode,
                           int64_t padding_mode,
                           bool align_corners) {
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty({in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());

  mps::grid_sampler_2d_mps_impl(output, input, grid, interpolation_mode, padding_mode, align_corners);
  return output;
}

Tensor grid_sampler_3d_mps(const Tensor& input,
                           const Tensor& grid,
                           int64_t interpolation_mode,
                           int64_t padding_mode,
                           bool align_corners) {
  auto output = at::empty({0}, input.options(), MemoryFormat::Contiguous);
  mps::grid_sampler_3d_mps_impl(output,
                                input,
                                grid,
                                interpolation_mode,
                                padding_mode,
                                align_corners,
                                /*sampler_dims=*/3,
                                /*op_name=*/"grid_sampler_3d");
  return output;
}

std::tuple<Tensor, Tensor> grid_sampler_3d_backward_mps(const Tensor& grad_output,
                                                        const Tensor& input,
                                                        const Tensor& grid,
                                                        int64_t interpolation_mode,
                                                        int64_t padding_mode,
                                                        bool align_corners,
                                                        std::array<bool, 2> output_mask) {
  using namespace mps;
  check_grid_sampler_common(input, grid);
  check_grid_sampler_3d(input, grid, interpolation_mode);

  TORCH_CHECK_NOT_IMPLEMENTED(interpolation_mode == 0 || interpolation_mode == 1,
                              "grid_sampler_3d backward on MPS only supports bilinear and nearest interpolation");

  TORCH_CHECK(input.scalar_type() == grid.scalar_type(),
              "expected input and grid to have the same type, but got ",
              input.scalar_type(),
              " and ",
              grid.scalar_type());

  auto orig_dtype = input.scalar_type();
  auto input_requires_grad = output_mask[0];
  auto grid_requires_grad = output_mask[1];
  int32_t interp_mode = static_cast<int32_t>(interpolation_mode);
  int32_t pad_mode = static_cast<int32_t>(padding_mode);

  // backward_input uses atomic<float> (Metal lacks atomic<half>/atomic<bfloat>),
  // so grad_input is always float32 and converted back after the kernel.
  Tensor grad_input;
  if (input_requires_grad) {
    grad_input = at::zeros(input.sizes(), input.options().dtype(at::kFloat));
  }
  auto grad_grid = grid_requires_grad ? at::empty_like(grid, MemoryFormat::Contiguous) : at::Tensor();

  const Tensor& input_contiguous = input.is_contiguous() ? input : input.contiguous();
  const Tensor& grid_contiguous = grid.is_contiguous() ? grid : grid.contiguous();
  const Tensor& grad_output_contiguous = grad_output.is_contiguous() ? grad_output : grad_output.contiguous();

  auto N = input_contiguous.size(0);
  auto C = input_contiguous.size(1);
  auto in_D = input_contiguous.size(2);
  auto in_H = input_contiguous.size(3);
  auto in_W = input_contiguous.size(4);
  auto out_D = grid_contiguous.size(1);
  auto out_H = grid_contiguous.size(2);
  auto out_W = grid_contiguous.size(3);

  std::array<uint64_t, 5> input_sizes = {static_cast<uint64_t>(N),
                                         static_cast<uint64_t>(C),
                                         static_cast<uint64_t>(in_D),
                                         static_cast<uint64_t>(in_H),
                                         static_cast<uint64_t>(in_W)};
  std::array<uint64_t, 5> output_sizes = {static_cast<uint64_t>(N),
                                          static_cast<uint64_t>(C),
                                          static_cast<uint64_t>(out_D),
                                          static_cast<uint64_t>(out_H),
                                          static_cast<uint64_t>(out_W)};
  std::array<uint64_t, 5> input_strides = {static_cast<uint64_t>(input_contiguous.stride(0)),
                                           static_cast<uint64_t>(input_contiguous.stride(1)),
                                           static_cast<uint64_t>(input_contiguous.stride(2)),
                                           static_cast<uint64_t>(input_contiguous.stride(3)),
                                           static_cast<uint64_t>(input_contiguous.stride(4))};
  std::array<uint64_t, 5> grid_strides = {static_cast<uint64_t>(grid_contiguous.stride(0)),
                                          static_cast<uint64_t>(grid_contiguous.stride(1)),
                                          static_cast<uint64_t>(grid_contiguous.stride(2)),
                                          static_cast<uint64_t>(grid_contiguous.stride(3)),
                                          static_cast<uint64_t>(grid_contiguous.stride(4))};
  std::array<uint64_t, 5> grad_output_strides = {static_cast<uint64_t>(grad_output_contiguous.stride(0)),
                                                 static_cast<uint64_t>(grad_output_contiguous.stride(1)),
                                                 static_cast<uint64_t>(grad_output_contiguous.stride(2)),
                                                 static_cast<uint64_t>(grad_output_contiguous.stride(3)),
                                                 static_cast<uint64_t>(grad_output_contiguous.stride(4))};

  MPSStream* mpsStream = getCurrentMPSStream();

  bool run_grad_input = input_requires_grad;
  bool run_grad_grid = grid_requires_grad && interp_mode != 1;

  if (grid_requires_grad && interp_mode == 1) {
    grad_grid.zero_();
  }

  dispatch_sync_with_rethrow(mpsStream->queue(), ^() {
    @autoreleasepool {
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();

      if (run_grad_input) {
        auto gradInputPSO = lib.getPipelineStateForFunc(
            fmt::format("grid_sampler_3d_backward_input_{}", scalarToMetalTypeString(input)));

        getMPSProfiler().beginProfileKernel(
            gradInputPSO, "grid_sampler_3d_backward_input", {grad_output_contiguous, grid_contiguous, grad_input});

        [computeEncoder setComputePipelineState:gradInputPSO];

        std::array<uint64_t, 5> grad_input_strides = {static_cast<uint64_t>(grad_input.stride(0)),
                                                      static_cast<uint64_t>(grad_input.stride(1)),
                                                      static_cast<uint64_t>(grad_input.stride(2)),
                                                      static_cast<uint64_t>(grad_input.stride(3)),
                                                      static_cast<uint64_t>(grad_input.stride(4))};

        mtl_setArgs(computeEncoder,
                    grad_output_contiguous,
                    grid_contiguous,
                    grad_input,
                    interp_mode,
                    pad_mode,
                    align_corners,
                    input_sizes,
                    output_sizes,
                    grad_input_strides,
                    grid_strides,
                    grad_output_strides);

        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
        MTLSize threadsPerGrid = MTLSizeMake(out_W, out_H * out_D, N);
        [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        getMPSProfiler().endProfileKernel(gradInputPSO);
      }

      if (run_grad_grid) {
        auto gradGridPSO = lib.getPipelineStateForFunc(
            fmt::format("grid_sampler_3d_backward_grid_{}", scalarToMetalTypeString(input)));

        getMPSProfiler().beginProfileKernel(gradGridPSO,
                                            "grid_sampler_3d_backward_grid",
                                            {grad_output_contiguous, input_contiguous, grid_contiguous, grad_grid});

        [computeEncoder setComputePipelineState:gradGridPSO];

        std::array<uint64_t, 5> grad_grid_strides = {static_cast<uint64_t>(grad_grid.stride(0)),
                                                     static_cast<uint64_t>(grad_grid.stride(1)),
                                                     static_cast<uint64_t>(grad_grid.stride(2)),
                                                     static_cast<uint64_t>(grad_grid.stride(3)),
                                                     static_cast<uint64_t>(grad_grid.stride(4))};

        mtl_setArgs(computeEncoder,
                    grad_output_contiguous,
                    input_contiguous,
                    grid_contiguous,
                    grad_grid,
                    interp_mode,
                    pad_mode,
                    align_corners,
                    input_sizes,
                    output_sizes,
                    input_strides,
                    grad_grid_strides,
                    grid_strides,
                    grad_output_strides);

        MTLSize threadsPerThreadgroup = MTLSizeMake(16, 16, 1);
        MTLSize threadsPerGrid = MTLSizeMake(out_W, out_H * out_D, N);
        [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];

        getMPSProfiler().endProfileKernel(gradGridPSO);
      }
    }
  });

  if (input_requires_grad && orig_dtype != ScalarType::Float) {
    grad_input = grad_input.to(orig_dtype);
  }

  return std::make_tuple(std::move(grad_input), std::move(grad_grid));
}

} // namespace at::native
