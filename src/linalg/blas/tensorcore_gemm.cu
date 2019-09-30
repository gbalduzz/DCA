// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Implements tensorcore_gemm.hpp

#include "dca/linalg/blas/tensorcore_gemm.hpp"

#include <iostream>
#include <cublas_v2.h>

#include "dca/linalg/util/handle_functions.hpp"
#include "dca/linalg/util/reductions.hpp"
#include "dca/linalg/util/stream_functions.hpp"
#include "dca/util/integer_division.hpp"

namespace dca {
namespace linalg {
namespace blas {
namespace kernel {
// dca::linalg::blas::kernel::

// Inverse transform of fp32_in[:] = fp16_high[:] * scale1 + fp16_low[:] * scale2
void __global__ split(const MatrixView<float, GPU> fp32_in, const float* scales,
                      MatrixView<__half, GPU> fp16_high, MatrixView<__half, GPU> fp16_low) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  const int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= fp16_high.nrRows() || j >= fp16_high.nrCols())
    return;
  if (i >= fp32_in.nrRows() || j >= fp32_in.nrCols()) {  // padding area.
    fp16_high(i, j) = fp16_low(i, j) = 0;
    return;
  }

  const float original = fp32_in(i, j);
  const float scale = scales[0];
  const __half high = __float2half(original * scale);
  fp16_high(i, j) = high;

  const float scale2 = scales[1];
  const float diff = original - __half2float(high) / scale;
  const __half low = __float2half(diff * scale2);
  fp16_low(i, j) = low;
}

union FloatU {
  float f;
  int i;
};

void __global__ computeScale(const float* max_val, float* scale) {
  auto prev_power2 = [](float x) -> float {
    FloatU u = {x};
    u.i = u.i & (0x1ff << 23);  // Set mantissa to zero.
    return u.f;
  };

  constexpr auto max_half = 65504;
  scale[0] = prev_power2(max_half / *max_val);
  constexpr unsigned pow11 = 1 << 11;
  scale[1] = scale[0] * pow11;
}

}  // namespace kernel

void TensorcoreGemm::execute(const float alpha, const MatrixView<float, GPU>& a,
                             const MatrixView<float, GPU>& b, const float beta,
                             MatrixView<float, GPU> c, int thread_id, int stream_id) {
  assert(a.nrCols() == b.nrRows());
  assert(a.nrRows() == c.nrRows());
  assert(b.nrCols() == c.nrCols());
  auto stream = util::getStream(thread_id, stream_id);

  // Compute scale factor once every calls_per_check_ calls.
  if (n_calls_ == 0) {
    computeScale(scales_dev_.ptr(), a, stream);
    computeScale(scales_dev_.ptr(2), b, stream);
    scales_host_.setAsync(scales_dev_, thread_id, stream_id);
    scale_copied_.record(stream);
  }

  const dim3 threads(16, 16);
  using dca::util::ceilDiv;

  auto padd = [](int size, int to) { return (size + to - 1) / to * to; };

  auto split = [&](const auto& m, const float* scales, auto& high, auto& low, int padx, int pady) {
    high.resizeNoCopy(std::make_pair(padd(m.nrRows(), padx), padd(m.nrCols(), pady)));
    low.resizeNoCopy(high.size());

    dim3 blocks(ceilDiv(high.nrRows(), int(threads.x)), ceilDiv(high.nrCols(), int(threads.y)));
    kernel::split<<<blocks, threads, 0, stream>>>(m, scales, high, low);
  };

  auto& a_high = (*workspace_)[0];
  auto& a_low = (*workspace_)[1];
  auto& b_high = (*workspace_)[2];
  auto& b_low = (*workspace_)[3];

  split(a, scales_dev_.ptr(), a_high, a_low, 8, 8);
  split(b, scales_dev_.ptr(2), b_high, b_low, 8, 8);

  auto handle = util::getHandle(thread_id, stream_id);

  auto multiply = [&](float alpha, const auto& a, const auto& b, float beta, auto& c) {
    const int m = a.nrRows();
    const int n = b.nrCols();
    const int k = a.nrCols();

    assert(n % 8 == 0);
    assert(m % 8 == 0);
    assert(k % 8 == 0);

    auto err = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.ptr(), CUDA_R_16F,
                            a.leadingDimension(), b.ptr(), CUDA_R_16F, b.leadingDimension(), &beta,
                            c.ptr(), CUDA_R_32F, c.leadingDimension(), CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    assert(err == CUBLAS_STATUS_SUCCESS);
  };

  // TODO: store alpha factor on the device. Remove sync.
  if (n_calls_ == 0)
    scale_copied_.block();

  const float* const scale_a = scales_host_.ptr(0);
  const float* const scale_b = scales_host_.ptr(2);

  // c <- beta* c + alpha * (a_high * b_high) / scale1**2
  const auto alpha_11 = alpha / (scale_a[0] * scale_b[0]);
  multiply(alpha_11, a_high, b_high, beta, c);

  // c += alpha * (a_high * b_low) / (scale1 * scale2)
  const auto alpha_12 = alpha / (scale_a[0] * scale_b[1]);
  multiply(alpha_12, a_high, b_low, 1., c);

  // c += alpha * (a_low * b_high) / (scale1 * scale2)
  const auto alpha_21 = alpha / (scale_a[1] * scale_b[0]);  // Note: equal to alpha_12.
  multiply(alpha_21, a_low, b_high, 1., c);

  ++n_calls_;
  if (n_calls_ >= calls_per_check_)
    n_calls_ = 0;
}

void TensorcoreGemm::computeScale(float* scales, const MatrixView<float, GPU>& m,
                                  cudaStream_t stream) {
  float* max_val = linalg::util::reduceAbsMatrix(m, reduction_wp_, stream);
  kernel::computeScale<<<1, 1, 0, stream>>>(max_val, scales);
}

}  // namespace blas
}  // namespace linalg
}  // namespace dca
