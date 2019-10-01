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
constexpr unsigned two_to_11 = 2048;

namespace kernel {
// dca::linalg::blas::kernel::

// Inverse transform of fp32_in[:] = fp16_high[:] * scale1 + fp16_low[:] * scale2
void __global__ split(int nri, int nci, int nro, int nco, const float* __restrict__ in,
                      const int ldi, const float* __restrict__ scale, __half* __restrict__ high,
                      __half* __restrict__ low, const int ldo) {
  const int i = threadIdx.x + blockDim.x * blockIdx.x;
  const int j = threadIdx.y + blockDim.y * blockIdx.y;
  if (i >= nro || j >= nco)
    return;

  const int dist_o = i + ldo * j;
  if (i >= nri || j >= nci) {  // padding area.
    high[dist_o] = low[dist_o] = 0;
    return;
  }

  const float scaled = in[i + ldi * j] * scale[0];

  const __half h = __float2half(scaled);
  const __half l = __float2half((scaled - __half2float(h)) * two_to_11);

  high[dist_o] = h;
  low[dist_o] = l;
}

union FloatU {
  float f;
  int i;
};

void __global__ computeScale(const float* __restrict__ max_val, float* __restrict__ scale) {
  auto prev_power2 = [](float x) -> float {
    FloatU u = {x};
    u.i = u.i & (0x1ff << 23);  // Set mantissa to zero.
    return u.f;
  };

  constexpr auto max_half = 65504;
  *scale = prev_power2(max_half / *max_val);
}

void __global__ setAlphaBeta(const float* __restrict__ scales, float* __restrict__ alphas_beta,
                             const float alpha, const float beta) {
  alphas_beta[0] = alpha / (scales[0] * scales[1]);
  alphas_beta[1] = alphas_beta[0] / two_to_11;
  alphas_beta[2] = beta;
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
    computeScale(scales_and_alphas_.ptr(), a, stream);
    computeScale(scales_and_alphas_.ptr(1), b, stream);
  }
  kernel::setAlphaBeta<<<1, 1, 0, stream>>>(scales_and_alphas_.ptr(), scales_and_alphas_.ptr(2),
                                            alpha, beta);

  const dim3 threads(16, 16);
  using dca::util::ceilDiv;

  auto padd = [](int size, int to) { return (size + to - 1) / to * to; };

  auto split = [&](const auto& m, const float* scale, auto& high, auto& low, int padx, int pady) {
    high.resizeNoCopy(std::make_pair(padd(m.nrRows(), padx), padd(m.nrCols(), pady)));
    low.resizeNoCopy(high.size());

    dim3 blocks(ceilDiv(high.nrRows(), int(threads.x)), ceilDiv(high.nrCols(), int(threads.y)));

    kernel::split<<<blocks, threads, 0, stream>>>(
        m.nrRows(), m.nrCols(), high.nrRows(), high.nrCols(), m.ptr(), m.leadingDimension(), scale,
        high.ptr(), low.ptr(), high.leadingDimension());
  };

  auto& a_high = (*workspace_)[0];
  auto& a_low = (*workspace_)[1];
  auto& b_high = (*workspace_)[2];
  auto& b_low = (*workspace_)[3];

  split(a, scales_and_alphas_.ptr(), a_high, a_low, 8, 8);
  split(b, scales_and_alphas_.ptr(1), b_high, b_low, 8, 8);

  auto handle = util::getHandle(thread_id, stream_id);

  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
  auto multiply = [&](const float* alpha_dev, const auto& a, const auto& b, const float* beta_dev,
                      auto& c) {
    const int m = a.nrRows();
    const int n = b.nrCols();
    const int k = a.nrCols();

    assert(n % 8 == 0);
    assert(m % 8 == 0);
    assert(k % 8 == 0);

    auto err = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha_dev, a.ptr(),
                            CUDA_R_16F, a.leadingDimension(), b.ptr(), CUDA_R_16F,
                            b.leadingDimension(), beta_dev, c.ptr(), CUDA_R_32F,
                            c.leadingDimension(), CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    assert(err == CUBLAS_STATUS_SUCCESS);
  };

  // c <- beta * c + alpha * (a_high * b_high) / scale1**2
  multiply(scales_and_alphas_.ptr(2), a_high, b_high, scales_and_alphas_.ptr(4), c);

  // c += alpha * (a_high * b_low) / (scale1 * scale2)
  multiply(scales_and_alphas_.ptr(3), a_high, b_low, scales_and_alphas_.ptr(5), c);

  // c += alpha * (a_low * b_high) / (scale1 * scale2)
  multiply(scales_and_alphas_.ptr(3), a_low, b_high, scales_and_alphas_.ptr(5), c);

  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

  ++n_calls_;
  if (n_calls_ >= calls_per_check_)
    n_calls_ = 0;
}

void TensorcoreGemm::computeScale(float* scale, const MatrixView<float, GPU>& m, cudaStream_t stream) {
  float* max_val = linalg::util::reduceAbsMatrix(m, reduction_wp_, stream);
  kernel::computeScale<<<1, 1, 0, stream>>>(max_val, scale);
}

}  // namespace blas
}  // namespace linalg
}  // namespace dca
