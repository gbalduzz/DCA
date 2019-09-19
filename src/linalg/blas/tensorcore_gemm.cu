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
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>

#include "dca/linalg/util/handle_functions.hpp"
#include "dca/linalg/util/stream_functions.hpp"
#include "dca/util/integer_division.hpp"

namespace dca {
namespace linalg {
namespace blas {
namespace kernel {
// dca::linalg::blas::kernel::

// Inverse transform of fp32_in[:] = fp16_high[:] * scale1 + fp16_low[:] * scale2
void __global__ split(const MatrixView<float, GPU> fp32_in, const float scale, const float scale2,
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
  const __half high = __float2half(original * scale);
  fp16_high(i, j) = high;

  const float diff = original - __half2float(high) / scale;
  const __half low = __float2half(diff * scale2);
  fp16_low(i, j) = low;
}
}  // namespace kernel
// dca::linalg::blas::

void tensorcoreGemm(const float alpha, const MatrixView<float, GPU>& a,
                    const MatrixView<float, GPU>& b, std::array<Matrix<__half, GPU>, 4>& workspace,
                    const float beta, MatrixView<float, GPU> c, int thread_id, int stream_id) {
  assert(a.nrCols() == b.nrRows());
  assert(a.nrRows() == c.nrRows());
  assert(b.nrCols() == c.nrCols());

  auto max_matrix = [](const MatrixView<float, GPU>& m) -> float {
    float max_val = 0;
    for (int j = 0; j < m.nrCols(); ++j) {
      const float val =
          thrust::reduce(thrust::device_pointer_cast(m.ptr(0, j)),
                         thrust::device_pointer_cast(m.ptr(m.nrRows(), j)), 1.f,
                         [] __device__ __host__(float a, float b) { return max(fabs(a), fabs(b)); });

      max_val = std::max(val, max_val);
    }
    return max_val;
  };

  auto prev_power2 = [](float x) {
    int* casted = reinterpret_cast<int*>(&x);
    const int modded = *casted & (0x1ff << 23);
    return *reinterpret_cast<const float*>(&modded);
  };

  auto get_scale = [&](const auto& m) {
    const float max_mat = max_matrix(m);
    constexpr auto max_half = 65504;
    const float scale1 = prev_power2(max_half / max_mat);
    constexpr int two_to_11 = 1 << 11;
    return std::array<float, 2>{scale1, scale1 * two_to_11};
  };

  const auto scale_a = get_scale(a);
  const auto scale_b = get_scale(b);

  const dim3 threads(16, 16);
  using dca::util::ceilDiv;
  auto stream = util::getStream(thread_id, stream_id);

  auto padd = [](int size, int to) { return (size + to - 1) / to * to; };

  auto split = [&](const auto& m, const auto& scale, auto& high, auto& low, int padx, int pady) {
    high.resizeNoCopy(std::make_pair(padd(m.nrRows(), padx), padd(m.nrCols(), pady)));
    low.resizeNoCopy(high.size());

    dim3 blocks(ceilDiv(high.nrRows(), int(threads.x)), ceilDiv(high.nrCols(), int(threads.y)));
    kernel::split<<<blocks, threads, 0, stream>>>(m, scale[0], scale[1], high, low);
  };

  auto& a_high = workspace[0];
  auto& a_low = workspace[1];
  auto& b_high = workspace[2];
  auto& b_low = workspace[3];

  split(a, scale_a, a_high, a_low, 8, 8);
  split(b, scale_b, b_high, b_low, 8, 8);

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

  // c <- beta* c + alpha * (a_high * b_high) / scale1**2
  const auto alpha_11 = alpha / (scale_a[0] * scale_b[0]);
  multiply(alpha_11, a_high, b_high, beta, c);

  // c += alpha * (a_high * b_low) / (scale1 * scale2)
  const auto alpha_12 = alpha / (scale_a[0] * scale_b[1]);
  multiply(alpha_12, a_high, b_low, 1., c);

  // c += alpha * (a_low * b_high) / (scale1 * scale2)
  const auto alpha_21 = alpha / (scale_a[1] * scale_b[0]);  // Note: equal to alpha_12.
  multiply(alpha_21, a_low, b_high, 1., c);
}

}  // namespace blas
}  // namespace linalg
}  // namespace dca
