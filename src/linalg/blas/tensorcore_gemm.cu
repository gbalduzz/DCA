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
#include "dca/linalg/util/stream_functions.hpp"
#include "dca/util/integer_division.hpp"

namespace dca {
namespace linalg {
namespace blas {
namespace kernel {
// dca::linalg::blas::kernel::

// Inverse transform of fp32_in[:] = fp16_high[:] / scale1 + fp16_low[:] / scale2
// TODO: multiply here and divide later.
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
  const __half high = __float2half(original / scale);
  fp16_high(i, j) = high;

  const float diff = original - __half2float(high);
  const __half low = __float2half(diff * scale / scale2);
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

  const float scale1 = 1.;  // TODO: change
  const float scale2 = scale1 * std::pow(2., -11);

  const dim3 threads(16, 16);
  using dca::util::ceilDiv;
  auto stream = util::getStream(thread_id, stream_id);

  auto padd = [](int size, int to) { return (size + to - 1) / to * to; };

  auto split = [&](const auto& m, auto& high, auto& low, int padx, int pady) {
    high.resizeNoCopy(std::make_pair(padd(m.nrRows(), padx), padd(m.nrCols(), pady)));
    low.resizeNoCopy(high.size());

    dim3 blocks(ceilDiv(high.nrRows(), int(threads.x)), ceilDiv(high.nrCols(), int(threads.y)));
    kernel::split<<<blocks, threads, 0, stream>>>(m, scale1, scale2, high, low);
  };

  auto& a_high = workspace[0];
  auto& a_low = workspace[1];
  auto& b_high = workspace[2];
  auto& b_low = workspace[3];

  split(a, a_high, a_low, 8, 8);
  split(b, b_high, b_low, 8, 8);

  auto handle = util::getHandle(thread_id, stream_id);
  const int m = a.nrRows();
  const int n = b.nrCols();
  const int k = a.nrCols();

  assert(n % 8 == 0);
  assert(m % 8 == 0);
  assert(k % 8 == 0);

  auto multiply = [&](float alpha, const auto& a, const auto& b, float beta, auto& c) {
    auto err = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, a.ptr(), CUDA_R_16F,
                            a.leadingDimension(), b.ptr(), CUDA_R_16F, b.leadingDimension(), &beta,
                            c.ptr(), CUDA_R_32F, c.leadingDimension(), CUDA_R_32F,
                            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    assert(err == CUBLAS_STATUS_SUCCESS);
  };

  // c <- beta* c + alpha * (a_high * b_high) * scale1**2
  const auto alpha_11 = alpha * scale1 * scale1;
  multiply(alpha_11, a_high, b_high, beta, c);

  // c += alpha * (a_high * b_low) * scale1 * scale2
  const auto alpha_12 = alpha * scale2 * scale1;
  multiply(alpha_12, a_high, b_low, 1., c);

  // c += alpha * (a_low * b_high) * scale1 * scale2
  multiply(alpha_12, a_low, b_high, 1., c);
}

}  // namespace blas
}  // namespace linalg
}  // namespace dca
