// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file implements a single precision matrix-matrix multiplication in terms of
// 3 half precision multiplications:

#ifndef DCA_HAVE_CUDA
#error "This file requires CUDA."
#endif

#ifndef DCA_LINALG_BLAS_TENSORCORE_GEMM_HPP
#define DCA_LINALG_BLAS_TENSORCORE_GEMM_HPP

#include <array>
#include <cuda_fp16.h>

#include "dca/linalg/matrixop.hpp"
#include "dca/linalg/matrix_view.hpp"

namespace dca {
namespace linalg {
namespace blas {
// dca::linalg::blas::

// Perform the sgemm C <= beta C + alpha A B with three half precision multiplications.
void tensorcoreGemm(float alpha, const MatrixView<float, GPU>& a, const MatrixView<float, GPU>& b,
                    std::array<Matrix<__half, GPU>, 4>& workspace, float beta,
                    MatrixView<float, GPU> c, int thread_id = 0, int stream_id = 0);

// Default alpha = 1 and beta = 0
inline void tensorcoreGemm(const MatrixView<float, GPU>& a, const MatrixView<float, GPU>& b,
                           std::array<Matrix<__half, GPU>, 4>& workspace, MatrixView<float, GPU> c,
                           int thread_id = 0, int stream_id = 0) {
  return tensorcoreGemm(1., a, b, workspace, 0., c, thread_id, stream_id);
}

// Non tensor core fallback for different types and devices.
template <typename Scalar, DeviceType device>
void tensorcoreGemm(Scalar alpha, const MatrixView<Scalar, device>& a,
                    const MatrixView<Scalar, device>& b,
                    std::array<Matrix<__half, device>, 4>& workspace, Scalar beta,
                    MatrixView<Scalar, device> c, int thread_id = 0, int stream_id = 0) {
  return matrixop::gemm(alpha, a, b, workspace, beta, c, thread_id, stream_id);
}
template <typename Scalar, DeviceType device>
void tensorcoreGemm(const MatrixView<Scalar, device>& a, const MatrixView<Scalar, device>& b,
                    std::array<Matrix<__half, device>, 4>& workspace, MatrixView<Scalar, device> c,
                    int thread_id = 0, int stream_id = 0) {
  return tensorcoreGemm(1., a, b, workspace, 0., c, thread_id, stream_id);
}

}  // namespace blas
}  // namespace linalg
}  // namespace dca

#endif  // DCA_LINALG_BLAS_TENSORCORE_GEMM_HPP
