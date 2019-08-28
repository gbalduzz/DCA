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

#include "dca/linalg/matrix.hpp"
#include "dca/linalg/matrix_view.hpp"

namespace dca {
namespace linalg {
namespace blas {
// dca::linalg::blas::

// TODO: create in place version maybe?
void tensorcoreGemm(float alpha, const MatrixView<float, GPU>& a, const MatrixView<float, GPU>& b,
                    std::array<Matrix<__half, GPU>, 4>& workspace, float beta,
                    MatrixView<float, GPU> c, int thread_id = 0, int stream_id = 0);

inline void tensorcoreGemm(const MatrixView<float, GPU>& a, const MatrixView<float, GPU>& b,
                           std::array<Matrix<__half, GPU>, 4>& workspace, MatrixView<float, GPU> c,
                           int thread_id = 0, int stream_id = 0) {
  return tensorcoreGemm(1., a, b, workspace, 0., c, thread_id, stream_id);
}

}  // namespace blas
}  // namespace linalg
}  // namespace dca

#endif  // DCA_LINALG_BLAS_TENSORCORE_GEMM_HPP
