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
#include "dca/linalg/vector.hpp"
#include "dca/linalg/util/cuda_event.hpp"

namespace dca {
namespace linalg {
namespace blas {
// dca::linalg::blas::

class TensorcoreGemm {
public:
  TensorcoreGemm(unsigned calls_per_check = 1)
      : scales_dev_(4), scales_host_(4), calls_per_check_(calls_per_check) {
    workspace_ = std::make_shared<std::array<Matrix<__half, GPU>, 4>>();
  }

  void set_workspace(const std::shared_ptr<std::array<Matrix<__half, GPU>, 4>>& workspace) {
    workspace_ = workspace;
  }

  // Perform the sgemm C <= beta C + alpha A B with three half precision multiplications.
  void execute(float alpha, const MatrixView<float, GPU>& a, const MatrixView<float, GPU>& b,
               float beta, MatrixView<float, GPU> c, int thread_id = 0, int stream_id = 0);

  // Default alpha = 1 and beta = 0.
  void execute(const MatrixView<float, GPU>& a, const MatrixView<float, GPU>& b,
               MatrixView<float, GPU> c, int thread_id = 0, int stream_id = 0) {
    return execute(float(1.), a, b, float(0.), c, thread_id, stream_id);
  }

private:
  void computeScale(float* scales, const MatrixView<float, GPU>& m, cudaStream_t stream);

  std::shared_ptr<std::array<Matrix<__half, GPU>, 4>> workspace_;
  Vector<float, GPU> reduction_wp_;
  Vector<float, GPU> scales_dev_;
  Vector<float, CPU> scales_host_;
  util::CudaEvent scale_copied_;
  unsigned n_calls_ = 0;
  unsigned calls_per_check_;
};

}  // namespace blas
}  // namespace linalg
}  // namespace dca

#endif  // DCA_LINALG_BLAS_TENSORCORE_GEMM_HPP
