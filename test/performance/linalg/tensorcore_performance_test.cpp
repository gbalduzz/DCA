// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
//  See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Performance test for a matrix-matrix multiplication using the tensorcores.

#include "dca/linalg/blas/tensorcore_gemm.hpp"

#include <random>

#include "dca/linalg/matrixop.hpp"
#include "dca/linalg/util/cuda_event.hpp"
#include "dca/math/util/vector_operations.hpp"

int main(int argc, char** argv) {
  const int matrix_size = argc > 1 ? std::atoi(argv[1]) : 2000;
  const bool redo_reduction = argc > 2 ? std::atoi(argv[2]) : false;
  const int n_iter = 10;

  dca::linalg::Matrix<float, dca::linalg::CPU> a(matrix_size), b(matrix_size);
  std::mt19937_64 rng;
  std::normal_distribution<float> distro_a(0, 1);
  std::normal_distribution<float> distro_b(1e3, 500);

  for (int j = 0; j < matrix_size; ++j)
    for (int i = 0; i < matrix_size; ++i) {
      a(i, j) = distro_a(rng);
      b(i, j) = distro_b(rng);
    }

  dca::linalg::Matrix<float, dca::linalg::GPU> a_dev(a), b_dev(b), c_dev(matrix_size);
  auto stream = dca::linalg::util::getStream(0, 0);

  auto time_function = [&](auto&& f) {
    dca::linalg::util::CudaEvent start, stop;
    std::vector<double> perfs(n_iter);
    for (auto& perf : perfs) {
      start.record(stream);
      f();
      stop.record(stream);

      const auto time = dca::linalg::util::elapsedTime(stop, start);
      perf = std::pow(matrix_size, 3) * 2. / time;
    }

    auto [mean, err] = dca::math::util::meanAndErr(perfs);
    std::cout << "Performance: (" << mean * 1e-9 << " +- " << err * 1e-9 << ") GFLOP/s.";
  };

  std::cout << "Single precision CUBLAS gemm" << std::endl;
  time_function([&] { dca::linalg::matrixop::gemm(a_dev, b_dev, c_dev, 0, 0); });

  std::cout << "\n\nTensorcore gemm" << std::endl;
  dca::linalg::blas::TensorcoreGemm t_gemm(redo_reduction ? 1 : n_iter);
  time_function([&] { t_gemm.execute(a_dev, b_dev, c_dev, 0, 0); });

  std::cout << std::endl;
}
