// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file tests tensorcore_gemm.hpp.

#include "dca/linalg/blas/tensorcore_gemm.hpp"

#include <random>
#include <gtest/gtest.h>

#include "dca/linalg/matrixop.hpp"

TEST(TensorcoreGemmTest, Product) {
  using dca::linalg::Matrix;
  using dca::linalg::GPU;
  using dca::linalg::CPU;

  Matrix<float, CPU> a(8);
  Matrix<float, CPU> b(8);
  Matrix<float, CPU> c(8);

  std::mt19937_64 rng(0);
  std::uniform_real_distribution<float> distro(-1, 1);

  auto fill = [&](auto& m) {
    for (int j = 0; j < m.nrCols(); ++j)
      for (int i = 0; i < m.nrRows(); ++i)
        m(i, j) = distro(rng);
  };

  fill(a);
  fill(b);

  // Compute expected result.
  dca::linalg::matrixop::gemm(a, b, c);

  // Compute with tensorcores.
  Matrix<float, GPU> a_dev(a);
  Matrix<float, GPU> b_dev(b);
  Matrix<float, GPU> c_dev(c.size());
  std::array<Matrix<__half, GPU>, 4> workspace;

  dca::linalg::blas::tensorcoreGemm(a_dev, b_dev, workspace, c_dev);
  Matrix<float, CPU> c_tensor(c_dev);

  for (int j = 0; j < c.nrCols(); ++j)
    for (int i = 0; i < c.nrRows(); ++i)
      EXPECT_NEAR(c(i, j), c_tensor(i, j), 1e-5);
}
