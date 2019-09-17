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

#include "dca/io/hdf5/hdf5_reader.hpp"
#include "dca/linalg/matrixop.hpp"

using dca::linalg::Matrix;
using dca::linalg::GPU;
using dca::linalg::CPU;

void testProduct(const Matrix<float, CPU>& a, const Matrix<float, CPU>& b) {
  ASSERT_EQ(a.nrCols(), b.nrRows());
  Matrix<float, CPU> c(std::make_pair(a.nrRows(), b.nrCols()));

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

TEST(TensorcoreGemmTest, RandomMatrix) {
  Matrix<float, CPU> a(8), b(8);

  std::mt19937_64 rng(0);
  std::uniform_real_distribution<float> distro(-1, 1);

  auto fill = [&](auto& m) {
    for (int j = 0; j < m.nrCols(); ++j)
      for (int i = 0; i < m.nrRows(); ++i)
        m(i, j) = distro(rng);
  };

  fill(a);
  fill(b);

  testProduct(a, b);
}

TEST(TensorcoreGemmTest, DCAMatrix) {
  Matrix<float, CPU> a("a"), b("b");
  const std::string folder = DCA_SOURCE_DIR "/test/unit/linalg/blas/";
  const std::vector<std::string> inames{folder + "input_matrices1.hdf5",
                                        folder + "input_matrices2.hdf5"};
  dca::io::HDF5Reader reader;

  for (auto iname : inames) {
    reader.open_file(iname);
    reader.execute("a", a);
    reader.execute("b", b);
    reader.close_file();

    testProduct(a, b);
  }
}
