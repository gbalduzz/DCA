// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file tests reductions.hpp.

#include "dca/linalg/util/reductions.hpp"

#include <random>
#include <gtest/gtest.h>

#include "dca/linalg/matrix.hpp"
#include "dca/linalg/util/cuda_event.hpp"
#include "dca/linalg/util/cuda_stream.hpp"

using dca::linalg::Matrix;
using dca::linalg::Vector;
using dca::linalg::GPU;
using dca::linalg::CPU;

TEST(ReductionsTest, RandomInput) {
  Matrix<float, CPU> m_host(2047);

  std::mt19937_64 rng(0);
  std::normal_distribution<float> distro(0, 10);

  float max_abs = 0;
  for (int j = 0; j < m_host.nrCols(); ++j)
    for (int i = 0; i < m_host.nrRows(); ++i) {
      m_host(i, j) = distro(rng);
      max_abs = std::max(std::abs(m_host(i, j)), max_abs);
    }

  Matrix<float, GPU> m_dev(m_host);
  Vector<float, GPU> wp;
  dca::linalg::util::CudaStream stream;
  dca::linalg::util::CudaEvent start, end;

  start.record(stream);
  const float* res_dev = dca::linalg::util::reduceAbsMatrix<float>(m_dev, wp, stream);
  end.record(stream);

  float res_host;
  cudaMemcpy(&res_host, res_dev, sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_NEAR(max_abs, res_host, 5e-7);

  const float time = dca::linalg::util::elapsedTime(end, start);
  const float perf = m_dev.nrCols() * m_dev.nrRows() * sizeof(float) / time;

  std::cout << "Time: " << time << "\nPerf " << perf * 1e-9 << " GB/s" << std::endl;
}

TEST(ReductionsTest, CornerCase) {
  Matrix<float, GPU> m_dev(0);
  Vector<float, GPU> wp;
  dca::linalg::util::CudaStream stream;
  const float* res_dev = dca::linalg::util::reduceAbsMatrix<float>(m_dev, wp, stream);

  float res_host;
  cudaMemcpy(&res_host, res_dev, sizeof(float), cudaMemcpyDeviceToHost);

  EXPECT_EQ(0, res_host);
}
