// Copyright (C) 2009-2016 ETH Zurich
// Copyright (C) 2007?-2016 Center for Nanophase Materials Sciences, ORNL
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
// See CITATION.txt for citation guidelines if you use this code for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file tests magma_batched_gemm.hpp.

#include "dca/linalg/util/magma_batched_gemm.hpp"

#include "gtest/gtest.h"

#include "dca/linalg/matrix.hpp"
#include "dca/linalg/matrixop.hpp"
#include "dca/linalg/util/magma_queue.hpp"
#include "dca/linalg/util/util_cublas.hpp"

TEST(MagmaBatchedGemmTest, Batched) {
  dca::linalg::util::initializeMagma();

  const std::pair<int, int> size_a(3, 4);
  dca::linalg::Matrix<float, dca::linalg::CPU> a1(size_a), a2(size_a);
  const std::pair<int, int> size_b(4, 2);
  dca::linalg::Matrix<float, dca::linalg::CPU> b1(size_b), b2(size_b);

  // Set up an arbitrary input.
  for (int j = 0; j < a1.nrCols(); ++j)
    for (int i = 0; i < a1.nrRows(); ++i) {
      a1(i, j) = 3. * i - 2. * j;
      a2(i, j) = 5. * i + 1. * j;
    }
  for (int j = 0; j < b1.nrCols(); ++j)
    for (int i = 0; i < b1.nrRows(); ++i) {
      b1(i, j) = i * i - 2. * j;
      b2(i, j) = -i * i + 1. * j;
    }

  dca::linalg::Matrix<float, dca::linalg::GPU> a1_dev(a1), a2_dev(a2);
  dca::linalg::Matrix<float, dca::linalg::GPU> b1_dev(b1), b2_dev(b2);
  const std::pair<int, int> size_c(3, 2);
  dca::linalg::Matrix<float, dca::linalg::GPU> c1_dev(size_c), c2_dev(size_c);

  dca::linalg::util::MagmaQueue queue;
  dca::linalg::util::MagmaBatchedGemm<float> plan(queue);

  plan.resize(2);

  plan.set_a(0, a1_dev.ptr());
  plan.set_a(1, a2_dev.ptr());
  plan.set_b(0, b1_dev.ptr());
  plan.set_b(1, b2_dev.ptr());
  plan.set_c(0, c1_dev.ptr());
  plan.set_c(1, c2_dev.ptr());

  // Only two multiplications are expected.
  EXPECT_DEBUG_DEATH(plan.set_a(2, a1.ptr()), ".*");

  const float alpha = 1.;
  const float beta = 0;
  plan.executeBatched('N', 'N', c1_dev.nrRows(), c2_dev.nrCols(), a1_dev.nrCols(), alpha, beta,
                      a1_dev.leadingDimension(), b1_dev.leadingDimension(),
                      c1_dev.leadingDimension());

  dca::linalg::Matrix<float, dca::linalg::CPU> c1(c1_dev), c2(c2_dev);

  // Compute the expected result.
  dca::linalg::Matrix<float, dca::linalg::CPU> c1_test(size_c), c2_test(size_c);
  dca::linalg::matrixop::gemm('N', 'N', alpha, a1, b1, beta, c1_test);
  dca::linalg::matrixop::gemm('N', 'N', alpha, a2, b2, beta, c2_test);

  for (int j = 0; j < c1.nrCols(); ++j)
    for (int i = 0; i < c1.nrRows(); ++i) {
      EXPECT_NEAR(c1_test(i, j), c1(i, j), 3e-7);
      EXPECT_NEAR(c2_test(i, j), c2(i, j), 3e-7);
    }
}

TEST(MagmaBatchedGemmTest, VBatched) {
  //  dca::linalg::util::initializeMagma();

  dca::linalg::Matrix<double, dca::linalg::CPU> a1(3), a2(4);
  dca::linalg::Matrix<double, dca::linalg::CPU> b1(3), b2(4);

  // Set up an arbitrary input.
  for (int j = 0; j < a1.nrCols(); ++j)
    for (int i = 0; i < a1.nrRows(); ++i)
      a1(i, j) = 3. * i - 2. * j;
  for (int j = 0; j < a2.nrCols(); ++j)
    for (int i = 0; i < a2.nrRows(); ++i)
      a2(i, j) = 10. * +1. * j;
  for (int j = 0; j < b1.nrCols(); ++j)
    for (int i = 0; i < b1.nrRows(); ++i)
      b1(i, j) = i * i - 2. * j;
  for (int j = 0; j < b2.nrCols(); ++j)
    for (int i = 0; i < b2.nrRows(); ++i)
      b2(i, j) = -i * i + 1. * j;

  dca::linalg::Matrix<double, dca::linalg::GPU> a1_dev(a1), a2_dev(a2);
  dca::linalg::Matrix<double, dca::linalg::GPU> b1_dev(b1), b2_dev(b2);
  dca::linalg::Matrix<double, dca::linalg::GPU> c1_dev(3), c2_dev(4);

  dca::linalg::util::MagmaQueue queue;
  dca::linalg::util::MagmaBatchedGemm<double> plan(2, queue);

  plan.set_a(0, a1_dev.ptr());
  plan.set_a(1, a2_dev.ptr());
  plan.set_lda(0, a1_dev.leadingDimension());
  plan.set_lda(1, a2_dev.leadingDimension());
  plan.set_b(0, b1_dev.ptr());
  plan.set_b(1, b2_dev.ptr());
  plan.set_ldb(0, b1_dev.leadingDimension());
  plan.set_ldb(1, b2_dev.leadingDimension());
  plan.set_c(0, c1_dev.ptr());
  plan.set_c(1, c2_dev.ptr());
  plan.set_ldc(0, c1_dev.leadingDimension());
  plan.set_ldc(1, c2_dev.leadingDimension());

  plan.set_n(0, 3), plan.set_m(0, 3), plan.set_k(0, 3);
  plan.set_n(1, 4), plan.set_m(1, 4), plan.set_k(1, 4);

  plan.executeVBatched('T', 'N');

  dca::linalg::Matrix<double, dca::linalg::CPU> c1(c1_dev), c2(c2_dev);

  // Compute the expected result.
  dca::linalg::Matrix<double, dca::linalg::CPU> c1_test(c1_dev.size()), c2_test(c2_dev.size());
  dca::linalg::matrixop::gemm('T', 'N', a1, b1, c1_test);
  dca::linalg::matrixop::gemm('T', 'N', a2, b2, c2_test);

  for (int j = 0; j < c1.nrCols(); ++j)
    for (int i = 0; i < c1.nrRows(); ++i)
      EXPECT_NEAR(c1_test(i, j), c1(i, j), 3e-16);
  for (int j = 0; j < c2.nrCols(); ++j)
    for (int i = 0; i < c2.nrRows(); ++i)
      EXPECT_NEAR(c2_test(i, j), c2(i, j), 3e-16);
}
