// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file tests mpi_collective_sum.hpp.
//
// This test only passes for 8 MPI processes.

#include "dca/parallel/mpi_concurrency/mpi_gather.hpp"

#include <memory>

#include "gtest/gtest.h"

#include "dca/parallel/mpi_concurrency/mpi_concurrency.hpp"
#include "dca/parallel/mpi_concurrency/mpi_gang.hpp"
#include "dca/function/function.hpp"
#include "dca/function/domains/dmn.hpp"
#include "dca/function/domains/dmn_0.hpp"
#include "dca/function/domains/dmn_variadic.hpp"
#include "dca/function/domains/local_domain.hpp"
#include "dca/testing/minimalist_printer.hpp"

std::unique_ptr<dca::parallel::MPIConcurrency> concurrency;

TEST(MPIGatherTest, GatherLocalDmn) {
  using Dmn1 = dca::func::dmn<4>;
  using Dmn2 = dca::func::dmn<10>;
  using LocalDmn = dca::func::LocalDomain<Dmn2>;

  std::vector<int> val1(Dmn1::get_size());
  for (int i = 0; i < val1.size(); ++i)
    val1[i] = i;
  Dmn1::set_elements(val1);

  std::vector<int> val2(Dmn2::get_size());
  for (int i = 0; i < val2.size(); ++i)
    val2[i] = i;
  Dmn2::set_elements(val2);

  dca::parallel::MPIGang gang(*concurrency, 3);

  LocalDmn::initialize(gang);

  dca::func::function<int, dca::func::dmn_variadic<dca::func::dmn_0<Dmn1>, dca::func::dmn_0<LocalDmn>>> local_f;
  dca::func::function<int, dca::func::dmn_variadic<dca::func::dmn_0<Dmn1>, dca::func::dmn_0<Dmn2>>> f;

  for (int i2 = 0; i2 < LocalDmn::get_physical_size(); ++i2)
    for (int i1 = 0; i1 < Dmn1::get_size(); ++i1)
      local_f(i1, i2) = Dmn1::get_elements()[i1] * LocalDmn::get_elements()[i2];

  concurrency->allgather(local_f, f, gang);

  for (int i2 = 0; i2 < Dmn2::get_size(); ++i2)
    for (int i1 = 0; i1 < Dmn1::get_size(); ++i1)
      EXPECT_EQ(Dmn1::get_elements()[i1] * Dmn2::get_elements()[i2], f(i1, i2));
}

TEST(MPIGatherTest, GatherVector) {
  const int id = concurrency->id();
  std::vector<int> local_data(id + 1);
  for (int i = 0; i < local_data.size(); ++i)
    local_data[i] = id * (id + 1) / 2 + i;  // Generate distributed 0,1,..n sequence.

  std::vector<int> sizes, global_data;
    concurrency->gather(local_data, global_data, sizes, 0, *concurrency);

  if (id == 0) {
    EXPECT_EQ(concurrency->get_size(), sizes.size());
    for (int i = 0; i < sizes.size(); ++i)
      EXPECT_EQ(i + 1, sizes[i]);

    EXPECT_EQ((concurrency->get_size()) * (concurrency->get_size() + 1) / 2, global_data.size());
    for (int i = 0; i < global_data.size(); ++i)
      EXPECT_EQ(i, global_data[i]);
  }
}

TEST(MPIGatherTest, ScatterVector) {
  std::vector<int> global_data;
  std::vector<int> sizes;

  if (concurrency->id() == 0) {
    sizes.resize(concurrency->get_size());
    global_data.resize(concurrency->get_size() * (concurrency->get_size() + 1) / 2);

    for (int i = 0; i < global_data.size(); ++i)
      global_data[i] = i;
    for (int i = 0; i < sizes.size(); ++i)
      sizes[i] = i + 1;
  }

  std::vector<int> local_data;
    concurrency->scatter(global_data, local_data, sizes, 0, *concurrency);

  EXPECT_EQ(sizes[concurrency->id()], local_data.size());

  const int start = concurrency->id() * (concurrency->id() + 1) / 2;
  for (int i = 0; i < local_data.size(); ++i)
    EXPECT_EQ(start + i, local_data[i]);
}

int main(int argc, char** argv) {
  int result = 0;

  concurrency = std::make_unique<dca::parallel::MPIConcurrency>(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  if (concurrency->id() != 0) {
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new dca::testing::MinimalistPrinter);
  }

  result = RUN_ALL_TESTS();

  concurrency.release();
  return result;
}
