// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file tests thread_pool.hpp.

#include "dca/parallel/stdthread/thread_pool/thread_pool.hpp"

#include <numeric>

#include "gtest/gtest.h"

TEST(ThreadPoolTest, Enqueue) {
  const int n_items = 9;
  const int n_threads = 4;
  std::vector<int> input(n_items);
  std::vector<int> output(n_items, 0);
  std::iota(input.begin(), input.end(), 0);

  auto workload = [](const int id, const std::vector<int>& inp, std::vector<int>& out) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    out[id] = inp[id] * inp[id];
  };

  const int n_immediate_checks = 4;
  {
    dca::parallel::ThreadPool pool(n_threads);
    EXPECT_EQ(n_threads, pool.size());

    auto task = std::bind(workload, std::placeholders::_1, std::ref(input), std::ref(output));
    std::vector<std::future<void>> futures;
    for (int id = 0; id < n_items; ++id)
      futures.emplace_back(pool.enqueue(task, id));

    // Check the synchronization with futures.
    for (int id = 0; id < n_immediate_checks; ++id) {
      futures[id].wait();
      EXPECT_EQ(input[id] * input[id], output[id]);
    }
  }

  // Check that the other tasks finished before the pool is destroyed.
  for (int id = n_immediate_checks; id < n_items; ++id)
    EXPECT_EQ(input[id] * input[id], output[id]);
}

TEST(ThreadPoolTest, DefualtConstructor) {
  const std::size_t default_threads = std::thread::hardware_concurrency();
  dca::parallel::ThreadPool pool;
  EXPECT_EQ(default_threads, pool.size());
  EXPECT_EQ(default_threads, dca::parallel::ThreadPool::get_instance().size());
}
