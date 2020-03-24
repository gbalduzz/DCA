// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file tests affinity.hpp.

#include "dca/parallel/stdthread/thread_pool/affinity.hpp"

#include <iostream>
#include <future>
#include <thread>

#include "gtest/gtest.h"

TEST(AffinityTest, All) {
  auto print = [](const auto& v) {
    for (auto x : v)
      std::cout << x << " ";
    std::cout << std::endl;
  };

  auto f_count = std::async(std::launch::async, [&]() { return dca::parallel::get_core_count(); });

  std::future<void> f_set = std::async(std::launch::async, [&]() {
    auto a = dca::parallel::get_affinity();
    std::cout << "Old affinity set: ";
    print(a);
    std::vector<int> new_set{1, 2, 4};
    dca::parallel::set_affinity(new_set);

    auto b = dca::parallel::get_affinity();
    EXPECT_EQ(new_set, b);
  });

  EXPECT_EQ(std::thread::hardware_concurrency(), f_count.get());
  f_set.get();
}
