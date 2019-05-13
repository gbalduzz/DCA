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
//#include "dca/parallel/stdthread/thread_pool/thread_pool.hpp"

#include <iostream>
#include <future>

#include "gtest/gtest.h"

TEST(AffinityTest, All) {
  auto print = [](const auto& v) {
    for (auto x : v)
      std::cout << x << " ";
    std::cout << std::endl;
  };
  auto equal = [](const auto& a, const auto& b) {
    if (a.size() != b.size())
      return false;
    for (int i = 0; i < a.size(); ++i)
      if (a[i] != b[i])
        return false;
    return true;
  };

  std::future<void> f = std::async(std::launch::async, [&]() {
    auto a = get_affinity();
    std::cout << "Old affinity set: ";
    print(a);
    std::vector<int> new_set{1, 2, 4};
    set_affinity(new_set);

    auto b = get_affinity();
    EXPECT_TRUE(equal(new_set, b));
  });

  f.get();
}

// TODO: make general.
// TEST(AffinityTest, Count) {
//  EXPECT_EQ(8, get_core_count());
//}
