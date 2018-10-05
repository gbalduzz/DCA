// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Provides a method to create a directory for each supported system.
// Note that this functionality is available in ++17 filesystem methods.

#ifndef DCA_IO_UTIL_CREATE_DIRECTORY
#define DCA_IO_UTIL_CREATE_DIRECTORY

#if __cplusplus >= 201703L
#include <filesystem>
#endif
#include <string>

namespace dca {
namespace io {
namespace util {
// dca::io::util::

#if __cplusplus >= 201703L
void inline createDirectory(const std::string& name) {
  std::filesystem::create_directory(name);
}
#else

void inline createDirectory(const std::string& name) {
#if defined(_WIN64) || defined(_WIN32)
  const std::string cmd = "mkdir " + name;
#else
  const std::string cmd = "mkdir -p " + name;
#endif

  system(cmd.c_str());
}

#endif  // __cplusplus

}  // util
}  // io
}  // dca

#endif  // DCA_IO_UTIL_CREATE_DIRECTORY
