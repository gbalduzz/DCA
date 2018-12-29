// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This file provides allocators with pinned or mapped memory usable with std::vector.

#ifndef DCA_LINALG_UTIL_ALLOCATORS_MANAGED_ALLOCATOR_HPP
#define DCA_LINALG_UTIL_ALLOCATORS_MANAGED_ALLOCATOR_HPP

#ifndef DCA_HAVE_CUDA
#error "This file requires CUDA support."
#endif

#include <vector>
#include <cuda_runtime.h>

#include "dca/linalg/util/error_cuda.hpp"

namespace dca {
namespace linalg {
namespace util {
// dca::linalg::util::

template <typename T>
class ManagedAllocator : public std::allocator<T> {
public:
  void setStream(cudaStream_t stream) {
    stream_ = stream;
  }

protected:
  ManagedAllocator() {
    cudaGetDevice(&prefetch_device_);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, prefetch_device_);
    if (!properties.pageableMemoryAccess)  // No prefetch possible.
      prefetch_device_ = -1;
  }

  T* allocate(std::size_t n) {
    if (n == 0)
      return nullptr;

    T* ptr = nullptr;
    cudaError_t ret = cudaMallocManaged((void**)&ptr, n * sizeof(T));

    if (ret != cudaSuccess) {
      printErrorMessage(ret, __FUNCTION__, __FILE__, __LINE__,
                        "\t Managed size requested : " + std::to_string(n * sizeof(T)));
      throw(std::bad_alloc());
    }

    if (stream_) {
      cudaStreamAttachMemAsync(stream_, ptr, n * sizeof(T));
      if (prefetch_device_ >= 0)
        cudaMemPrefetchAsync(ptr, n * sizeof(T), prefetch_device_, stream_);
    }
    return ptr;
  }

  void deallocate(T*& ptr, std::size_t /*n*/ = 0) noexcept {
    cudaError_t ret = cudaFree(ptr);
    if (ret != cudaSuccess) {
      printErrorMessage(ret, __FUNCTION__, __FILE__, __LINE__);
      std::terminate();
    }
    ptr = nullptr;
  }

private:
  int prefetch_device_;
  cudaStream_t stream_ = nullptr;
};

}  // util
}  // linalg
}  // dca

#endif  // DCA_LINALG_UTIL_ALLOCATORS_MANAGED_ALLOCATOR_HPP
