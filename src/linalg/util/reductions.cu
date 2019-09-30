// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// Implements a reduction on a 2D matrix.

#include "dca/linalg/util/reductions.hpp"

#include <cuda.h>

#include "dca/linalg/util/error_cuda.hpp"
#include "dca/util/integer_division.hpp"

namespace dca {
namespace linalg {
namespace util {
namespace kernels {
// dca::linalg::util::kernels::

template <class T>
__device__ void inline reduceWarp(volatile T* sdata, const int tid) {
  sdata[tid] = max(sdata[tid], sdata[tid + 32]);
  sdata[tid] = max(sdata[tid], sdata[tid + 16]);
  sdata[tid] = max(sdata[tid], sdata[tid + 8]);
  sdata[tid] = max(sdata[tid], sdata[tid + 4]);
  sdata[tid] = max(sdata[tid], sdata[tid + 2]);
  sdata[tid] = max(sdata[tid], sdata[tid + 1]);
}

template <class T>
__device__ void inline reduceShared(volatile T* sdata, const int tid) {
  for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {  // 32 is the warp size.
    if (tid < stride)
      sdata[tid] = max(sdata[tid], sdata[tid + stride]);
    __syncthreads();
  }
  if (tid < 32)
    reduceWarp(sdata, tid);
}

// TODO: generalize with arbitrary operator.
template <class T>
void __global__ reduceSingleBlock(T* data) {
  assert(blockIdx.x == 0);
  extern __shared__ T sdata[];
  const int tid = threadIdx.x;

  sdata[tid] = data[tid];
  __syncthreads();

  reduceShared(sdata, tid);

  data[0] = sdata[0];
}

template <class T>
void __global__ reduce2DAbsArray(const MatrixView<T, GPU> m, T* out) {
  extern __shared__ T sdata[];
  const int tid = threadIdx.x;
  sdata[tid] = 0;

  const int n = m.nrCols() * m.nrRows();
  const int grid_stride = blockDim.x * gridDim.x;
  for (int inp_idx = tid + blockIdx.x * blockDim.x; inp_idx < n; inp_idx += grid_stride) {
    const int j = inp_idx / m.nrRows();
    const int i = inp_idx - j * m.nrRows();

    sdata[tid] = max(sdata[tid], abs(m(i, j)));
  }
  __syncthreads();

  reduceShared(sdata, tid);

  if (tid == 0)
    out[blockIdx.x] = sdata[0];
}

}  // namespace kernels
// dca::linalg::util::

template <class T>
T* reduceAbsMatrix(const MatrixView<T, GPU>& m, Vector<T, GPU>& workspace, cudaStream_t stream) {
  constexpr unsigned threads = 1024;

  const unsigned n = m.nrRows() * m.nrCols();
  if (!n) {
    workspace.resizeNoCopy(1);
    cudaMemsetAsync(workspace.ptr(), 0, sizeof(float), stream);
    return workspace.ptr();
  }

  const unsigned blocks = std::min(128u, dca::util::ceilDiv(n, threads));
  workspace.resizeNoCopy(blocks);

  kernels::reduce2DAbsArray<<<blocks, threads, threads * sizeof(T), stream>>>(m, workspace.ptr());

  const unsigned new_threads = blocks;
  kernels::reduceSingleBlock<<<1, new_threads, new_threads * sizeof(T), stream>>>(workspace.ptr());
  return workspace.ptr();
}

template float* reduceAbsMatrix(const MatrixView<float, GPU>&, Vector<float, GPU>&, cudaStream_t);

}  // namespace util
}  // namespace linalg
}  // namespace dca
