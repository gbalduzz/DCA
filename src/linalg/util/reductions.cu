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

namespace dca {
namespace linalg {
namespace util {
namespace kernels {
// dca::linalg::util::kernels::

template <class T>
__device__ T inline reduceWarp(T datum) {

  datum = max(datum, __shfl_down_sync(0xffffffff, datum, 16));
  datum = max(datum, __shfl_down_sync(0x0000ffff, datum, 8));
  datum = max(datum, __shfl_down_sync(0x000000ff, datum, 4));
  datum = max(datum, __shfl_down_sync(0x0000000f, datum, 2));
  datum = max(datum, __shfl_down_sync(0x00000003, datum, 1));

  return datum;
}

template <int block_size, class T>
__device__ void inline reduceShared(volatile T* sdata, const int tid) {
  if (block_size >= 1024) {
    if (tid < 512)
      sdata[tid] = max(sdata[tid], sdata[tid + 512]);
    __syncthreads();
  }
  if (block_size >= 512) {
    if (tid < 256)
      sdata[tid] = max(sdata[tid], sdata[tid + 256]);
    __syncthreads();
  }
  if (block_size >= 256) {
    if (tid < 128)
      sdata[tid] = max(sdata[tid], sdata[tid + 128]);
    __syncthreads();
  }
  if (block_size >= 128) {
    if (tid < 64)
      sdata[tid] = max(sdata[tid], sdata[tid + 64]);
    __syncthreads();
  }
  if (block_size >= 64) {
    if (tid < 32)
      sdata[tid] = max(sdata[tid], sdata[tid + 32]);
    __syncthreads();
  }

  assert(block_size >= 32);
  if (tid < 32)
    sdata[tid] = reduceWarp(sdata[tid]);
}

// TODO: generalize with arbitrary operator.
template <unsigned block_size, class T>
void __global__ reduceSingleBlock(T* data) {
  assert(blockIdx.x == 0);
  extern __shared__ T sdata[];
  const int tid = threadIdx.x;

  sdata[tid] = data[tid];
  __syncthreads();

  reduceShared<block_size>(sdata, tid);

  data[0] = sdata[0];
}

template <unsigned block_size, class T>
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

  reduceShared<block_size>(sdata, tid);

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

  const unsigned div = n / threads;
  const unsigned blocks = div > 64 ? 128 : div > 1 ? 64 : 1;

  workspace.resizeNoCopy(blocks);

  kernels::reduce2DAbsArray<threads>
      <<<blocks, threads, threads * sizeof(T), stream>>>(m, workspace.ptr());

  switch (blocks) {
    case 1:
      return workspace.ptr();
    case 64:
      kernels::reduceSingleBlock<64><<<1, 64, 64 * sizeof(T), stream>>>(workspace.ptr());
      break;
    case 128:
      kernels::reduceSingleBlock<128><<<1, 128, 128 * sizeof(T), stream>>>(workspace.ptr());
      break;
    default:
      throw(std::logic_error("Case not supported."));
  }

  return workspace.ptr();
}

template float* reduceAbsMatrix(const MatrixView<float, GPU>&, Vector<float, GPU>&, cudaStream_t);

}  // namespace util
}  // namespace linalg
}  // namespace dca
