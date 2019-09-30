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

// TODO: generalize with arbitrary operator.
// TODO: Optimize according to https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <class T>
void __global__ reduceArray(const T* input, T* out, int n) {
  extern __shared__ T sdata[];

  const int tid = threadIdx.x;
  const int inp_idx = tid + blockIdx.x * blockDim.x;

  sdata[tid] = inp_idx < n ? input[inp_idx] : 0;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {  // 32 is the warp size.
    if (tid < stride)
      sdata[tid] = max(sdata[tid], sdata[tid + stride]);
    __syncthreads();
  }
  if (tid < 32)
    reduceWarp(sdata, tid);

  if (tid == 0)
    out[blockIdx.x] = sdata[0];
}

template <class T>
void __global__ reduce2DAbsArray(const MatrixView<T, GPU> m, T* out) {
  extern __shared__ T sdata[];

  const int tid = threadIdx.x;
  const int inp_idx = tid + blockIdx.x * blockDim.x;

  const int j = inp_idx / m.nrRows();
  const int i = inp_idx - j * m.nrRows();
  const int n = m.nrCols() * m.nrRows();

  sdata[tid] = inp_idx < n ? abs(m(i, j)) : 0;
  __syncthreads();

  for (unsigned stride = blockDim.x / 2; stride > 32; stride >>= 1) {
    if (tid < stride)  //
      sdata[tid] = max(sdata[tid], sdata[tid + stride]);
    __syncthreads();
  }
  if (tid < 32)
    reduceWarp(sdata, tid);

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

  unsigned blocks = dca::util::ceilDiv(n, threads);
  workspace.resizeNoCopy(blocks +
                         dca::util::ceilDiv(blocks, threads));  // Reserve space for next step.

  kernels::reduce2DAbsArray<<<blocks, threads, threads * sizeof(T), stream>>>(m, workspace.ptr());
  checkRC(cudaPeekAtLastError());

  T* in = workspace.ptr(0);
  T* out = workspace.ptr(blocks);
  while (true) {
    const unsigned new_blocks = dca::util::ceilDiv(blocks, threads);
    kernels::reduceArray<<<new_blocks, threads, threads * sizeof(T), stream>>>(in, out, blocks);
    checkRC(cudaPeekAtLastError());

    if (new_blocks == 1) {
      return out;
    }

    std::swap(in, out);
    blocks = new_blocks;
  }
}

template float* reduceAbsMatrix(const MatrixView<float, GPU>&, Vector<float, GPU>&, cudaStream_t);

}  // namespace util
}  // namespace linalg
}  // namespace dca
