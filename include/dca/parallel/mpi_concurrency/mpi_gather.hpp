// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This class provides an interface to allgather functions with MPI.

#ifndef DCA_PARALLEL_MPI_CONCURRENCY_MPI_GATHER_HPP
#define DCA_PARALLEL_MPI_CONCURRENCY_MPI_GATHER_HPP

#include <vector>

#include <mpi.h>

#include "dca/function/domains.hpp"
#include "dca/function/function.hpp"
#include "dca/parallel/mpi_concurrency/mpi_gang.hpp"
#include "dca/parallel/mpi_concurrency/mpi_type_map.hpp"
#include "dca/util/integer_division.hpp"

namespace dca {
namespace parallel {
// dca::parallel::

class MPIGather {
public:
  MPIGather() = default;

  template <class Scalar, class DmnIn, class DmnOut, class Gang>
  void allgather(const func::function<Scalar, DmnIn>& f_in, func::function<Scalar, DmnOut>& f_out,
                 const Gang& gang) const;

  template <class T, class Gang>
  void gather(const std::vector<T> &in, std::vector<T> &out, std::vector<int> &sizes, int root,
              const Gang &gang) const;

  template <class T, class Gang>
  void
  scatter(const std::vector<T> &in, std::vector<T> &out, const std::vector<int> &sizes, int root,
          const Gang &gang) const;
};

template <class Scalar, class DmnIn, class DmnOut, class Gang>
void MPIGather::allgather(const func::function<Scalar, DmnIn>& f_in,
                          func::function<Scalar, DmnOut>& f_out, const Gang& gang) const {
  std::vector<Scalar> gathered(f_in.size() * gang.get_size());

  MPI_Allgather(f_in.values(), f_in.size(), MPITypeMap<Scalar>::value(), gathered.data(),
                f_in.size(), MPITypeMap<Scalar>::value(), gang.get());

  if (f_out.size() < f_in.size() * (gang.get_size() - 1))
    throw(std::logic_error("Output function is too small."));
  if (f_out.size() > gathered.size())
    throw(std::logic_error("Output function is too large."));

  // TODO: move.
  std::copy_n(gathered.data(), f_out.size(), f_out.values());
}

template <class T, class Gang>
void
MPIGather::gather(const std::vector<T> &in, std::vector<T> &out, std::vector<int> &sizes, int root,
                  const Gang &gang) const {
  int size = in.size();
  sizes.resize(gang.get_size());

  MPI_Gather(&size, 1, MPI_INT, sizes.data(), 1, MPI_INT, root, gang.get());

  std::vector<int> displ(sizes.size());
  displ[0] = 0;
  int tot_size = sizes[0];
  for (int i = 1; i < displ.size(); ++i) {
    displ[i] = displ[i - 1] + sizes[i - 1];
    tot_size += sizes[i];
  }

  out.resize(tot_size);
  MPI_Gatherv(in.data(), in.size(), MPITypeMap<T>::value(), out.data(), sizes.data(), displ.data(),
              MPITypeMap<T>::value(), root, gang.get());
}

template <class T, class Gang>
void
MPIGather::scatter(const std::vector<T> &in, std::vector<T> &out, const std::vector<int> &sizes,
                   int root,
                   const Gang &gang) const {
  if (root == gang.get_id() && sizes.size() != gang.get_size())
    throw(std::logic_error("One size is needed per process."));

  int size;
  MPI_Scatter(sizes.data(), 1, MPI_INT, &size, 1, MPI_INT, root, gang.get());

  std::vector<int> displ;
  if (gang.get_id() == root) {
    displ.resize(sizes.size());
    displ[0] = 0;
    int tot_size = sizes[0];
    for (int i = 1; i < displ.size(); ++i) {
      displ[i] = displ[i - 1] + sizes[i - 1];
      tot_size += sizes[i];
    }

    if (in.size() != tot_size)
      throw(std::logic_error("Mismatch between local and global sizes."));
  }

  out.resize(size);
  MPI_Scatterv(in.data(), sizes.data(), displ.data(), MPITypeMap<T>::value(), out.data(), size,
               MPITypeMap<T>::value(), root, gang.get());
}

}  // namespace parallel
}  // namespace dca

#endif  // DCA_PARALLEL_MPI_CONCURRENCY_MPI_GATHER_HPP
