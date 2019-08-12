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
  void gather(const std::vector<T>& in, std::vector<T>& out, int root, const Gang& gang) const;

  template <class T, class Gang>
  void scatter(const std::vector<T>& in, std::vector<T>& out, int root, const Gang& gang) const;
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
void MPIGather::gather(const std::vector<T>& in, std::vector<T>& out, int root,
                       const Gang& gang) const {
  if (gang.get_id() == root)
    out.resize(in.size() * gang.get_size());
  MPI_Gather(in.data(), in.size(), MPITypeMap<T>::value(), out.data(), in.size(),
             MPITypeMap<T>::value(), root, gang.get());
}

template <class T, class Gang>
void MPIGather::scatter(const std::vector<T>& in, std::vector<T>& out, int root,
                        const Gang& gang) const {
  int local_size;

  if (gang.get_id() == root) {
    local_size = in.size() / gang.get_size();

    if (local_size * gang.get_size() != in.size())
      throw(
          std::logic_error("The size of the scattered vector is not a multiple of the gang size."));
  }

  MPI_Bcast(&local_size, 1, MPI_INT, root, gang.get());

  out.resize(local_size);
  MPI_Scatter(in.data(), local_size, MPITypeMap<T>::value(), out.data(), local_size,
              MPITypeMap<T>::value(), root, gang.get());
}

}  // namespace parallel
}  // namespace dca

#endif  // DCA_PARALLEL_MPI_CONCURRENCY_MPI_GATHER_HPP
