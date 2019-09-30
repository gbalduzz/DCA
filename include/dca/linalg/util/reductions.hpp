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

#ifndef DCA_INCLUDE_DCA_LINALG_UTIL_REDUCTIONS_HPP
#define DCA_INCLUDE_DCA_LINALG_UTIL_REDUCTIONS_HPP

#include <dca/linalg/matrix_view.hpp>
#include <dca/linalg/vector.hpp>

namespace dca {
namespace linalg {
namespace util {
// dca::linalg::util::

// Compute the maximum entry of m in absolute value.
// Out: workspace.
// Returns: device pointer to the result.
// Postcondition: the result is the first element of workspace.
template <class T>
T* reduceAbsMatrix(const MatrixView<T, GPU>& m, Vector<T, GPU>& workspace, cudaStream_t stream);

}  // namespace util
}  // namespace linalg
}  // namespace dca

#endif  // DCA_INCLUDE_DCA_LINALG_UTIL_REDUCTIONS_HPP
