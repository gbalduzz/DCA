// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE.txt for terms of usage.
//  See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Giovanni Balduzzi (gbalduzz@itp.phys.ethz.ch)
//
// This class organizes the MC walker in the CT-INT QMC purely on the CPU.

#ifndef DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_CTINT_WALKER_CPU_HPP
#define DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_CTINT_WALKER_CPU_HPP

#include "dca/phys/dca_step/cluster_solver/ctint/walker/ctint_walker_base.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "dca/linalg/linalg.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/walker/tools/d_matrix_builder.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/walker/tools/walker_methods.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/structs/solver_configuration.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/walker/tools/function_proxy.hpp"
#include "dca/phys/dca_step/cluster_solver/ctint/walker/tools/g0_interpolation.hpp"
#include "dca/util/integer_division.hpp"

namespace dca {
namespace phys {
namespace solver {
namespace ctint {
// dca::phys::solver::ctint::

template <class Parameters>
class CtintWalker<linalg::CPU, Parameters> : public CtintWalkerBase<Parameters> {
public:
  using this_type = CtintWalker<linalg::CPU, Parameters>;
  using BaseClass = CtintWalkerBase<Parameters>;
  using typename BaseClass::Rng;
  using typename BaseClass::Data;

public:
  CtintWalker(const Parameters& pars_ref, const Data& /*data*/, Rng& rng_ref, int id = 0);

  void initialize();

  void doSweep();

protected:
  void doStep();
  bool tryVertexInsert();
  bool tryVertexRemoval();

  void moveRemovalToEnd();
  void popBack(int delta_vertices);

  // Test handle.
  const auto& getM() const {
    return M_;
  }

  void initializeStep();

private:
  double insertionProbability(int delta_vertices);

  using Matrix = typename BaseClass::Matrix;
  using MatrixPair = typename BaseClass::MatrixPair;
  using MatrixView = typename linalg::MatrixView<double, linalg::CPU>;

  void applyInsertion(const MatrixPair& S, const MatrixPair& Q, const MatrixPair& R);

  double removalProbability();
  void applyRemoval();

  virtual void smallInverse(const MatrixView& in, MatrixView& out, int s);
  virtual void smallInverse(MatrixView& in_out, int s);

protected:
  using BaseClass::parameters_;
  using BaseClass::configuration_;
  using BaseClass::rng_;
  using BaseClass::d_builder_ptr_;
  using BaseClass::total_interaction_;
  using BaseClass::beta_;
  using BaseClass::possible_partners_;
  using BaseClass::sign_;
  using BaseClass::thread_id_;

  using BaseClass::thermalized_;
  using BaseClass::nb_steps_per_sweep_;
  using BaseClass::n_accepted_;

  using BaseClass::M_;

  // For testing purposes.
  using BaseClass::acceptance_prob_;
  std::array<double, 2> det_ratio_;

private:
  std::array<linalg::Matrix<double, linalg::CPU>, 2> S_, Q_, R_;
  // work spaces
  MatrixPair M_Q_;
  Matrix ws_dn_;
  linalg::Vector<double, linalg::CPU> v_work_;
  linalg::Vector<int, linalg::CPU> ipiv_;

  std::array<linalg::util::HostVector<int>, 2> matrix_removal_list_;
  std::array<linalg::util::HostVector<int>, 2> matrix_source_list_;
  std::vector<int> removal_list_;
};

template <class Parameters>
CtintWalker<linalg::CPU, Parameters>::CtintWalker(const Parameters& parameters_ref,
                                                  const Data& /*data*/, Rng& rng_ref, int id)
    : BaseClass(parameters_ref, rng_ref, id), det_ratio_{1, 1} {}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::initialize() {
  BaseClass::initialize();
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::doSweep() {
  int nb_of_steps;
  if (nb_steps_per_sweep_ < 0)  // Not thermalized or fixed.
    nb_of_steps = BaseClass::avgOrder() + 1;
  else
    nb_of_steps = nb_steps_per_sweep_;

  for (int i = 0; i < nb_of_steps; i++) {
    doStep();
  }

  BaseClass::n_steps_ += nb_of_steps;
  BaseClass::updateSweepAverages();
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::doStep() {
  initializeStep();

  if (rng_() <= 0.5) {
    n_accepted_ += tryVertexInsert();
  }
  else {
    if (configuration_.size())
      n_accepted_ += tryVertexRemoval();
    else {
      rng_(), rng_();  // Burn random numbers for testing consistency.
    }
  }

  assert(configuration_.checkConsistency());
}

template <class Parameters>
bool CtintWalker<linalg::CPU, Parameters>::tryVertexInsert() {
  configuration_.insertRandom(rng_);
  const int delta_vertices = configuration_.lastInsertionSize();

  // Compute the new pieces of the D(= M^-1) matrix.
  d_builder_ptr_->buildSQR(S_, Q_, R_, configuration_);

  acceptance_prob_ = insertionProbability(delta_vertices);

  const bool accept = rng_() < std::min(std::abs(acceptance_prob_), 1.);

  if (!accept) {
    popBack(delta_vertices);
  }

  else {
    if (acceptance_prob_ < 0)
      sign_ *= -1;
    applyInsertion(S_, Q_, R_);
  }
  return accept;
}

template <class Parameters>
bool CtintWalker<linalg::CPU, Parameters>::tryVertexRemoval() {
  acceptance_prob_ = removalProbability();
  const bool accept = rng_() < std::min(std::abs(acceptance_prob_), 1.);

  if (accept) {
    if (acceptance_prob_ < 0)
      sign_ *= -1;
    applyRemoval();
  }
  return accept;
}

template <class Parameters>
double CtintWalker<linalg::CPU, Parameters>::insertionProbability(const int delta_vertices) {
  const int old_size = configuration_.size() - delta_vertices;

  for (int s = 0; s < 2; ++s) {
    const auto& Q = Q_[s];
    if (Q.nrCols() == 0) {
      det_ratio_[s] = 1.;
      continue;
    }
    const auto& R = R_[s];
    auto& S = S_[s];
    auto& M = M_[s];

    if (M.nrRows()) {
      auto& M_Q = M_Q_[s];
      M_Q.resizeNoCopy(Q.size());
      linalg::matrixop::gemm(M, Q, M_Q);
      // S <- S_tilde^(-1) = S - R*M*Q
      linalg::matrixop::gemm(-1., R, M_Q, 1., S);
    }

    det_ratio_[s] = details::smallDeterminant(S);
  }

  const double combinatorial_factor =
      delta_vertices == 1 ? old_size + 1 : (old_size + 2) * (configuration_.nPartners(old_size) + 1);

  const double strength_factor =
      delta_vertices == 1 ? -beta_ * total_interaction_ * configuration_.getSign(old_size)
                          : total_interaction_ * beta_ * beta_ *
                                std::abs(configuration_.getStrength(old_size)) * possible_partners_;

  const double det_ratio = det_ratio_[0] * det_ratio_[1];

  return det_ratio * strength_factor / combinatorial_factor;

  //  return details::computeAcceptanceProbability(
  //      delta_vertices, det_ratio_[0] * det_ratio_[1], total_interaction_, beta_,
  //      configuration_.getStrength(old_size), combinatorial_factor, details::VERTEX_INSERTION);
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::applyInsertion(const MatrixPair& Sp, const MatrixPair& Qp,
                                                          const MatrixPair& Rp) {
  for (int s = 0; s < 2; ++s) {
    const int delta = Qp[s].nrCols();
    if (not delta)
      continue;
    // update M matrix.
    const auto& R = Rp[s];
    const auto S = linalg::makeConstantView(Sp[s]);
    auto& M = M_[s];
    const auto& M_Q = M_Q_[s];
    const int m_size = M.nrCols();

    if (not m_size) {
      M.resizeNoCopy(delta);
      auto M_view = MatrixView(M);
      smallInverse(*S, M_view, s);
      continue;
    }

    auto& R_M = ws_dn_;
    R_M.resizeNoCopy(R.size());
    linalg::matrixop::gemm(R, M, R_M);

    M.resize(m_size + delta);
    //  S_tilde = S^-1.
    MatrixView S_tilde(M, m_size, m_size, delta, delta);
    smallInverse(*S, S_tilde, s);

    // R_tilde = - S * R * M
    MatrixView R_tilde(M, m_size, 0, delta, m_size);
    linalg::matrixop::gemm(-1., S_tilde, R_M, double(0.), R_tilde);

    // Q_tilde = -M * Q * S
    MatrixView Q_tilde(M, 0, m_size, m_size, delta);
    linalg::matrixop::gemm(-1., M_Q, S_tilde, 0., Q_tilde);

    // update bulk: M += M*Q*S*R*M
    MatrixView M_bulk(M, 0, 0, m_size, m_size);
    linalg::matrixop::gemm(-1., Q_tilde, R_M, 1., M_bulk);
  }

  const int delta_vertices = configuration_.lastInsertionSize();
  for (int i = 0; i < delta_vertices; ++i)
    configuration_.commitInsertion(configuration_.size() - delta_vertices + i);
}

template <class Parameters>
double CtintWalker<linalg::CPU, Parameters>::removalProbability() {
  removal_list_ = configuration_.randomRemovalCandidate(rng_);

  const int n = configuration_.size();
  const int n_removed = removal_list_.size();

  for (int s = 0; s < 2; ++s) {
    matrix_removal_list_[s].clear();
    for (auto index : removal_list_) {
      configuration_.findIndices(matrix_removal_list_[s], index, s);
    }
    if (matrix_removal_list_[s].size() == 0) {
      det_ratio_[s] = 1.;
      continue;
    }
    std::sort(matrix_removal_list_[s].begin(), matrix_removal_list_[s].end());
    det_ratio_[s] = details::separateIndexDeterminant(M_[s], matrix_removal_list_[s]);
  }
  assert((matrix_removal_list_[0].size() + matrix_removal_list_[1].size()) % 2 == 0);

  const double combinatorial_factor =
      n_removed == 1 ? n : n * configuration_.nPartners(removal_list_[0]);
  const double strength_factor =
      n_removed == 1 ? -beta_ * total_interaction_ * configuration_.getSign(removal_list_[0])
                     : total_interaction_ * beta_ * beta_ * possible_partners_ *
                           std::abs(configuration_.getStrength(removal_list_[0]));

  const double det_ratio = det_ratio_[0] * det_ratio_[1];

  return det_ratio * combinatorial_factor / strength_factor;
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::applyRemoval() {
  for (auto idx : removal_list_)
    configuration_.markForRemoval(idx);

  std::array<int, 2> removal_size;
  for (int s = 0; s < 2; ++s)
    removal_size[s] = configuration_.getSector(s).size();

  moveRemovalToEnd();

  for (int s = 0; s < 2; ++s)
    removal_size[s] -= configuration_.getSector(s).size();

  for (int s = 0; s < 2; ++s) {
    auto& M = M_[s];
    const int delta = removal_size[s];
    if (delta == 0)  // Nothing to update.
      continue;
    const int m_size = M.nrCols() - delta;
    if (m_size == 0) {  // removing last vertex
      M.resize(0);
      continue;
    }

    MatrixView Q(M, 0, m_size, m_size, delta);
    MatrixView R(M, m_size, 0, delta, m_size);

    MatrixView S(M, m_size, m_size, delta, delta);
    smallInverse(S, s);

    auto& Q_S = M_Q_[s];
    Q_S.resizeNoCopy(Q.size());
    linalg::matrixop::gemm(Q, S, Q_S);

    // M -= Q*S^-1*R
    MatrixView M_bulk(M, 0, 0, m_size, m_size);
    linalg::matrixop::gemm(-1., Q_S, R, 1., M_bulk);
    M.resize(m_size);
  }
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::popBack(int delta_vertices) {
  for (int i = configuration_.size() - delta_vertices; i < configuration_.size(); ++i) {
    removal_list_.push_back(i);
  }

  configuration_.moveAndShrink(matrix_source_list_, matrix_removal_list_, removal_list_);

  for (int s = 0; s < 2; ++s) {
    M_[s].resize(configuration_.getSector(s).size());
  }
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::moveRemovalToEnd() {
  configuration_.moveAndShrink(matrix_source_list_, matrix_removal_list_, removal_list_);

  for (int s = 0; s < 2; ++s) {
    for (int i = 0; i < matrix_source_list_[s].size(); ++i)
      linalg::matrixop::swapRowAndCol(M_[s], matrix_source_list_[s][i], matrix_removal_list_[s][i]);
  }
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::smallInverse(const MatrixView& in, MatrixView& out,
                                                        const int s) {
  details::smallInverse(in, out, det_ratio_[s], ipiv_, v_work_);
}
template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::smallInverse(MatrixView& in_out, const int s) {
  details::smallInverse(in_out, det_ratio_[s], ipiv_, v_work_);
}

template <class Parameters>
void CtintWalker<linalg::CPU, Parameters>::initializeStep() {
  removal_list_.clear();
  for (int s = 0; s < 2; ++s) {
    matrix_removal_list_[s].clear();
    matrix_source_list_[s].clear();
  }
}

}  // namespace ctint
}  // namespace solver
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_DCA_STEP_CLUSTER_SOLVER_CTINT_WALKER_CTINT_WALKER_CPU_HPP