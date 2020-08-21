// Copyright (C) 2018 ETH Zurich
// Copyright (C) 2018 UT-Battelle, LLC
// All rights reserved.
//
// See LICENSE for terms of usage.
// See CITATION.md for citation guidelines, if DCA++ is used for scientific publications.
//
// Author: Peter Staar (taa@zurich.ibm.com)
//         Andrei Plamada (plamada@itp.phys.ethz.ch)
//
// Using the regula falsi method this class determines the value of the chemical potential that
// produces the target electron density.

#ifndef DCA_PHYS_DCA_STEP_CLUSTER_MAPPING_UPDATE_CHEMICAL_POTENTIAL_HPP
#define DCA_PHYS_DCA_STEP_CLUSTER_MAPPING_UPDATE_CHEMICAL_POTENTIAL_HPP

#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "dca/function/domains.hpp"
#include "dca/function/function.hpp"
#include "dca/math/function_transform/function_transform.hpp"
#include "dca/phys/domains/cluster/cluster_domain.hpp"
#include "dca/phys/domains/quantum/electron_band_domain.hpp"
#include "dca/phys/domains/quantum/electron_spin_domain.hpp"
#include "dca/phys/domains/time_and_frequency/frequency_domain.hpp"
#include "dca/phys/domains/time_and_frequency/time_domain.hpp"
#include "dca/phys/dca_algorithms/compute_greens_function.hpp"
#include "dca/phys/domains/cluster/cluster_domain_aliases.hpp"
#include "dca/util/print_time.hpp"

namespace dca {
namespace phys {
namespace clustermapping {
// dca::phys::clustermapping::

template <typename Parameters, typename Data, typename Coarsegraining>
class update_chemical_potential {
public:
  using Concurrency = typename Parameters::concurrency_type;
  constexpr static int spin_sectors = Parameters::lattice_type::spin_symmetric ? 2 : 1;

  using TDmn = func::dmn_0<domains::time_domain>;
  using WDmn = func::dmn_0<domains::frequency_domain>;

  using BDmn = func::dmn_0<domains::electron_band_domain>;
  using SDmn = func::dmn_0<domains::electron_spin_domain>;
  using NuDmn = func::dmn_variadic<BDmn, SDmn>;  // orbital-spin index

  using CDA = ClusterDomainAliases<Parameters::lattice_type::DIMENSION>;
  using RClusterDmn = typename CDA::RClusterDmn;
  using KClusterDmn = typename CDA::KClusterDmn;

public:
  update_chemical_potential(/*const*/ Parameters& parameters_ref, Data& MOMS_ref,
                            Coarsegraining& coarsegraining_ref);

  // Executes the search for the new value of the chemical potential.
  void execute();

  // Computes the current electron density.
  double compute_density();

private:
  // Computes the new estimate for the chemical potential within the regula falsi method.
  double get_new_chemical_potential(double d_0, double mu_lb, double mu_ub, double n_lb, double n_ub);

  void compute_density_correction(func::function<double, NuDmn>& result);

  void compute_density_coefficients(
      func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>>& A,
      func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>>& B,
      const func::function<std::complex<double>, func::dmn_variadic<NuDmn, NuDmn, KClusterDmn, WDmn>>& G);

  // Determines initial lower and upper bounds of the chemical potential.
  void search_bounds(double dens);

  // Prints the current upper and lower bounds of the chemical potential and the corresponding
  // densities.
  void print_bounds();

private:
  /*const*/ Parameters& parameters;
  const Concurrency& concurrency;

  Data& MOMS;
  Coarsegraining& coarsegraining;

  std::pair<double, double> lower_bound;
  std::pair<double, double> upper_bound;
};

template <typename Parameters, typename Data, typename Coarsegraining>
update_chemical_potential<Parameters, Data, Coarsegraining>::update_chemical_potential(
    /*const*/ Parameters& parameters_ref, Data& MOMS_ref, Coarsegraining& coarsegraining_ref)
    : parameters(parameters_ref),
      concurrency(parameters.get_concurrency()),

      MOMS(MOMS_ref),
      coarsegraining(coarsegraining_ref) {}

template <typename Parameters, typename Data, typename Coarsegraining>
void update_chemical_potential<Parameters, Data, Coarsegraining>::execute() {
  double dens = compute_density();

  if (concurrency.id() == concurrency.first()) {
    std::cout.precision(6);
    std::cout << std::scientific;

    std::cout << "\n\t\t initial chemical potential : " << parameters.get_chemical_potential()
              << " (" << dens << ")\n\n";
  }

  // Current value is good enough, return.
  if (std::abs(dens - parameters.get_density()) < 1.e-3)
    return;

  search_bounds(dens);

  while (true) {
    double d_0 = parameters.get_density();  // desired density

    // Lower and upper bound of the chemical potential
    double mu_lb = lower_bound.first;
    double mu_ub = upper_bound.first;

    // Densities corresponding to lower and upper bounds
    double n_lb = lower_bound.second;
    double n_ub = upper_bound.second;

    // TODO: do not write in parameters.
    parameters.get_chemical_potential() = get_new_chemical_potential(d_0, mu_lb, mu_ub, n_lb, n_ub);

    dens = compute_density();

    // Stop.
    if (std::abs(dens - parameters.get_density()) < 1.e-3) {
      if (concurrency.id() == concurrency.first()) {
        std::cout.precision(6);
        std::cout << std::scientific;

        std::cout << "\n\t\t final chemical potential : " << parameters.get_chemical_potential()
                  << " (" << dens << ")\n";
      }

      break;
    }

    // Continue.
    else {
      if (dens < parameters.get_density()) {
        lower_bound.first = parameters.get_chemical_potential();
        lower_bound.second = dens;
      }
      else {
        upper_bound.first = parameters.get_chemical_potential();
        upper_bound.second = dens;
      }
    }

    print_bounds();
  }
}

template <typename Parameters, typename Data, typename Coarsegraining>
double update_chemical_potential<Parameters, Data, Coarsegraining>::get_new_chemical_potential(
    const double d_0, const double mu_lb, const double mu_ub, const double n_lb, const double n_ub) {
  return (mu_ub - mu_lb) / (n_ub - n_lb) * (d_0 - n_lb) + mu_lb;
}

template <typename Parameters, typename Data, typename Coarsegraining>
void update_chemical_potential<Parameters, Data, Coarsegraining>::search_bounds(double dens) {
  const double factor = 2;
  double delta = 0.1;

  if (dens < parameters.get_density()) {
    lower_bound.first = parameters.get_chemical_potential();
    lower_bound.second = dens;

    while (true) {
      parameters.get_chemical_potential() += delta;

      dens = compute_density();

      upper_bound.first = parameters.get_chemical_potential();
      upper_bound.second = dens;

      print_bounds();

      if (parameters.get_density() < dens)
        break;
      else
        lower_bound = upper_bound;

      delta *= factor;
    }
  }
  else {
    upper_bound.first = parameters.get_chemical_potential();
    upper_bound.second = dens;

    while (true) {
      parameters.get_chemical_potential() -= delta;

      dens = compute_density();

      lower_bound.first = parameters.get_chemical_potential();
      lower_bound.second = dens;

      print_bounds();

      if (dens < parameters.get_density())
        break;
      else
        upper_bound = lower_bound;

      delta *= factor;
    }
  }
}

template <typename Parameters, typename Data, typename Coarsegraining>
double update_chemical_potential<Parameters, Data, Coarsegraining>::compute_density() {
  if (parameters.do_finite_size_qmc())
    compute_G_k_w(MOMS.H_DCA, MOMS.Sigma, parameters.get_chemical_potential(),
                  parameters.get_coarsegraining_threads(), MOMS.G_k_w);
  else if (parameters.do_dca_plus())
    coarsegraining.compute_G_K_w(MOMS.Sigma_lattice, MOMS.G_k_w);
  else
    coarsegraining.compute_G_K_w(MOMS.Sigma, MOMS.G_k_w);

  MOMS.G_k_w -= MOMS.G0_k_w;

  math::transform::FunctionTransform<WDmn, TDmn>::execute(MOMS.G_k_w, MOMS.G_k_t);

  MOMS.G_k_t += MOMS.G0_k_t;

  math::transform::FunctionTransform<KClusterDmn, RClusterDmn>::execute(MOMS.G_k_t, MOMS.G_r_t);

  MOMS.G_k_w += MOMS.G0_k_w;

  const int origin = RClusterDmn::parameter_type::origin_index();

  MOMS.orbital_occupancy = 0.;
  compute_density_correction(MOMS.orbital_occupancy);
  double result = 0.0;

  for (int b = 0; b < BDmn::dmn_size(); ++b) {
    for (int s = 0; s < spin_sectors; ++s) {
      const auto G_r_t_val = MOMS.G_r_t(b, s, b, s, origin, 0);
      if (std::abs(std::imag(G_r_t_val)) >= 1e-6) {
        throw(std::logic_error("G_ii(r = 0, t = 0) is complex"));
      }

      MOMS.orbital_occupancy(b, s) += 1. - std::real(G_r_t_val);
      result += MOMS.orbital_occupancy(b, s);
    }
  }

  return result;
}

/*!
 *  We assume that G_ii(w>>0) ~ 1/(i w_m + A + B/(i w_m))
 */
template <typename Parameters, typename Data, typename Coarsegraining>
void update_chemical_potential<Parameters, Data, Coarsegraining>::compute_density_correction(
    func::function<double, NuDmn>& result) {
  std::complex<double> I(0, 1);

  double N_k = KClusterDmn::dmn_size();
  double beta = parameters.get_beta();

  func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>> A;
  func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>> B;

  func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>> A0;
  func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>> B0;

  compute_density_coefficients(A, B, MOMS.G_k_w);
  compute_density_coefficients(A0, B0, MOMS.G0_k_w);

  for (int k_i = 0; k_i < KClusterDmn::dmn_size(); k_i++) {
    for (int nu_i = 0; nu_i < NuDmn::dmn_size(); nu_i++) {
      double tmp = 0.0;
      double sum = 1.e-16;

      int l = WDmn::dmn_size() / 2;

      do {
        std::complex<double> I_wn = (M_PI / beta) * (1 + 2 * l) * I;

        std::complex<double> G = 1. / (I_wn + A(nu_i, k_i) + B(nu_i, k_i) / I_wn);
        std::complex<double> G0 = 1. / (I_wn + A0(nu_i, k_i) + B0(nu_i, k_i) / I_wn);

        tmp = std::real(G - G0);
        sum += tmp;

        l += 1;
      } while (std::abs(tmp / sum) > 1.e-6 and l < 1.e6);

      result(nu_i) += sum;
    }
  }
  for (int nu_i = 0; nu_i < NuDmn::dmn_size(); nu_i++)
    result(nu_i) *= (2. / (beta * N_k));
}

template <typename Parameters, typename Data, typename Coarsegraining>
void update_chemical_potential<Parameters, Data, Coarsegraining>::compute_density_coefficients(
    func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>>& A,
    func::function<double, func::dmn_variadic<NuDmn, KClusterDmn>>& B,
    const func::function<std::complex<double>, func::dmn_variadic<NuDmn, NuDmn, KClusterDmn, WDmn>>& G) {
  A = 0;
  B = 0;

  const int nb_wm = parameters.get_tail_frequencies();

  if (nb_wm > 0) {
    for (int k_i = 0; k_i < KClusterDmn::dmn_size(); k_i++) {
      for (int nu_i = 0; nu_i < NuDmn::dmn_size(); nu_i++) {
        for (int w_i = WDmn::dmn_size() - nb_wm; w_i < WDmn::dmn_size(); w_i++) {
          double wm = WDmn::get_elements()[w_i];

          A(nu_i, k_i) += std::real(1. / G(nu_i, nu_i, k_i, w_i));
          B(nu_i, k_i) += wm * (wm - std::imag(1. / G(nu_i, nu_i, k_i, w_i)));
        }
      }
    }

    A /= nb_wm;
    B /= nb_wm;

    if (nb_wm == 1) {
      const std::complex<double> I(0, 1);

      for (int k_i = 0; k_i < KClusterDmn::dmn_size(); k_i++) {
        for (int nu_i = 0; nu_i < NuDmn::dmn_size(); nu_i++) {
          int w_i = WDmn::dmn_size() - 1;
          double wm = WDmn::get_elements()[w_i];

          if (std::abs((G(nu_i, nu_i, k_i, w_i)) -
                       1. / (I * wm + A(nu_i, k_i) + B(nu_i, k_i) / (I * wm))) > 1.e-12)
            throw std::logic_error(__FUNCTION__);
        }
      }
    }
  }
}

template <typename Parameters, typename Data, typename Coarsegraining>
void update_chemical_potential<Parameters, Data, Coarsegraining>::print_bounds() {
  if (concurrency.id() == concurrency.first()) {
    std::cout.precision(6);
    std::cout << std::scientific;

    std::cout << "\t";
    std::cout << "\t mu : " << lower_bound.first << " (n = " << lower_bound.second << ")";
    std::cout << "\t mu : " << upper_bound.first << " (n = " << upper_bound.second << ")\t";
    std::cout << dca::util::print_time() << "\n";
  }
}

}  // namespace clustermapping
}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_DCA_STEP_CLUSTER_MAPPING_UPDATE_CHEMICAL_POTENTIAL_HPP
