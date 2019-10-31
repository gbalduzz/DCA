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
// This class executes the DCA(+) loop.

#ifndef DCA_PHYS_DCA_LOOP_DCA_LOOP_HPP
#define DCA_PHYS_DCA_LOOP_DCA_LOOP_HPP

#include <cmath>
#include <cstdio>  // remove
#include <iostream>
#include <stdexcept>
#include <string>

#include "dca/function/domains.hpp"
#include "dca/io/hdf5/hdf5_writer.hpp"
#include "dca/io/json/json_writer.hpp"
#include "dca/phys/dca_algorithms/compute_greens_function.hpp"
#include "dca/phys/dca_loop/dca_loop_data.hpp"
#include "dca/phys/dca_step/cluster_mapping/cluster_exclusion.hpp"
#include "dca/phys/dca_step/cluster_mapping/coarsegraining/coarsegraining_sp.hpp"
#include "dca/phys/dca_step/cluster_mapping/double_counting_correction.hpp"
#include "dca/phys/dca_step/cluster_mapping/update_chemical_potential.hpp"
#include "dca/phys/dca_step/cluster_solver/high_temperature_series_expansion/high_temperature_series_expansion_solver.hpp"
#include "dca/phys/dca_step/lattice_mapping/lattice_mapping_sp.hpp"
#include "dca/phys/dca_step/symmetrization/symmetrize.hpp"
#include "dca/phys/domains/cluster/cluster_domain.hpp"
#include "dca/phys/domains/quantum/electron_band_domain.hpp"
#include "dca/phys/domains/quantum/electron_spin_domain.hpp"
#include "dca/util/print_time.hpp"

namespace dca {
namespace phys {
// dca::phys::

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
class DcaLoop {
public:
  using profiler_type = typename ParametersType::profiler_type;
  using concurrency_type = typename ParametersType::concurrency_type;

  using Lattice = typename ParametersType::lattice_type;

  using b = func::dmn_0<domains::electron_band_domain>;
  using s = func::dmn_0<domains::electron_spin_domain>;
  using k_DCA =
      func::dmn_0<domains::cluster_domain<double, ParametersType::lattice_type::DIMENSION, domains::CLUSTER,
                                          domains::MOMENTUM_SPACE, domains::BRILLOUIN_ZONE>>;
  using k_HOST =
      func::dmn_0<domains::cluster_domain<double, ParametersType::lattice_type::DIMENSION, domains::LATTICE_SP,
                                          domains::MOMENTUM_SPACE, domains::BRILLOUIN_ZONE>>;

  using cluster_exclusion_type = clustermapping::cluster_exclusion<ParametersType, DcaDataType>;
  using double_counting_correction_type =
      clustermapping::double_counting_correction<ParametersType, DcaDataType>;
  using coarsegraining_sp_type = clustermapping::CoarsegrainingSp<ParametersType>;
  using lattice_map_sp_type = latticemapping::lattice_mapping_sp<ParametersType, k_DCA, k_HOST>;
  using update_chemical_potential_type =
      clustermapping::update_chemical_potential<ParametersType, DcaDataType, coarsegraining_sp_type>;
  using HTS_solver_type =
      solver::HighTemperatureSeriesExpansionSolver<dca::linalg::CPU, ParametersType, DcaDataType>;

  DcaLoop(ParametersType& parameters_ref, DcaDataType& data__ref, concurrency_type& concurrency_ref);

  void read();
  void write();

  void initialize();
  void execute();
  void finalize();

private:
  void adjust_chemical_potential();

  void perform_cluster_mapping();
  void perform_cluster_mapping_self_energy();
  void perform_cluster_mapping_Greens_function();

  void adjust_coarsegrained_self_energy();
  void adjust_impurity_self_energy();

  void perform_cluster_exclusion_step();

  double solve_cluster_problem(int DCA_iteration);

  void perform_lattice_mapping();

  void update_DCA_loop_data_functions(int DCA_iteration);

  void writeTempData(int iteration) const;
  void cleanTempData() const;

  ParametersType& parameters_;
  DcaDataType& data_;
  concurrency_type& concurrency;

  DcaLoopData<ParametersType> DCA_info_struct;

  cluster_exclusion_type cluster_exclusion_obj;
  double_counting_correction_type double_counting_correction_obj;

  coarsegraining_sp_type cluster_mapping_obj;
  lattice_map_sp_type lattice_mapping_obj;

  update_chemical_potential_type update_chemical_potential_obj;

protected:
  MCIntegratorType monte_carlo_integrator_;
};

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::DcaLoop(ParametersType& parameters_ref,
                                                                DcaDataType& data__ref,
                                                                concurrency_type& concurrency_ref)
    : parameters_(parameters_ref),
      data_(data__ref),
      concurrency(concurrency_ref),

      DCA_info_struct(),

      cluster_exclusion_obj(parameters_, data_),
      double_counting_correction_obj(parameters_, data_),

      cluster_mapping_obj(parameters_),
      lattice_mapping_obj(parameters_),

      update_chemical_potential_obj(parameters_, data_, cluster_mapping_obj),

      monte_carlo_integrator_(parameters_ref, data__ref) {
  if (concurrency.id() == concurrency.first())
    std::cout << "\n\n\t" << __FUNCTION__ << " has started \t" << dca::util::print_time() << "\n\n";
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::read() {
  if (parameters_.get_initial_self_energy() != "zero")
    data_.read(parameters_.get_initial_self_energy());
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::write() {
  const std::string& output_format = parameters_.get_output_format();
  const std::string& file_name = parameters_.get_directory() + parameters_.get_filename_dca();

  std::cout << "\n\n\t\t start writing " << file_name << "\t" << dca::util::print_time() << "\n\n";

  if (output_format == "JSON") {
    dca::io::JSONWriter writer;
    writer.open_file(file_name);

    parameters_.write(writer);
    data_.write(writer);
    monte_carlo_integrator_.write(writer);
    DCA_info_struct.write(writer);

    writer.close_file();
  }

  else if (output_format == "HDF5") {
    dca::io::HDF5Writer writer;
    writer.open_file(file_name);

    parameters_.write(writer);
    data_.write(writer);
    monte_carlo_integrator_.write(writer);
    DCA_info_struct.write(writer);

    writer.close_file();
  }

  else
    throw std::logic_error(__FUNCTION__);
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::initialize() {
  if (parameters_.get_initial_self_energy() != "zero") {
    data_.initialize_Sigma();
    perform_lattice_mapping();
  }
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::execute() {
  for (int i = 0; i < parameters_.get_dca_iterations(); i++) {
    adjust_chemical_potential();

    perform_cluster_mapping();

    adjust_coarsegrained_self_energy();  // double-counting-correction

    perform_cluster_exclusion_step();

    double L2_Sigma_difference = solve_cluster_problem(i);  // returned from cluster_solver::finalize

    adjust_impurity_self_energy();  // double-counting-correction

    perform_lattice_mapping();

    if (i != parameters_.get_dca_iterations() - 1)
      writeTempData(i);

    update_DCA_loop_data_functions(i);

    if (L2_Sigma_difference <
        parameters_.get_dca_accuracy())  // set the acquired accuracy on |Sigma_QMC - Sigma_cg|
      break;
  }
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::finalize() {
  perform_cluster_mapping_self_energy();
  data_.compute_Sigma_bands();
  data_.compute_single_particle_properties();
  cleanTempData();
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::adjust_chemical_potential() {
  if (parameters_.adjust_chemical_potential())
    update_chemical_potential_obj.execute();
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::perform_cluster_mapping() {
  perform_cluster_mapping_self_energy();
  perform_cluster_mapping_Greens_function();
  // perform_cluster_exclusion_step();
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::perform_cluster_mapping_self_energy() {
  if (concurrency.id() == concurrency.first())
    std::cout << "\n\t\t coarsegrain-Selfenergy " << dca::util::print_time();

  profiler_type profiler("coarsegrain-Selfenergy", "DCA", __LINE__);

  if (parameters_.do_dca_plus())
    cluster_mapping_obj.compute_S_K_w(data_.Sigma_lattice, data_.Sigma_cluster);
  else
    data_.Sigma_cluster = data_.Sigma;

  data_.print_Sigma_QMC_versus_Sigma_cg();

  symmetrize::execute<Lattice>(data_.Sigma_cluster, data_.H_symmetry);
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::perform_cluster_mapping_Greens_function() {
  if (concurrency.id() == concurrency.first())
    std::cout << "\n\t\t coarsegrain-Greens-function " << dca::util::print_time();

  profiler_type profiler("coarsegrain-Greens-function", "DCA", __LINE__);

  // Finite-size QMC
  if (parameters_.do_finite_size_qmc())
    compute_G_k_w(data_.H_DCA, data_.Sigma, parameters_.get_chemical_potential(),
                  parameters_.get_coarsegraining_threads(), data_.G_k_w);
  // DCA+
  else if (parameters_.do_dca_plus())
    cluster_mapping_obj.compute_G_K_w(data_.Sigma_lattice, data_.G_k_w);
  // Standard DCA
  else
    cluster_mapping_obj.compute_G_K_w(data_.Sigma, data_.G_k_w);

  symmetrize::execute<Lattice>(data_.G_k_w, data_.H_symmetry);
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::adjust_coarsegrained_self_energy() {
  double_counting_correction_obj.execute_before_solver();
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::adjust_impurity_self_energy() {
  double_counting_correction_obj.execute_after_solver();
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::perform_cluster_exclusion_step() {
  if (concurrency.id() == concurrency.first())
    std::cout << "\n\t\t cluster-exclusion-step " << dca::util::print_time();

  profiler_type profiler("cluster-exclusion-step", "DCA", __LINE__);

  cluster_exclusion_obj.execute();
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
double DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::solve_cluster_problem(int DCA_iteration) {
  {
    profiler_type profiler("initialize cluster-solver", "DCA", __LINE__);
    monte_carlo_integrator_.initialize(DCA_iteration);
  }

  {
    profiler_type profiler("Quantum Monte Carlo integration", "DCA", __LINE__);
    monte_carlo_integrator_.integrate();
  }

  {
    profiler_type profiler("finalize cluster-solver", "DCA", __LINE__);
    double L2_Sigma_difference = monte_carlo_integrator_.finalize(DCA_info_struct);

    return L2_Sigma_difference;
  }
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::perform_lattice_mapping() {
  profiler_type profiler("lattice-mapping", "DCA", __LINE__);

  if (concurrency.id() == concurrency.first())
    std::cout << "\n\t\t lattice-mapping " << dca::util::print_time();

  if (parameters_.do_dca_plus()) {
    if (parameters_.hts_approximation()) {
      DcaDataType data__HTS(parameters_);

      data__HTS.H_HOST = data_.H_HOST;
      data__HTS.H_interactions = data_.H_interactions;

      HTS_solver_type HTS_solver(parameters_, data__HTS);

      lattice_mapping_obj.execute_with_HTS_approximation(
          data__HTS, HTS_solver, cluster_mapping_obj, data_.Sigma, data_.Sigma_lattice_interpolated,
          data_.Sigma_lattice_coarsegrained, data_.Sigma_lattice);
    }
    else {
      lattice_mapping_obj.execute(data_.Sigma, data_.Sigma_lattice_interpolated,
                                  data_.Sigma_lattice_coarsegrained, data_.Sigma_lattice);
    }
  }
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::update_DCA_loop_data_functions(int i) {
  DCA_info_struct.density(i) = update_chemical_potential_obj.compute_density();
  DCA_info_struct.chemical_potential(i) = parameters_.get_chemical_potential();

  if (concurrency.id() == concurrency.first()) {
    std::cout << "\n\n\t\t\t total-density : " << DCA_info_struct.density(i)
              << "\t (time : " << dca::util::print_time() << ")\n\n";
  }

  for (int l1 = 0; l1 < b::dmn_size() * s::dmn_size(); l1++)
    DCA_info_struct.orbital_occupancies(l1, i) = data_.orbital_occupancy(l1);

  for (int l1 = 0; l1 < b::dmn_size() * s::dmn_size(); l1++)
    for (int k_ind = 0; k_ind < k_DCA::dmn_size(); k_ind++)
      DCA_info_struct.n_k(l1, k_ind, i) = 1. - std::abs(data_.G_k_t(l1, l1, k_ind, 0));

  for (int l1 = 0; l1 < b::dmn_size() * s::dmn_size(); l1++)
    for (int k_ind = 0; k_ind < k_DCA::dmn_size(); k_ind++)
      // TODO: Use t::dmn_size() instead of parameters_.get_sp_time_intervals().
      DCA_info_struct.A_k(l1, k_ind, i) =
          std::abs(data_.G_k_t(l1, l1, k_ind, parameters_.get_sp_time_intervals() / 2)) *
          parameters_.get_beta() / M_PI;
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::writeTempData(const int iteration) const {
  if (concurrency.id() != concurrency.last())
    return;

  const std::string filename = parameters_.get_filename_qmc() + ".tmp";
  io::HDF5Writer writer;

  writer.open_file(filename);
  // Write self-energy.
  writer.open_group("functions");
  writer.execute(data_.Sigma);
  writer.close_group();
  // Write chemical potential.
  writer.open_group("parameters");
  writer.open_group("physics");
  writer.execute("chemical-potential", parameters_.get_chemical_potential());
  writer.close_group();
  writer.close_group();
  // Write lightweight metadata.
  DCA_info_struct.write(writer);
  writer.execute("iteration", iteration);
  writer.close_file();
}

template <typename ParametersType, typename DcaDataType, typename MCIntegratorType>
void DcaLoop<ParametersType, DcaDataType, MCIntegratorType>::cleanTempData() const {
  if (concurrency.id() != concurrency.last())
    return;

  const std::string filename = parameters_.get_filename_qmc() + ".tmp";
  std::remove(filename.c_str());
}

}  // namespace phys
}  // namespace dca

#endif  // DCA_PHYS_DCA_LOOP_DCA_LOOP_HPP
