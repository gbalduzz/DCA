//-*-C++-*-

#ifndef ADVANCED_FERMIONIC_OVERLAP_MATRICES_H
#define ADVANCED_FERMIONIC_OVERLAP_MATRICES_H

namespace DCA
{
  namespace ADVANCED_EXACT_DIAGONALIZATION
  {
    template<typename parameter_type, typename ed_options>
    class fermionic_overlap_matrices
    {
#include "type_definitions.h"

    public:

      typedef typename ed_options::b_dmn b_dmn;
      typedef typename ed_options::s_dmn s_dmn;
      typedef typename ed_options::r_dmn r_dmn;
      typedef typename ed_options::k_dmn k_dmn;

      typedef typename ed_options::profiler_t       profiler_t;
      typedef typename ed_options::concurrency_type concurrency_type;

      typedef typename ed_options::scalar_type  scalar_type;
      typedef typename ed_options::complex_type complex_type;

      typedef typename ed_options::vector_type         vector_type;
      typedef typename ed_options::matrix_type         matrix_type;
      typedef typename ed_options::int_matrix_type int_matrix_type;

      typedef typename ed_options::nu_dmn nu_dmn;
      typedef typename ed_options::b_s_r  b_s_r_dmn_type;

      typedef fermionic_Hamiltonian<parameter_type, ed_options> fermionic_Hamiltonian_type;

      typedef typename ed_options::phi_type phi_type;

      typedef Fock_space       <parameter_type, ed_options> fermionic_Fock_space_type;
      typedef Hilbert_space    <parameter_type, ed_options> Hilbert_space_type;
      typedef Hilbert_space_phi_representation<parameter_type, ed_options> Hilbert_space_phi_representation_type;
      typedef psi_state  <parameter_type, ed_options> psi_state_type;

      typedef dmn_0    <fermionic_Fock_space_type>  fermionic_Fock_dmn_type;
      typedef operators<parameter_type, ed_options> fermionic_operators_type;

      typedef sparse_element<parameter_type, ed_options> sparse_element_type;

    public:

      fermionic_overlap_matrices(parameter_type& parameters_ref, fermionic_Hamiltonian_type& Hamiltonian_ref);

      ~fermionic_overlap_matrices();

      void construct_creation_set_all();
      void construct_annihilation_set_all();

      void construct_creation_set_nonzero_sparse();
      void construct_annihilation_set_nonzero_sparse();

      void compute_creation_matrix    (int HS_i, int HS_j, int b_s_r, matrix_type& dense_creation);
      void compute_annihilation_matrix(int HS_i, int HS_j, int b_s_r, matrix_type& dense_annihilation);

      void compute_creation_matrix_fast    (int HS_i, int HS_j, int b_s_r, matrix_type& dense_creation    , matrix_type& tmp);
      void compute_annihilation_matrix_fast(int HS_i, int HS_j, int b_s_r, matrix_type& dense_annihilation, matrix_type& tmp);

      bool check_hermitianess();

      void compute_overlap(int HS_n, int HS_m);
      void print_overlap(const char* filename, int HS_n, int HS_m);

      void print_creation_matrix(const char* filename);

      //void compute_all_creation_matrices();

      FUNC_LIB::function<int, dmn_3<fermionic_Fock_dmn_type, fermionic_Fock_dmn_type, b_s_r_dmn_type> >& get_creation_set_all()
      { return creation_set_all; }
      FUNC_LIB::function<int, dmn_3<fermionic_Fock_dmn_type, fermionic_Fock_dmn_type, b_s_r_dmn_type> >& get_annihilation_set_all()
      { return annihilation_set_all; }


    private:

      void compute_sparse_creation    (int HS_i, int HS_j, int b_s_r);
      void compute_sparse_annihilation(int HS_i, int HS_j, int b_s_r);

      void sort(std::vector< sparse_element_type >& sparse_matrix);

      void merge(std::vector< sparse_element_type >& sparse_matrix);

    private:

      parameter_type&   parameters;
      concurrency_type& concurrency;

      fermionic_Hamiltonian_type& Hamiltonian;

      // creation_set(i,j,l)
      // creation_set(i,j,b_i,s_i,r_i)
      FUNC_LIB::function<int, dmn_3<fermionic_Fock_dmn_type, fermionic_Fock_dmn_type, b_s_r_dmn_type> > creation_set_all;
      FUNC_LIB::function< std::vector< sparse_element_type >,
                dmn_3<fermionic_Fock_dmn_type, fermionic_Fock_dmn_type, b_s_r_dmn_type> > creation_set_nonzero_sparse;

    FUNC_LIB::function<int, dmn_3<fermionic_Fock_dmn_type, fermionic_Fock_dmn_type, b_s_r_dmn_type> > annihilation_set_all;
    FUNC_LIB::function< std::vector< sparse_element_type >,
              dmn_3<fermionic_Fock_dmn_type, fermionic_Fock_dmn_type, b_s_r_dmn_type> > annihilation_set_nonzero_sparse;

  matrix_type sparse_creation; // <psi_i | c^\dagger| psi>
  matrix_type sparse_annihilation;

  // matrix_type dense_creation;  // < V_i^\dagger  | c^\dagger| V_j  >
  matrix_type helper;

  FUNC_LIB::function<complex_type, dmn_2<b_s_r_dmn_type, b_s_r_dmn_type> > overlap;

};


template<typename parameter_type, typename ed_options>
fermionic_overlap_matrices<parameter_type, ed_options>::fermionic_overlap_matrices(parameter_type& parameters_ref,
                                                                                   fermionic_Hamiltonian_type& Hamiltonian_ref):
  parameters(parameters_ref),
  concurrency(parameters.get_concurrency()),

  Hamiltonian(Hamiltonian_ref),

  creation_set_all("creation_set_all"),
  creation_set_nonzero_sparse("createn_set_nonzero_sparse")
{}

template<typename parameter_type, typename ed_options>
fermionic_overlap_matrices<parameter_type, ed_options>::~fermionic_overlap_matrices()
{}


template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::construct_creation_set_all()
{
  if(concurrency.id()==0)
    cout << "\n\t" << __FUNCTION__ << endl;

  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  creation_set_all.reset();
  creation_set_all = -1;

  vector<int> Sz(2);
  Sz[0] = 1;
  Sz[1] = -1;

  int index = 0;

  double memory_estimate = 0.;

  // loop order?

  for(int j = 0; j < Hilbert_spaces.size(); ++j){
    for(int i = 0; i < Hilbert_spaces.size(); ++i){

      if(Hilbert_spaces[i].get_occupation() == Hilbert_spaces[j].get_occupation()+1){
        for(int r = 0; r < r_dmn::dmn_size(); ++r){
          for(int s = 0; s < s_dmn::dmn_size(); ++s){

            if(Hilbert_spaces[i].get_magnetization() == Hilbert_spaces[j].get_magnetization()+Sz[s]){
              for(int b = 0; b < b_dmn::dmn_size(); ++b){

                creation_set_all(i,j,b,s,r) = index++;
                memory_estimate += Hilbert_spaces[i].size()*Hilbert_spaces[j].size();
              }
            }
          }
        }
      }
    }
  }

  memory_estimate *= sizeof(complex_type) * 1.e-9;

  cout << "number of creation matrices : " << index << endl;

  cout << "memory estimate : " << memory_estimate << " (giga-bytes)" << endl;

}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::construct_annihilation_set_all()
{
  if(concurrency.id()==0)
    cout << "\n\t" << __FUNCTION__ << endl;

  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  annihilation_set_all.reset();
  annihilation_set_all = -1;

  vector<int> Sz(2);
  Sz[0] = 1;
  Sz[1] = -1;

  int index = 0;

  double memory_estimate = 0.;

  // loop order?

  for(int j = 0; j < Hilbert_spaces.size(); ++j){
    for(int i = 0; i < Hilbert_spaces.size(); ++i){

      if(Hilbert_spaces[i].get_occupation() == Hilbert_spaces[j].get_occupation()-1){
        for(int r = 0; r < r_dmn::dmn_size(); ++r){
          for(int s = 0; s < s_dmn::dmn_size(); ++s){

            if(Hilbert_spaces[i].get_magnetization() == Hilbert_spaces[j].get_magnetization()-Sz[s]){
              for(int b = 0; b < b_dmn::dmn_size(); ++b){

                annihilation_set_all(i,j,b,s,r) = index++;
                memory_estimate += Hilbert_spaces[i].size()*Hilbert_spaces[j].size();
              }
            }
          }
        }
      }
    }
  }

  memory_estimate *= sizeof(complex_type) * 1.e-9;

  cout << "number of creation matrices : " << index << endl;

  cout << "memory estimate : " << memory_estimate << " (giga-bytes)" << endl;

}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::construct_creation_set_nonzero_sparse()
{
  if(concurrency.id()==0)
    cout << "\n\t" << __FUNCTION__ << endl;

  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  unsigned long non_zero = 0;

  creation_set_nonzero_sparse.reset();

  for(int r_idx = 0; r_idx < Hilbert_spaces.size(); ++r_idx){
    for(int l_idx = 0; l_idx < Hilbert_spaces.size(); ++l_idx){
      for(int k = 0; k < b_s_r_dmn_type::dmn_size(); ++k){

        if(creation_set_all(l_idx,r_idx,k) != -1){

          creation_set_nonzero_sparse(l_idx,r_idx,k).reserve(1024);
          creation_set_nonzero_sparse(l_idx,r_idx,k).resize(0);

          Hilbert_space_phi_representation_type& rep_r = Hilbert_spaces[r_idx].get_rep();
          Hilbert_space_phi_representation_type& rep_l = Hilbert_spaces[l_idx].get_rep();

          for(int j = 0; j < rep_r.size(); ++j){

            int      sign = 1;
            phi_type phi  = rep_r.get_phi(j);

            if( fermionic_operators_type::create_at(k, phi, sign) ){

              std::vector<int>&          column_index = rep_r.get_indices(j);
              std::vector<complex_type>& column_alpha = rep_r.get_alphas(j);

              int i = rep_l.find(phi);

              if(i < rep_l.size()){

                std::vector<int>&          row_index = rep_l.get_indices(i);
                std::vector<complex_type>& row_alpha = rep_l.get_alphas(i);

                for (int c = 0; c < column_index.size(); ++c){
                  for (int r = 0; r < row_index.size(); ++r){

                    sparse_element_type tmp;
                    tmp.i = row_index[r];
                    tmp.j = column_index[c];
                    tmp.value = conj(row_alpha[r]) * column_alpha[c] * scalar_type(sign);

                    creation_set_nonzero_sparse(l_idx,r_idx,k).push_back(tmp);
                  }
                }
              }
            }
          }
        }

        if(creation_set_nonzero_sparse(l_idx,r_idx,k).size() != 0){
          ++non_zero;

          sort(creation_set_nonzero_sparse(l_idx,r_idx,k));

          merge(creation_set_nonzero_sparse(l_idx,r_idx,k));
        }
      }
    }
  }

  cout << "\n\tnumber of non-zero matrices : " << non_zero << endl;
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::construct_annihilation_set_nonzero_sparse()
{
  if(concurrency.id()==0)
    cout << "\n\t" << __FUNCTION__ << endl;

  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  unsigned long non_zero = 0;

  annihilation_set_nonzero_sparse.reset();

  for(int r_idx = 0; r_idx < Hilbert_spaces.size(); ++r_idx){
    for(int l_idx = 0; l_idx < Hilbert_spaces.size(); ++l_idx){
      for(int k = 0; k < b_s_r_dmn_type::dmn_size(); ++k){

        //         std::vector<int> x, y;

        if(annihilation_set_all(l_idx,r_idx,k) != -1)
          {
            annihilation_set_nonzero_sparse(l_idx,r_idx,k).reserve(1024);

            Hilbert_space_phi_representation_type& rep_r = Hilbert_spaces[r_idx].get_rep();
            Hilbert_space_phi_representation_type& rep_l = Hilbert_spaces[l_idx].get_rep();

            for(int j = 0; j < rep_r.size(); ++j){

              int      sign = 1;
              phi_type phi  = rep_r.get_phi(j);

              if( fermionic_operators_type::annihilate_at(k, phi, sign) ){

                std::vector<int>&          column_index = rep_r.get_indices(j);
                std::vector<complex_type>& column_alpha = rep_r.get_alphas(j);

                int i = rep_l.find(phi);

                if(i < rep_l.size())
                  {
                    std::vector<int>&          row_index = rep_l.get_indices(i);
                    std::vector<complex_type>& row_alpha = rep_l.get_alphas(i);

                    for(int c = 0; c < column_index.size(); ++c){
                      for(int r = 0; r < row_index.size(); ++r){

                        sparse_element_type tmp;
                        tmp.i = row_index[r];
                        tmp.j = column_index[c];
                        tmp.value = conj(row_alpha[r]) * column_alpha[c] * scalar_type(sign);

                        annihilation_set_nonzero_sparse(l_idx,r_idx,k).push_back(tmp);
                      }
                    }

                    //                     if((r_idx==310 and l_idx==377) or
                    //                        (r_idx==377 and l_idx==310))
                    //                       {
                    //                         x.push_back(i);
                    //                         y.push_back(j);
                    //                       }
                  }
              }
            }
          }

        //         if((r_idx==310 and l_idx==377) or
        //            (r_idx==377 and l_idx==310))
        //           SHOW::plot_points(x, y);

        if(annihilation_set_nonzero_sparse(l_idx,r_idx,k).size() != 0)
          {
            ++non_zero;

            sort(annihilation_set_nonzero_sparse(l_idx,r_idx,k));

            merge(annihilation_set_nonzero_sparse(l_idx,r_idx,k));
          }
      }
    }
  }

  cout << "\n\tnumber of non-zero matrices : " << non_zero << endl;
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::compute_sparse_creation(int HS_i, int HS_j, int b_s_r)
{
  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  sparse_creation.resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));

  for(int j = 0; j < Hilbert_spaces[HS_j].size(); ++j)
    for(int i = 0; i < Hilbert_spaces[HS_i].size(); ++i)
      sparse_creation(i,j) = complex_type(0.);

  for(int k = 0; k < creation_set_nonzero_sparse(HS_i,HS_j,b_s_r).size(); ++k){
    sparse_creation(creation_set_nonzero_sparse(HS_i,HS_j,b_s_r)[k].i,
                    creation_set_nonzero_sparse(HS_i,HS_j,b_s_r)[k].j) += creation_set_nonzero_sparse(HS_i,HS_j,b_s_r)[k].value;
  }
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::compute_sparse_annihilation(int HS_i, int HS_j, int b_s_r)
{
  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  sparse_annihilation.resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));

  for(int j = 0; j < Hilbert_spaces[HS_j].size(); ++j)
    for(int i = 0; i < Hilbert_spaces[HS_i].size(); ++i)
      sparse_annihilation(i,j) = complex_type(0.);

  for(int k = 0; k < annihilation_set_nonzero_sparse(HS_i,HS_j,b_s_r).size(); ++k){
    sparse_annihilation(annihilation_set_nonzero_sparse(HS_i,HS_j,b_s_r)[k].i,
                        annihilation_set_nonzero_sparse(HS_i,HS_j,b_s_r)[k].j) += annihilation_set_nonzero_sparse(HS_i,HS_j,b_s_r)[k].value;
  }
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::compute_creation_matrix(int HS_i, int HS_j, int b_s_r,
                                                                                     matrix_type& dense_creation)
{
  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  compute_sparse_creation(HS_i, HS_j, b_s_r);

  dense_creation.resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));
  helper        .resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));

  matrix_type& V_i = Hamiltonian.get_eigen_states()(HS_i);
  matrix_type& V_j = Hamiltonian.get_eigen_states()(HS_j);

  LIN_ALG::GEMM<LIN_ALG::CPU>::execute('N','N', sparse_creation, V_j, helper);
  LIN_ALG::GEMM<LIN_ALG::CPU>::execute('C','N', V_i, helper, dense_creation);
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::compute_creation_matrix_fast(int HS_i, int HS_j, int b_s_r,
                                                                                          matrix_type& dense_creation,
                                                                                          matrix_type& tmp)
{
  //  cout << "\n\n\t" << __FUNCTION__ << "\n\n";

  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  dense_creation.resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));
  tmp           .resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));

  matrix_type& V_i = Hamiltonian.get_eigen_states()(HS_i);
  matrix_type& V_j = Hamiltonian.get_eigen_states()(HS_j);

  {
    for(int j=0; j<tmp.get_number_of_cols(); j++)
      for(int i=0; i<tmp.get_number_of_rows(); i++)
        tmp(i,j)=0.;

    std::vector<sparse_element_type>& sparse_elements = creation_set_nonzero_sparse(HS_i,HS_j,b_s_r);

    for(int k=0; k<sparse_elements.size(); ++k)
      {
        complex_type value = sparse_elements[k].value;

        int i = sparse_elements[k].i;
        int j = sparse_elements[k].j;

        for(int l=0; l<V_j.get_number_of_cols(); l++)
          tmp(i, l) += value*V_j(j, l);
      }
  }

  LIN_ALG::GEMM<LIN_ALG::CPU>::execute('C','N', V_i, tmp, dense_creation);
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::compute_annihilation_matrix(int HS_i, int HS_j, int b_s_r,
                                                                                         matrix_type& dense_annihilation)
{
  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  compute_sparse_annihilation(HS_i, HS_j, b_s_r);

  dense_annihilation.resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));
  helper            .resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));

  matrix_type& V_i = Hamiltonian.get_eigen_states()(HS_i);
  matrix_type& V_j = Hamiltonian.get_eigen_states()(HS_j);

  LIN_ALG::GEMM<LIN_ALG::CPU>::execute('N','N', sparse_annihilation, V_j, helper);
  LIN_ALG::GEMM<LIN_ALG::CPU>::execute('C','N', V_i, helper, dense_annihilation);
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::compute_annihilation_matrix_fast(int HS_i, int HS_j, int b_s_r,
                                                                                              matrix_type& dense_annihilation,
                                                                                              matrix_type& tmp)
{
//   cout << "\n\n\t" << __FUNCTION__ << "\n\n";

  std::vector<Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  dense_annihilation.resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));
  tmp               .resize_no_copy(std::pair<int,int>(Hilbert_spaces[HS_i].size(),Hilbert_spaces[HS_j].size()));

  matrix_type& V_i = Hamiltonian.get_eigen_states()(HS_i);
  matrix_type& V_j = Hamiltonian.get_eigen_states()(HS_j);

//   {
//   dense_annihilation.print_fingerprint();
//   tmp.print_fingerprint();
//   V_i.print_fingerprint();
//   V_j.print_fingerprint();
//   }

  {
    for(int j=0; j<tmp.get_number_of_cols(); j++)
      for(int i=0; i<tmp.get_number_of_rows(); i++)
        tmp(i,j)=0.;

    std::vector<sparse_element_type>& sparse_elements = annihilation_set_nonzero_sparse(HS_i,HS_j,b_s_r);

    for(int k=0; k<sparse_elements.size(); ++k)
      {
        complex_type value = sparse_elements[k].value;

        int i = sparse_elements[k].i;
        int j = sparse_elements[k].j;

        for(int l=0; l<V_j.get_number_of_cols(); l++)
          tmp(i, l) += value*V_j(j, l);
      }
  }

  LIN_ALG::GEMM<LIN_ALG::CPU>::execute('C','N', V_i, tmp, dense_annihilation);
}

template<typename parameter_type, typename ed_options>
bool fermionic_overlap_matrices<parameter_type, ed_options>::check_hermitianess()
{
  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  matrix_type creation;
  matrix_type annihilation;

  for(int HS_j = 0; HS_j < Hilbert_spaces.size(); ++HS_j)
    for(int HS_i = 0; HS_i < Hilbert_spaces.size(); ++HS_i)
      for(int k = 0; k < b_s_r_dmn_type::dmn_size(); ++k){

        if(creation_set_all(HS_i, HS_j, k) != -1){
          assert(annihilation_set_all(HS_j, HS_i, k) != -1);

          compute_creation_matrix    (HS_i, HS_j, k, creation);
          compute_annihilation_matrix(HS_j, HS_i, k, annihilation);

          for(int j = 0; j < Hilbert_spaces[HS_j].size(); ++j)
            for(int i = 0; i < Hilbert_spaces[HS_i].size(); ++i){

              if(abs(creation(i,j) - conj(annihilation(j,i))) > ed_options::get_epsilon())
                return false;
            }
        }
      }

  return true;
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::compute_overlap(int HS_n, int HS_m)
{
  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  matrix_type annihilation_nu;
  matrix_type creation_mu;

  overlap = 0.;

  for(int nu = 0; nu < b_s_r_dmn_type::dmn_size(); ++nu)
    for(int mu = 0; mu < b_s_r_dmn_type::dmn_size(); ++mu)

      if(creation_set_all(HS_m, HS_n, mu) != -1 && annihilation_set_all(HS_n, HS_m, nu) != -1){

        compute_annihilation_matrix(HS_n, HS_m, nu, annihilation_nu);
        compute_creation_matrix    (HS_m, HS_n, mu, creation_mu);

        for(int n = 0; n < Hilbert_spaces[HS_n].size(); ++n)
          for(int m = 0; m < Hilbert_spaces[HS_m].size(); ++m)
            overlap(nu, mu) += annihilation_nu(n,m) * creation_mu(m,n);

      }
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::print_overlap(const char* filename, int HS_n, int HS_m)
{
  cout << "Print overlap of Hilbert-spaces #" << HS_n << " and #" << HS_m << endl;

  std::ofstream data;
  data.open(filename);

  for(int nu = 0; nu < b_s_r_dmn_type::dmn_size(); ++nu){
    for(int mu = 0; mu < b_s_r_dmn_type::dmn_size(); ++mu){
      assert(imag(overlap(nu,mu)) < ed_options::get_epsilon());
      data << real(overlap(nu, mu)) << "\t";
    }
    data << "\n" ;
  }

  data.close();

}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::print_creation_matrix(const char* filename)
{
  std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

  int HS_0 = 0;
  int HS_1 = 0;
  int b_s_r = parameters.get_nu();

  for(int i = 0; i < Hilbert_spaces.size(); ++i){
    if(Hilbert_spaces[i].get_occupation()    == parameters.get_n_0() &&
       Hilbert_spaces[i].get_magnetization() == parameters.get_Sz_0())
      HS_0 = i;

    if(Hilbert_spaces[i].get_occupation()    == parameters.get_n_1() &&
       Hilbert_spaces[i].get_magnetization() == parameters.get_Sz_1())
      HS_1 = i;
  }

  cout << "Print creation matrix for HS_0 = " << HS_0 << ", HS_1 = " << HS_1 << ", nu = " << b_s_r << endl;

  assert(creation_set_all(HS_0, HS_1, b_s_r) != -1);

  //      matrix_type creation_matrix;

  compute_sparse_creation(HS_0, HS_1, b_s_r);

  std::ofstream data;
  data.open(filename);

  for(int l0 = 0; l0 < Hilbert_spaces[HS_0].size(); ++l0){
    for(int l1 = 0; l1 < Hilbert_spaces[HS_1].size(); ++l1){
      assert(abs(imag(sparse_creation(l0,l1))) < 1.e-10);
      data << real(sparse_creation(l0,l1)) << "\t";
    }
    data << "\n" ;
  }

  data.close();

}


// template<typename parameter_type, typename ed_options>
// void fermionic_overlap_matrices<parameter_type, ed_options>::compute_all_creation_matrices()
// {
//   if(concurrency.id()==0)
//     cout << "\n\t" << __FUNCTION__ << endl;

//   std::vector< Hilbert_space_type >& Hilbert_spaces = fermionic_Fock_dmn_type::get_elements();

//   int counter = 0;

//   for(int HS_j = 0; HS_j < Hilbert_spaces.size(); ++HS_j){
//     for(int HS_i = 0; HS_i < Hilbert_spaces.size(); ++HS_i){
//       for(int k = 0; k < b_s_r_dmn_type::dmn_size(); ++k){

//         if(creation_set_all(HS_i,HS_j,k) != -1){

//           ++counter;
//           cout << counter << endl;

//           compute_sparse_creation(HS_i, HS_j, k);
//           compute_dense_creation(HS_i, HS_j, k);
//         }
//       }
//     }
//   }
// }




template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::sort(std::vector< sparse_element_type >& sparse_matrix)
{
  std::sort(sparse_matrix.begin(), sparse_matrix.end());
}

template<typename parameter_type, typename ed_options>
void fermionic_overlap_matrices<parameter_type, ed_options>::merge(std::vector< sparse_element_type >& sparse_matrix)
{
  std::vector< sparse_element_type > merged_sparse_matrix;

  merged_sparse_matrix.push_back(sparse_matrix[0]);

  for(int k = 1; k < sparse_matrix.size(); ++k){

    if(sparse_matrix[k].i == merged_sparse_matrix.back().i && sparse_matrix[k].j == merged_sparse_matrix.back().j)
      merged_sparse_matrix.back().value += sparse_matrix[k].value;

    else
      merged_sparse_matrix.push_back(sparse_matrix[k]);
  }

  sparse_matrix.swap(merged_sparse_matrix);
}



}
}

#endif