//-*-C++-*-

#ifndef ADVANCED_FERMIONIC_PHI_STATE_H
#define ADVANCED_FERMIONIC_PHI_STATE_H

namespace DCA
{
  namespace ADVANCED_EXACT_DIAGONALIZATION
  {
    template<typename parameter_type, typename ed_options, phi_names phi_name>       // N: size of bitset sequence
    struct phi_state
    {};

    template<typename parameter_type, typename ed_options>       // N: size of bitset sequence
    struct phi_state<parameter_type, ed_options, PHI_SINGLET>
    {    
    public:

      typedef typename ed_options::scalar_type  scalar_type;
      typedef typename ed_options::complex_type complex_type;
 
      typedef typename ed_options::phi_type phi_type;

    public:
      
      phi_type     phi;
      complex_type alpha;
    };


    template<typename parameter_type, typename ed_options>       // N: size of bitset sequence
    struct phi_state<parameter_type, ed_options, PHI_MULTIPLET>
    {
    public:

      typedef typename ed_options::scalar_type  scalar_type;
      typedef typename ed_options::complex_type complex_type;
 
      typedef typename ed_options::phi_type phi_type;
 
    public:

      void sort();

    public:

      phi_type                  phi;

      std::vector<int>          index;
      std::vector<complex_type> alpha;
    };

    template<typename parameter_type, typename ed_options>
    void phi_state<parameter_type, ed_options, PHI_MULTIPLET>::sort()
    {
      std::vector<int>          sorted_index;
      std::vector<complex_type> sorted_alpha;

      for(int i = 0; i < index.size(); ++i)
	{
	  int idx = 0;
	
	  while(idx < sorted_index.size() && index[i] >= sorted_index[idx])
	    {
	      ++idx;
	    }
	  
	  sorted_index.insert(sorted_index.begin()+idx, index[i]);
	  sorted_alpha.insert(sorted_alpha.begin()+idx, alpha[i]);
	}
      
      index.swap(sorted_index);
      alpha.swap(sorted_alpha);
    }

  }

}

#endif