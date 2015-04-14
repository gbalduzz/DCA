//-*-C++-*-

#ifndef PARTICLE_NUMBER_DOMAIN_H_
#define PARTICLE_NUMBER_DOMAIN_H_

/*!
 *   \author: Peter Staar
 */
template<typename band_dmn_t, typename cluster_t, typename e_spin_t>
class particle_number_domain 
{
public:

  typedef int element_type;

public:

  static int&                 get_size();
  static std::vector<int>& get_elements();

private:

  static std::vector<int>& initialize_elements();
};

template<typename band_dmn_t, typename cluster_t, typename e_spin_t>
int& particle_number_domain<band_dmn_t, cluster_t, e_spin_t>::get_size()  
{
  static int size = band_dmn_t::dmn_size()*cluster_t::dmn_size()*e_spin_t::dmn_size()+1;
  return size;
}

template<typename band_dmn_t, typename cluster_t, typename e_spin_t>
std::vector<int>& particle_number_domain<band_dmn_t, cluster_t, e_spin_t>::get_elements()
{
  static std::vector<int>& v = initialize_elements();
  return v;
}

template<typename band_dmn_t, typename cluster_t, typename e_spin_t>
std::vector<int>& particle_number_domain<band_dmn_t, cluster_t, e_spin_t>::initialize_elements()
{
  static std::vector<int> v(get_size());

  for(int i=0; i<get_size(); i++)
    v[i] = i;

  return v;
}

#endif