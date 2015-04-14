//-*-C++-*-

#ifndef CLUSTER_DOMAIN_H
#define CLUSTER_DOMAIN_H

/*!
 *  \author Peter Staar
 */
template<typename cluster_type>
class cluster_symmetry
{};

template<CLUSTER_REPRESENTATION R>
struct dual_cluster
{};

template<>
struct dual_cluster<MOMENTUM_SPACE>
{
  const static CLUSTER_REPRESENTATION REPRESENTATION = REAL_SPACE;
};

template<>
struct dual_cluster<REAL_SPACE>
{
  const static CLUSTER_REPRESENTATION REPRESENTATION = MOMENTUM_SPACE;
};

template<typename scalar_type, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
struct cluster_specifications
{};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, CLUSTER, MOMENTUM_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::DISCRETE, MATH_ALGORITHMS::KRONECKER_DELTA, 
						 MATH_ALGORITHMS::PERIODIC, MATH_ALGORITHMS::EQUIDISTANT    > dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, CLUSTER, REAL_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::EXPANSION, MATH_ALGORITHMS::HARMONICS, 
						 MATH_ALGORITHMS::PERIODIC , MATH_ALGORITHMS::EQUIDISTANT> dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, LATTICE_SP, MOMENTUM_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::DISCRETE, MATH_ALGORITHMS::KRONECKER_DELTA, 
						 MATH_ALGORITHMS::PERIODIC, MATH_ALGORITHMS::EQUIDISTANT    > dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, LATTICE_SP, REAL_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::EXPANSION, MATH_ALGORITHMS::HARMONICS, 
						 MATH_ALGORITHMS::PERIODIC , MATH_ALGORITHMS::EQUIDISTANT> dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, LATTICE_TP, MOMENTUM_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::DISCRETE, MATH_ALGORITHMS::KRONECKER_DELTA, 
						 MATH_ALGORITHMS::PERIODIC, MATH_ALGORITHMS::EQUIDISTANT    > dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, LATTICE_TP, REAL_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::EXPANSION, MATH_ALGORITHMS::HARMONICS, 
						 MATH_ALGORITHMS::PERIODIC , MATH_ALGORITHMS::EQUIDISTANT> dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, VASP_LATTICE, MOMENTUM_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::DISCRETE, MATH_ALGORITHMS::KRONECKER_DELTA, 
						 MATH_ALGORITHMS::PERIODIC, MATH_ALGORITHMS::EQUIDISTANT    > dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, VASP_LATTICE, REAL_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::EXPANSION, MATH_ALGORITHMS::HARMONICS, 
						 MATH_ALGORITHMS::PERIODIC , MATH_ALGORITHMS::EQUIDISTANT> dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, TMP_CLUSTER, MOMENTUM_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::DISCRETE, MATH_ALGORITHMS::KRONECKER_DELTA, 
						 MATH_ALGORITHMS::PERIODIC, MATH_ALGORITHMS::EQUIDISTANT    > dmn_specifications_type;
};

template<typename scalar_type, CLUSTER_SHAPE S>
struct cluster_specifications<scalar_type, TMP_CLUSTER, REAL_SPACE, S>
{
  typedef MATH_ALGORITHMS::domain_specifications<scalar_type, std::vector<scalar_type>, 
						 MATH_ALGORITHMS::EXPANSION, MATH_ALGORITHMS::HARMONICS, 
						 MATH_ALGORITHMS::PERIODIC , MATH_ALGORITHMS::EQUIDISTANT> dmn_specifications_type;
};

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
class cluster_domain
{
public:

  const static int DIMENSION = D;

  const static CLUSTER_NAMES          NAME           = N;
  const static CLUSTER_REPRESENTATION REPRESENTATION = R;
  const static CLUSTER_SHAPE          SHAPE          = S;

  const static CLUSTER_REPRESENTATION DUAL_REPRESENTATION = dual_cluster<REPRESENTATION>::REPRESENTATION;

  typedef cluster_domain<scalar_type, D, N,      REPRESENTATION, S> this_type;
  typedef cluster_domain<scalar_type, D, N, DUAL_REPRESENTATION, S> dual_type;

  typedef typename cluster_specifications<scalar_type,    N, R, S>::dmn_specifications_type dmn_specifications_type;

  //typedef             scalar_type  scalar_type;
  typedef std::vector<scalar_type> element_type;

public:
  
  static bool& is_initialized();
  
  static int& get_size();

  static std::vector<int>& get_dimensions();

  /*!
   *   The convention is that the basis and superbasis vectors are stored in the columns!
   */
  static scalar_type*& get_basis();
  static scalar_type*& get_super_basis();

  /*!
   *   The convention is that the inverse basis (and inverse superbasis) is defined as the inverse of the basis (superbasis) matrix!
   */
  static scalar_type*& get_inverse_basis();
  static scalar_type*& get_inverse_super_basis();

  static std::vector<element_type>& get_basis_vectors();
  static std::vector<element_type>& get_super_basis_vectors();

  static std::string& get_name();

  static std::vector<element_type>& get_elements();

  static scalar_type& get_volume();

  static int         origin_index();
  
  static int add     (int i, int j);
  static int subtract(int i, int j);

  static LIN_ALG::matrix<int, LIN_ALG::CPU>& get_add_matrix();
  static LIN_ALG::matrix<int, LIN_ALG::CPU>& get_subtract_matrix();

  static void reset();

  template<typename ss_type>
  static void print(ss_type& ss);

//   template<class stream_type>
//   static void to_JSON(stream_type& ss);
};

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
bool& cluster_domain<scalar_type, D, N, R, S>::is_initialized()
{
  static bool initialized = false;
  return initialized;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
int& cluster_domain<scalar_type, D, N, R, S>::get_size()
{
  static int size = 0;
  return size;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
std::vector<int>& cluster_domain<scalar_type, D, N, R, S>::get_dimensions()
{
  static std::vector<int> dimensions;
  return dimensions;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
scalar_type*& cluster_domain<scalar_type, D, N, R, S>::get_basis()
{
  static scalar_type* basis = NULL;
  return basis;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
scalar_type*& cluster_domain<scalar_type, D, N, R, S>::get_super_basis()
{
  static scalar_type* basis = NULL;
  return basis;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
scalar_type*& cluster_domain<scalar_type, D, N, R, S>::get_inverse_basis()
{
  static scalar_type* basis = NULL;
  return basis;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
scalar_type*& cluster_domain<scalar_type, D, N, R, S>::get_inverse_super_basis()
{
  static scalar_type* basis = NULL;
  return basis;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
std::vector<std::vector<scalar_type> >& cluster_domain<scalar_type, D, N, R, S>::get_basis_vectors()
{
  static std::vector<element_type> basis;
  return basis;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
std::vector<std::vector<scalar_type> >& cluster_domain<scalar_type, D, N, R, S>::get_super_basis_vectors()
{
  static std::vector<element_type> super_basis;
  return super_basis;
}
  
template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
std::string& cluster_domain<scalar_type, D, N, R, S>::get_name()
{
  static std::string name = to_str(NAME)+" "+to_str(REPRESENTATION)+" "+to_str(SHAPE)+" (DIMENSION : " + to_str(DIMENSION) + ")";
  return name;
}
  
template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
std::vector<std::vector<scalar_type> >& cluster_domain<scalar_type, D, N, R, S>::get_elements()
{
  static std::vector<element_type> elements(get_size());
  return elements;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
scalar_type& cluster_domain<scalar_type, D, N, R, S>::get_volume()
{
  static scalar_type volume = 0;
  return volume;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
int cluster_domain<scalar_type, D, N, R, S>::origin_index()
{
  static int index = cluster_operations::origin_index(get_elements(), SHAPE);
  assert(VECTOR_OPERATIONS::L2_NORM(get_elements()[index])<1.e-6);
  return index;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
int cluster_domain<scalar_type, D, N, R, S>::add(int i, int j)
{
  static LIN_ALG::matrix<int, LIN_ALG::CPU>& A = get_add_matrix();
  return A(i,j);
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
LIN_ALG::matrix<int, LIN_ALG::CPU>& cluster_domain<scalar_type, D, N, R, S>::get_add_matrix()
{
  assert(SHAPE==BRILLOUIN_ZONE);
  static LIN_ALG::matrix<int, LIN_ALG::CPU> A("add", get_size());
  return A;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
int cluster_domain<scalar_type, D, N, R, S>::subtract(int i, int j)
{
  static LIN_ALG::matrix<int, LIN_ALG::CPU>& A = get_subtract_matrix();
  return A(i,j);
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
LIN_ALG::matrix<int, LIN_ALG::CPU>& cluster_domain<scalar_type, D, N, R, S>::get_subtract_matrix()
{
  assert(SHAPE==BRILLOUIN_ZONE);
  static LIN_ALG::matrix<int, LIN_ALG::CPU> A("subtract", get_size());
  return A;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
void cluster_domain<scalar_type, D, N, R, S>::reset()
{
  get_size() = 0;
  get_name() = "";
  
  get_elements().resize(0);
  
  if(get_basis() != NULL)
    delete [] get_basis();

  if(get_super_basis() != NULL)
    delete [] get_super_basis();

  if(get_inverse_basis() != NULL)
    delete [] get_inverse_basis();

  if(get_inverse_super_basis() != NULL)
    delete [] get_inverse_super_basis();

  is_initialized() = false;
}

template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
template<typename ss_type>
void cluster_domain<scalar_type, D, N, R, S>::print(ss_type& ss)
{
  ss << scientific;
  ss.precision(6);

  ss << "\t name        : " << get_name() << "\n";
  ss << "\t name (dual) : " << dual_type::get_name() << "\n\n";

  ss << "\t size        : " << get_size() << "\n\n";

  ss << "\t\t\t" << to_str(REPRESENTATION) << "\t\t\t|\t" << to_str(DUAL_REPRESENTATION) << "\n";
  ss << "\t origin-index : " << origin_index() << "\t\t\t\t|\t" << dual_type::origin_index() << "\n";
  ss << "\t volume       : " << get_volume()   << "\t\t\t|\t" << dual_type::get_volume  () << "\n\n";

  ss << "\t basis : \n";
  for(int d0=0; d0<DIMENSION; d0++){
    ss << "\t\t\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << get_basis()[d0+d1*DIMENSION] << "\t";
    ss << "|\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << dual_type::get_basis()[d0+d1*DIMENSION] << "\t";
    ss << "\n";
  }
  ss << "\n";

  ss << "\t super-basis : \n";
  for(int d0=0; d0<DIMENSION; d0++){
    ss << "\t\t\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << get_super_basis()[d0+d1*DIMENSION] << "\t";
    ss << "|\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << dual_type::get_super_basis()[d0+d1*DIMENSION] << "\t";
    ss << "\n";
  }
  ss << "\n";

  ss << "\t inverse-basis : \n";
  for(int d0=0; d0<DIMENSION; d0++){
    ss << "\t\t\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << get_inverse_basis()[d0+d1*DIMENSION] << "\t";
    ss << "|\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << dual_type::get_inverse_basis()[d0+d1*DIMENSION] << "\t";
    ss << "\n";
  }
  ss << "\n";

  ss << "\t inverse-super-basis : \n";
  for(int d0=0; d0<DIMENSION; d0++){
    ss << "\t\t\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << get_inverse_super_basis()[d0+d1*DIMENSION] << "\t";
    ss << "|\t";
    for(int d1=0; d1<DIMENSION; d1++)
      ss << dual_type::get_inverse_super_basis()[d0+d1*DIMENSION] << "\t";
    ss << "\n";
  }
  ss << "\n\n";

  if(NAME == CLUSTER)
    {
      for(int l=0; l<get_size(); l++)
	{
	  ss << "\t" << l << "\t|\t";
	  VECTOR_OPERATIONS::PRINT(get_elements()[l]);
	  ss << "\t";
	  VECTOR_OPERATIONS::PRINT(dual_type::get_elements()[l]);
	  ss << "\n";
	}    
      ss << "\n\n\t" << to_str(REPRESENTATION) << " k-space symmetries : \n\n";
      {
	for(int i=0; i<this_type::get_size(); ++i){
	  for(int j=0; j<cluster_symmetry<this_type>::b_dmn_t::dmn_size(); ++j){
	    
	    ss << "\t" << i << ", " << j << "\t|\t";
	    
	    for(int l=0; l<cluster_symmetry<this_type>::sym_super_cell_dmn_t::dmn_size(); ++l)
	      ss << "\t" << cluster_symmetry<this_type>::get_symmetry_matrix()(i,j, l).first 
		 << ", " << cluster_symmetry<this_type>::get_symmetry_matrix()(i,j, l).second;

	    ss << "\n";
	  }
	}
	ss << "\n";
      }
      
      ss << "\n\n\t" << to_str(DUAL_REPRESENTATION) << " symmetries : \n\n";
      {
	for(int i=0; i<dual_type::get_size(); ++i){
	  for(int j=0; j<cluster_symmetry<dual_type>::b_dmn_t::dmn_size(); ++j){
	    
	    ss << "\t" << i << ", " << j << "\t|\t";
	    
	    for(int l=0; l<cluster_symmetry<dual_type>::sym_super_cell_dmn_t::dmn_size(); ++l)
	      ss << "\t" << cluster_symmetry<dual_type>::get_symmetry_matrix()(i,j, l).first 
		 << ", " << cluster_symmetry<dual_type>::get_symmetry_matrix()(i,j, l).second;
	    
	    ss << "\n";
	  }
	}
	ss << "\n";
      }
    }
  
  //get_add_matrix().print();

  //get_subtract_matrix().print();
}

// template<typename scalar_type, int D, CLUSTER_NAMES N, CLUSTER_REPRESENTATION R, CLUSTER_SHAPE S>
// template<typename ss_type>
// void cluster_domain<scalar_type, D, N, R, S>::to_JSON(ss_type& ss)
// {

// }

#endif