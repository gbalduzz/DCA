//-*-C++-*-

#ifndef DOMAIN_SPECIFICATIONS_H
#define DOMAIN_SPECIFICATIONS_H

namespace MATH_ALGORITHMS
{
  template<typename scalartype ,
	   typename elementtype,
	   DOMAIN_REPRESENTATIONS DMN_REP,
	   BASIS_EXPANSIONS       EXP,
	   BOUNDARY_CONDITIONS    BC,
	   ELEMENT_SPACINGS       SPACING>
  class domain_specifications
  {
  public:

    typedef scalartype  scalar_type;
    typedef elementtype element_type;

    const static DOMAIN_REPRESENTATIONS DOMAIN_REPRESENTATION = DMN_REP;
    const static BASIS_EXPANSIONS       BASIS_EXPANSION       = EXP;

    const static BOUNDARY_CONDITIONS    BOUNDARY_CONDITION    = BC;
    const static ELEMENT_SPACINGS       ELEMENT_SPACING       = SPACING;

  public :

    static std::string& to_str()
    {
      static std::string name = MATH_ALGORITHMS::to_str(DOMAIN_REPRESENTATION) + " " +
	MATH_ALGORITHMS::to_str(BASIS_EXPANSION)       + " " +
	MATH_ALGORITHMS::to_str(BOUNDARY_CONDITION)    + " " +
	MATH_ALGORITHMS::to_str(ELEMENT_SPACING);

      return name;
    }

  };
  
  typedef domain_specifications<double,             double , CONTINUOUS, HERMITE_CUBIC_SPLINE, INTERVAL, EQUIDISTANT> interval_dmn_1D_type;
  typedef domain_specifications<double, std::vector<double>, CONTINUOUS, HERMITE_CUBIC_SPLINE, INTERVAL, EQUIDISTANT> interval_dmn_nD_type;

  typedef domain_specifications<double,             double , CONTINUOUS, HERMITE_CUBIC_SPLINE, PERIODIC, EQUIDISTANT> periodic_interval_dmn_1D_type;
  typedef domain_specifications<double, std::vector<double>, CONTINUOUS, HERMITE_CUBIC_SPLINE, PERIODIC, EQUIDISTANT> periodic_interval_dmn_nD_type;

  typedef domain_specifications<double,             double , EXPANSION, HARMONICS, INTERVAL, EQUIDISTANT> harmonic_dmn_1D_type;
  typedef domain_specifications<double, std::vector<double>, EXPANSION, HARMONICS, INTERVAL, EQUIDISTANT> harmonic_dmn_nD_type;

  typedef domain_specifications<double,             double , EXPANSION, LEGENDRE_P, INTERVAL, EQUIDISTANT> legendre_dmn_1D_type;
  typedef domain_specifications<double, std::vector<double>, EXPANSION, LEGENDRE_P, INTERVAL, EQUIDISTANT> legendre_dmn_nD_type;

  typedef domain_specifications<double,             std::pair<int,int>  , EXPANSION, LEGENDRE_LM, INTERVAL, EQUIDISTANT> Y_lm_dmn_1D_type;
  typedef domain_specifications<double, std::vector<std::pair<int,int> >, EXPANSION, LEGENDRE_LM, INTERVAL, EQUIDISTANT> Y_lm_dmn_nD_type;

  typedef domain_specifications<double,             double , DISCRETE, KRONECKER_DELTA, INTERVAL, EQUIDISTANT> discrete_interval_dmn_1D_type;
  typedef domain_specifications<double, std::vector<double>, DISCRETE, KRONECKER_DELTA, INTERVAL, EQUIDISTANT> discrete_interval_dmn_nD_type;

  typedef domain_specifications<double,             double , DISCRETE, KRONECKER_DELTA, PERIODIC, EQUIDISTANT> discrete_periodic_dmn_1D_type;
  typedef domain_specifications<double, std::vector<double>, DISCRETE, KRONECKER_DELTA, PERIODIC, EQUIDISTANT> discrete_periodic_dmn_nD_type;

}

#endif