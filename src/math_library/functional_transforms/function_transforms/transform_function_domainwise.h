//-*-C++-*-

#ifndef BASIS_TRANSFORMATIONS_GENERIC_H
#define BASIS_TRANSFORMATIONS_GENERIC_H

namespace MATH_ALGORITHMS
{
  /*!
   *  \class   
   *  \ingroup 
   *
   *  \author  Peter Staar
   *  \brief   ...
   */
  template<typename domain_input, typename domain_output, 
	   typename type_input  , typename type_output>
  struct TRANSFORM_DOMAINWISE
  {
    const static bool VERBOSE = false;
    
    //typedef typename SWAP_COND<domain_input, type_input, type_output, 0>::Result TRANSFORMED_DOMAIN;
    typedef typename SWAP_FIRST<domain_input, type_input, type_output>::Result TRANSFORMED_DOMAIN;
    
    const static int CURR_DMN_INDEX = IndexOf<typename domain_input      ::this_type, type_input>::value;
    const static int NEXT_DMN_INDEX = IndexOf<typename TRANSFORMED_DOMAIN::this_type, type_input>::value;
        
    typedef typename type_input ::dmn_specifications_type input_specs_type;
    typedef typename type_output::dmn_specifications_type output_specs_type;
    
    const static DOMAIN_REPRESENTATIONS DMN_REP_LHS = input_specs_type ::DOMAIN_REPRESENTATION;
    const static DOMAIN_REPRESENTATIONS DMN_REP_RHS = output_specs_type::DOMAIN_REPRESENTATION;
    
    template<typename scalartype_input, typename scalartype_output>
    static void execute_on_first(FUNC_LIB::function<scalartype_input , domain_input >& f_input, 
				 FUNC_LIB::function<scalartype_output, domain_output>& f_output)
    {
//       GENERIC_ASSERT<IS_EQUAL<TRANSFORMED_DOMAIN, domain_output>::CHECK>::execute();

      if(VERBOSE)
	cout << "\n\n\t" << __FUNCTION__ << "\t" << f_input.get_name() << " --> " << f_output.get_name() << "\n\n";
      
      assert(CURR_DMN_INDEX>-1 and CURR_DMN_INDEX<f_input.signature());
      
      TRANSFORM_DOMAIN<type_input, DMN_REP_LHS, type_output, DMN_REP_RHS, CURR_DMN_INDEX>::execute(f_input,f_output);
    }

    template<typename scalartype_input, typename scalartype_output, typename scalartype_T>
    static void execute_on_first(FUNC_LIB::function<scalartype_input , domain_input >&  f_input, 
				 FUNC_LIB::function<scalartype_output, domain_output>&  f_output,
				 LIN_ALG::matrix<scalartype_T, LIN_ALG::CPU>& T)
    {
//       GENERIC_ASSERT<IS_EQUAL<TRANSFORMED_DOMAIN, domain_output>::CHECK>::execute();

      if(VERBOSE)
	cout << "\n\n\t" << __FUNCTION__ << "\t" << f_input.get_name() << " --> " << f_output.get_name() << "\n\n";
      
      assert(CURR_DMN_INDEX>-1 and CURR_DMN_INDEX<f_input.signature());
      
      TRANSFORM_DOMAIN_PROCEDURE<CURR_DMN_INDEX>::transform(f_input, f_output, T);
    }
    
    template<typename scalartype_input, typename scalartype_output>
    static void execute_on_all(FUNC_LIB::function<scalartype_input , domain_input >& f_input, 
			       FUNC_LIB::function<scalartype_output, domain_output>& f_output)
    {
      if(VERBOSE){
	cout << "\n\n\t" << __FUNCTION__ << "\t" << f_input.get_name() << " --> " << f_output.get_name() << "\n\n";

	f_input.print_fingerprint();
      }

      FUNC_LIB::function<scalartype_output, TRANSFORMED_DOMAIN> f_output_new("f_output_new");
      
      TRANSFORM_DOMAIN<type_input, DMN_REP_LHS, type_output, DMN_REP_RHS, CURR_DMN_INDEX>::execute(f_input, f_output_new);

      if(NEXT_DMN_INDEX == -1)
	{
	  if(VERBOSE)
	    cout << "\n\n\tSTOP\t  start the copy "<< endl;
	  
	  assert(f_output_new.size()==f_output.size());
	  
	  for(int l=0; l<f_output.size(); l++)
	    f_output(l) = f_output_new(l);
	  
	  if(VERBOSE){
	    f_output.print_fingerprint();

	    cout << "\n\n\tSTOP\t finished the copy\n\n\n" << endl;
	  }
	}
      else
	{
	  if(VERBOSE){
	    cout << "\n\n\tSTART\t" << __FUNCTION__ << endl;
	  
	    f_output_new.print_fingerprint();
	  }

	  TRANSFORM_DOMAINWISE<TRANSFORMED_DOMAIN, domain_output, type_input, type_output>::execute_on_all(f_output_new, f_output);
	}
    }

    template<typename scalartype_input, typename scalartype_output, typename scalartype_T>
    static void execute_on_all(FUNC_LIB::function<scalartype_input , domain_input >&  f_input, 
			       FUNC_LIB::function<scalartype_output, domain_output>&  f_output,
			       LIN_ALG::matrix<scalartype_T, LIN_ALG::CPU>& T)
    {
      if(VERBOSE){
	cout << "\n\n\t" << __FUNCTION__ << "\t" << f_input.get_name() << " --> " << f_output.get_name() << "\n\n";

	f_input.print_fingerprint();
      }

      FUNC_LIB::function<scalartype_output, TRANSFORMED_DOMAIN> f_output_new("f_output_new");

      TRANSFORM_DOMAIN_PROCEDURE<CURR_DMN_INDEX>::transform(f_input, f_output_new, T);

      if(NEXT_DMN_INDEX == -1)
	{
	  if(VERBOSE)
	    cout << "\n\n\tSTOP\t  start the copy "<< endl;
	  
	  assert(f_output_new.size()==f_output.size());
	  
	  for(int l=0; l<f_output.size(); l++)
	    f_output(l) = f_output_new(l);
	  
	  if(VERBOSE){
	    f_output.print_fingerprint();

	    cout << "\n\n\tSTOP\t finished the copy\n\n\n" << endl;
	  }
	}
      else
	{
	  if(VERBOSE){
	    cout << "\n\n\tSTART\t" << __FUNCTION__ << endl;
	  
	    f_output_new.print_fingerprint();
	  }

	  TRANSFORM_DOMAINWISE<TRANSFORMED_DOMAIN, domain_output, type_input, type_output>::execute_on_all(f_output_new, f_output, T);
	}
    }

  };

}

#endif