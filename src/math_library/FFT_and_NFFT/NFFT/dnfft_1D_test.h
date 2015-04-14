//-*-C++-*-

#ifndef DNFFT_1D_TEST_H
#define DNFFT_1D_TEST_H

namespace MATH_ALGORITHMS
{
  namespace NFFT
  {

    template<typename scalartype, typename t_dmn_t, typename w_dmn_t>
    class dnfft_1D_test
    {
      typedef dmn_0<dmn<1, int> > p_dmn_t;

    public:

      static void execute()
      {
        srand(55423543);

	double begin =  t_dmn_t::get_elements().front();
	double delta = (t_dmn_t::get_elements().back()-t_dmn_t::get_elements().front());

        std::vector<scalartype> t(0);
        std::vector<scalartype> f(0);
        for(int l=0; l<1.e6; l++)
          {
            t.push_back(t_dmn_t::get_elements()[0] + double(rand())/double(RAND_MAX)*delta);
            //f.push_back(double(rand())/double(RAND_MAX));
	    f.push_back(exp(-2.*M_PI/delta*(t[l]-begin)));
	    
	    //cout << t[l] << "\t" << f[l] << "\n";
          }

        FUNC_LIB::function<std::complex<scalartype>, w_dmn_t> f_w_1("f_w_1");
        FUNC_LIB::function<std::complex<scalartype>, w_dmn_t> f_w_2("f_w_2");
	FUNC_LIB::function<std::complex<scalartype>, w_dmn_t> error("error");

	compute_f_w      (t, f, f_w_1);
        compute_f_w_dnfft(t, f, f_w_2);

	for(int i=0; i<w_dmn_t::dmn_size(); i++)
	  error(i) = log10(abs(f_w_1(i)-f_w_2(i)));

	SHOW::execute(error);

	for(int i=0; i<w_dmn_t::dmn_size(); i++)
	  error(i) = log10(abs(f_w_1(i)-f_w_2(i))/(abs(f_w_1(i))+1.e-6));

	SHOW::execute(error);

	cout << "\n\n\n\t STOP !!!!  \n\n\n" << endl;
	
	throw std::logic_error(__FUNCTION__);
      }

    private:

      static void compute_f_w(std::vector<scalartype>& t,
                              std::vector<scalartype>& f,
                              FUNC_LIB::function<std::complex<scalartype>, w_dmn_t>& f_w)
      {
	cout << __FUNCTION__ << endl;

	std::complex<scalartype> I(0,1);

	{
	  clock_t t0 = clock();

        for(int j=0; j<t.size(); j++)
          for(int i=0; i<w_dmn_t::dmn_size(); i++){

	    scalartype t_val = t[j];
	    scalartype w_val = w_dmn_t::get_elements()[i];

            f_w(i) += f[j]*std::exp(I*t_val*w_val);
	  }
	  clock_t t1 = clock();

	  cout << "\n\n\t time : " << double(t1-t0)/double(CLOCKS_PER_SEC) << "\n";
	}

        SHOW::execute(f_w);
      }

      static void compute_f_w_dnfft(std::vector<scalartype>& t,
                                    std::vector<scalartype>& f,
                                    FUNC_LIB::function<std::complex<scalartype>, w_dmn_t>& f_w)
      {
	cout << __FUNCTION__ << endl;

        FUNC_LIB::function<std::complex<scalartype>, dmn_2<w_dmn_t, p_dmn_t> > f_w_tmp;

        dnfft_1D<scalartype, w_dmn_t, p_dmn_t> nfft_obj;

	std::vector<scalartype> t_vals(0);
	for(int j=0; j<t.size(); j++)
	  t_vals.push_back( (t[j]-t_dmn_t::get_elements().front())/(t_dmn_t::get_elements().back()-t_dmn_t::get_elements().front())-0.5);
	
	{
	  clock_t t0 = clock();
	  
	  //while(true)
	  for(int j=0; j<t.size(); j++){
	    //scalartype t_val = (t[j]-t_dmn_t::get_elements().front())/(t_dmn_t::get_elements().back()-t_dmn_t::get_elements().front())-0.5;
	    
	    nfft_obj.accumulate_at(0, t_vals[j], f[j]);
	  }
	  
	  clock_t t1 = clock();

	  cout << "\n\n\t time : " << double(t1-t0)/double(CLOCKS_PER_SEC) << "\n";
	}

        nfft_obj.finalize(f_w_tmp);

        for(int i=0; i<w_dmn_t::dmn_size(); i++)
          f_w(i) = f_w_tmp(i);

        SHOW::execute(f_w);
      }

    };

  }
}

#endif