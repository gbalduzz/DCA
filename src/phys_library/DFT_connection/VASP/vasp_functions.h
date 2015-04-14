//-*-C++-*-

#ifndef DFT_VASP_DATA_H
#define DFT_VASP_DATA_H

namespace DFT
{
  namespace VASP
  {

    struct t_ij_element
    {
      int    oi;
      int    oj;

      int    dRx;
      int    dRy;
      int    dRz;

      //       double tijRe;
      //       double tijIm;

      std::complex<double> value;

      void print(){
        cout << "\t" << oi << "\t" << oi
             << "\t" << dRx << "\t" << dRy << "\t" << dRz
             << "\t" << value << "\n";
      }
    };

    bool operator<(const t_ij_element& lhs, const t_ij_element& rhs)
    {
      return (abs(lhs.value)>abs(rhs.value));
    }

    /*!
     *   \ingroup  PARAMETERS
     *
     *   \author   Peter Staar, Long Zhang
     *   \brief    ...
     */
    template<class parameters_type>
    class data
    {
#include "type_definitions.h"

      //typedef typename k_vasp::parameter_type::dual_type           vasp_r_cluster_type;
      typedef dmn_0<centered_cluster_domain<vasp_r_cluster_type> > vasp_r_centered_dmn;

    public:

      data(parameters_type& parameters_ref);

      ~data();

      void print();

      void execute();

      void construct_bloch_hamiltonians();

      void compute_band_structure();

      void check_bloch_hamiltonians();

      void downfold_bloch_hamiltonians();

      void construct_t_ij();

      void compute_vasp_band_structure();
      void compute_dmft_band_structure();

    public:

      parameters_type& parameters;

      function<double, k_vasp> kx;
      function<double, k_vasp> ky;
      function<double, k_vasp> kz;

      function<double, dmn_2<k_vasp, b_vasp> > band_e;
      function<double, dmn_2<k_vasp, b_vasp> > band_o;

      function<double, dmn_3<k_vasp, b_vasp, o_vasp> > proj_magni; // proj_magni[nKpoints][nBands][nOrbi]
      function<double, dmn_3<k_vasp, b_vasp, o_vasp> > proj_phaRe; // proj_phaRe[nKpoints][nBands][nOrbi]
      function<double, dmn_3<k_vasp, b_vasp, o_vasp> > proj_phaIm; // proj_phaIm[nKpoints][nBands][nOrbi]

      function<double, dmn_3<k_vasp, b_dmft, o_dmft> > projection_re; // projection_Re[nKpoints][nCorrBands][nCorrOrbis]
      function<double, dmn_3<k_vasp, b_dmft, o_dmft> > projection_im; // projection_Im[nKpoints][nCorrBands][nCorrOrbis]

      function<std::complex<double>, dmn_3<k_vasp, b_dmft, o_dmft> > projector;

      function<double, dmn_3<k_vasp, b_dmft, b_dmft> > BlochHami   ; // BlochHami     [nKpoints][nCorrBands][nCorrBands]

      function<double, dmn_3<k_vasp, o_dmft, o_dmft> > dfBlochHami_re;// dfBlochHamiRe  [nKpoints][nCorrOrbis][nCorrOrbis]
      function<double, dmn_3<k_vasp, o_dmft, o_dmft> > dfBlochHami_im;// dfBlochHamiIm  [nKpoints][nCorrOrbis][nCorrOrbis]

      function<std::complex<double>, dmn_3<k_vasp, o_dmft, o_dmft> > H_0 ; // BlochHami     [nKpoints][nCorrBands][nCorrBands]

      //function<double, dmn_3<k_vasp, b_dmft, b_dmft> > dfBlochHami_re;// dfBlochHamiRe  [nKpoints][nCorrOrbis][nCorrOrbis]

      function<double, dmn_2<o_dmft, b_dmft> > p1_re;//[nCorrOrbis][nCorrBands];   // orbital x band
      function<double, dmn_2<o_dmft, b_dmft> > p1_im;//[nCorrOrbis][nCorrBands];

      function<double, dmn_2<b_dmft, o_dmft> > p2_re;//[nCorrBands][nCorrOrbis];   // band x orbital
      function<double, dmn_2<b_dmft, o_dmft> > p2_im;//[nCorrBands][nCorrOrbis];
      function<double, dmn_2<b_dmft, o_dmft> > p3_re;//[nCorrBands][nCorrOrbis];
      function<double, dmn_2<b_dmft, o_dmft> > p3_im;//[nCorrBands][nCorrOrbis];

      function<double, dmn_2<b_dmft, b_dmft> > nn_re;//[nCorrBands][nCorrBands];    // band x band
      function<double, dmn_2<b_dmft, b_dmft> > nn_im;//[nCorrBands][nCorrBands];

      function<double, dmn_2<o_dmft, o_dmft> > oo_re;//[nCorrOrbis][nCorrOrbis];    // obital x obital
      function<double, dmn_2<o_dmft, o_dmft> > oo_im;//[nCorrOrbis][nCorrOrbis];
      function<double, dmn_2<o_dmft, o_dmft> >  o_re;//[nCorrOrbis][nCorrOrbis];
      function<double, dmn_2<o_dmft, o_dmft> >  o_im;//[nCorrOrbis][nCorrOrbis];

      //       std::vector<int>    vec_oi;
      //       std::vector<int>    vec_oj;
      //       std::vector<int>    vec_dRx;
      //       std::vector<int>    vec_dRy;
      //       std::vector<int>    vec_dRz;
      //       std::vector<double> vec_tijRe;
      //       std::vector<double> vec_tijIm;

      std::vector<t_ij_element> t_ij_vector;

      function<double, dmn_2<bz_cut, b_vasp> > E_0_vasp;
      function<double, dmn_2<bz_cut, o_dmft> > E_0_dmft;

      function<std::complex<double>, dmn_3<bz_cut, b_vasp, b_vasp> > H_0_vasp;
      function<std::complex<double>, dmn_3<bz_cut, o_dmft, o_dmft> > H_0_dmft;
    };

    template<class parameters_type>
    data<parameters_type>::data(parameters_type& parameters_ref):
      parameters(parameters_ref),

      proj_magni("proj_magni"),

      proj_phaRe("proj_phaRe"),
      proj_phaIm("proj_phaIm"),

      projection_re("projection_re"),
      projection_im("projection_im"),

      projector("projector"),

      BlochHami("BlochHami"),

      dfBlochHami_re("BlochHami_re"),
      dfBlochHami_im("BlochHami_im")
    {
      cout << "\t" << __FUNCTION__ << endl;

      cout << scientific;
      cout.precision(6);

      std::vector<std::vector<double> > k_vecs;

      std::vector<double>& b0 = k_vasp::parameter_type::get_super_basis_vectors()[0];
      std::vector<double>& b1 = k_vasp::parameter_type::get_super_basis_vectors()[1];
      std::vector<double>& b2 = k_vasp::parameter_type::get_super_basis_vectors()[2];

      if(parameters.get_coordinate_type()=="default")
        {
          std::vector<double> k0(3);
          std::vector<double> k1(3);
          std::vector<double> k2(3);
          std::vector<double> k3(3);

          for(int i=0; i<3; i++)
            {
              k0[i]  = 0    *b0[i]+0.   *b1[i]+0.   *b2[i];
              k1[i]  = 1./2.*b0[i]+1./2.*b1[i]+1./2.*b2[i];
              k2[i]  = 1./2.*b0[i]+1./2.*b1[i]+0.   *b2[i];
              k3[i]  = 1./2.*b0[i]+0.   *b1[i]+0.   *b2[i];
            }

          k_vecs.resize(0);

          k_vecs.push_back(k0);
          k_vecs.push_back(k1);
          k_vecs.push_back(k2);
          k_vecs.push_back(k3);
          k_vecs.push_back(k0);
        }
      else
        {
// 	  std::vector<std::vector<double> > c_vecs = parameters.get_Brillouin_zone_vectors();

// 	  std::vector<double> k0(3);
// 	  for(int j=0; j<3; j++)
// 	    k0[j]=0.;

// 	  for(int i=0; i<c_vecs.size(); i++){
// 	    for(int j=0; j<3; j++)
// 	      k0[j] += (c_vecs[i][j]*b0[j]+c_vecs[i][j]*b1[j]+c_vecs[i][j]*b2[j]);

// 	    k_vecs.push_back(k0);
// 	  }

	  k_vecs = parameters.get_Brillouin_zone_vectors();
        }


      bz_cut::parameter_type::initialize(k_vecs);

      E_0_vasp.reset();
      E_0_dmft.reset();

      H_0_vasp.reset();
      H_0_dmft.reset();
    }

    template<class parameters_type>
    data<parameters_type>::~data()
    {}

    template<class parameters_type>
    void data<parameters_type>::print()
    {
      cout << "\t" << __FUNCTION__ << endl;

      BlochHami.print_fingerprint();
      projector.print_fingerprint();

      proj_magni.print_fingerprint();

      proj_phaRe.print_fingerprint();
      proj_phaIm.print_fingerprint();

      projection_re.print_fingerprint();
      projection_im.print_fingerprint();
    }

    template<class parameters_type>
    void data<parameters_type>::execute()
    {
      print();

      construct_bloch_hamiltonians();

      check_bloch_hamiltonians();

      downfold_bloch_hamiltonians();

      compute_band_structure();

      construct_t_ij();
    }

    template<class parameters_type>
    void data<parameters_type>::construct_bloch_hamiltonians()
    {
      cout << "\t" << __FUNCTION__ << endl;

      double norm=0;

      int corrband_min = parameters.get_lower_index_of_chosen_bands();
      int corrband_max = parameters.get_upper_index_of_chosen_bands();

      std::vector<int> CorrOrbiIndex = parameters.get_correlated_orbitals();

      for(int ik=0; ik<parameters.get_nKpoints(); ik++ )
        {
          for(int ib=0; ib<parameters.get_nBands(); ib++ )
            {
              if(ib>=corrband_min && ib<=corrband_max )
                {      // chosen bands
                  for(int io=0; io<CorrOrbiIndex.size(); io++ )
                    {
                      norm = sqrt(   proj_phaRe(ik,ib,CorrOrbiIndex[io])*proj_phaRe(ik,ib,CorrOrbiIndex[io])
                                     + proj_phaIm(ik,ib,CorrOrbiIndex[io])*proj_phaIm(ik,ib,CorrOrbiIndex[io]) );

                      if ( norm < parameters.get_epsilon() )
                        {
                          projection_re(ik,ib-corrband_min,io) = 0.0;
                          projection_im(ik,ib-corrband_min,io) = 0.0;

                          projector(ik,ib-corrband_min,io) = 0.0;
                        }
                      else
                        {
                          projection_re(ik,ib-corrband_min,io)
                            = sqrt(proj_magni(ik,ib,CorrOrbiIndex[io])) * proj_phaRe(ik,ib,CorrOrbiIndex[io]) / norm;

                          projection_im(ik,ib-corrband_min,io)
                            = sqrt(proj_magni(ik,ib,CorrOrbiIndex[io])) * proj_phaIm(ik,ib,CorrOrbiIndex[io]) / norm;

                          double z_re = sqrt(proj_magni(ik,ib,CorrOrbiIndex[io])) * proj_phaRe(ik,ib,CorrOrbiIndex[io]) / norm;
                          double z_im = sqrt(proj_magni(ik,ib,CorrOrbiIndex[io])) * proj_phaIm(ik,ib,CorrOrbiIndex[io]) / norm;

                          projector(ik,ib-corrband_min,io) = std::complex<double>(z_re, z_im);
                        }

                      if ( fabs(projection_re(ik,ib-corrband_min,io)) < parameters.get_epsilon() )
                        projection_re(ik,ib-corrband_min,io) = 0.0;

                      if ( fabs(projection_im(ik,ib-corrband_min,io)) < parameters.get_epsilon() )
                        projection_im(ik,ib-corrband_min,io) = 0.0;

                    } // chosen orbitals
                } // chosen bands

            }  // ib
        }  // ik

      {
        for(int ik=0; ik<parameters.get_nKpoints(); ik++ ) {
          for(int ib=0;  ib<parameters.get_nBands();  ib++ ) {
            if ( ib>=corrband_min && ib<=corrband_max ) {
              BlochHami(ik,ib-corrband_min,ib-corrband_min) = band_e(ik,ib);
            }
          }
        }
      }

      //       if(false)
      //         {
      //           Gnuplot plot_obj;

      //           for(int ib=0;  ib<parameters.get_nBands();  ib++ )
      //             {
      //               if ( ib>=corrband_min && ib<=corrband_max )
      //                 {
      //                   std::vector<double> x,y;
      //                   for(int ik=0; ik<parameters.get_nKpoints(); ik++ )
      //                     {
      //                       x.push_back(ik);
      //                       y.push_back(band_e(ik,ib));
      //                     }

      //                   SHOW::plot_points(plot_obj, x, y, "SrVO3", "lines");
      //                 }
      //             }
      //         }

      {
        cout << "\n\t checking projector \n\n";

        for(int ik=0; ik<parameters.get_nKpoints(); ik++ )
          {
            for(int ib=0; ib<b_dmft::dmn_size(); ib++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                if(abs(real(projector(ik,ib,io))-projection_re(ik,ib,io))>1.e-6)
                  throw std::logic_error(__FUNCTION__);

            for(int ib=0; ib<b_dmft::dmn_size(); ib++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                if(abs(imag(projector(ik,ib,io))-projection_im(ik,ib,io))>1.e-6)
                  throw std::logic_error(__FUNCTION__);
          }


      }
    }

    template<class parameters_type>
    void data<parameters_type>::compute_band_structure()
    {
      cout << "\t" << __FUNCTION__ << endl;

      {
        compute_vasp_band_structure();
        compute_dmft_band_structure();
      }

      if(true)
        {
          int corrband_min = parameters.get_lower_index_of_chosen_bands();
          int corrband_max = parameters.get_upper_index_of_chosen_bands();

          Gnuplot plot_obj;

          //plot_obj.set_style("lines black");

          for(int ib=0; ib<parameters.get_nBands();  ib++ )
            {
              if ( ib>=corrband_min && ib<=corrband_max )
                {
                  std::vector<double> x,y;
                  for(int ik=0; ik<bz_cut::dmn_size(); ik++ )
                    {
                      x.push_back(ik);
                      y.push_back(E_0_vasp(ik,ib));
                    }

                  SHOW::plot_points(plot_obj, x, y, "correlated-vasp-bands", "lines lc -1 lt 1 lw 4");
                }
            }

          //plot_obj.set_style("points red");

          for(int i=0; i<o_dmft::dmn_size(); i++)
            {
              std::vector<double> x,y;

              for(int ik=0; ik<bz_cut::dmn_size(); ik++ )
                {
                  x.push_back(ik);
                  y.push_back(E_0_dmft(ik,i));
                }

              SHOW::plot_points(plot_obj, x, y, "dmft-lines", "lines lc 1 lt 3 lw 2");
            }

          //plot_obj.savetops("band-structure");
	  
	  plot_obj.cmd("set terminal pdf");
	  plot_obj.cmd("set output \"./pic.pdf\"");
        }
    }

    template<class parameters_type>
    void data<parameters_type>::compute_vasp_band_structure()
    {
      cout << "\t" << __FUNCTION__ << endl;

      vasp_r_centered_dmn::parameter_type::initialize();

      function<double, dmn_2<vasp_r_centered_dmn, b_vasp> > tmp;

      MATH_ALGORITHMS::TRANSFORM<k_vasp, vasp_r_centered_dmn>::execute(band_e, tmp);

      for(int j=0; j<b_vasp::dmn_size(); j++)
        for(int r_ind=0; r_ind<vasp_r_centered_dmn::dmn_size(); r_ind++)
          tmp(r_ind, j) *= vasp_r_centered_dmn::parameter_type::get_weights()[r_ind];

      MATH_ALGORITHMS::TRANSFORM<vasp_r_centered_dmn, bz_cut>::execute(tmp, E_0_vasp);

      //       if(true)
      //         {
      //           Gnuplot plot_obj;

      //           for(int ib=0; ib<parameters.get_nBands();  ib++ )
      //             {
      //               //if ( ib>=corrband_min && ib<=corrband_max )
      //               {
      //                 std::vector<double> x,y;
      //                 for(int ik=0; ik<bz_cut::dmn_size(); ik++ )
      //                   {
      //                     x.push_back(ik);
      //                     y.push_back(E_0_vasp(ik,ib));
      //                   }

      //                 SHOW::plot_points(plot_obj, x, y, "SrVO3", "vasp-bands");
      //               }
      //             }
      //         }

      //       if(true)
      //         {
      //           int corrband_min = parameters.get_lower_index_of_chosen_bands();
      //           int corrband_max = parameters.get_upper_index_of_chosen_bands();

      //           Gnuplot plot_obj;

      //           for(int ib=0; ib<parameters.get_nBands();  ib++ )
      //             {
      //               if ( ib>=corrband_min && ib<=corrband_max )
      //                 {
      //                   std::vector<double> x,y;
      //                   for(int ik=0; ik<bz_cut::dmn_size(); ik++ )
      //                     {
      //                       x.push_back(ik);
      //                       y.push_back(E_0_vasp(ik,ib));
      //                     }

      //                   SHOW::plot_points(plot_obj, x, y, "SrVO3", "correlated-vasp-bands");
      //                 }
      //             }
      //         }
    }

    template<class parameters_type>
    void data<parameters_type>::compute_dmft_band_structure()
    {
      cout << "\t" << __FUNCTION__ << endl;

      vasp_r_centered_dmn::parameter_type::initialize();

      function<double, dmn_3<vasp_r_centered_dmn, o_dmft, o_dmft> > tmp;

      MATH_ALGORITHMS::TRANSFORM<k_vasp, vasp_r_centered_dmn>::execute(H_0, tmp);

      for(int j=0; j<o_dmft::dmn_size(); j++)
        for(int i=0; i<o_dmft::dmn_size(); i++)
          for(int r_ind=0; r_ind<vasp_r_centered_dmn::dmn_size(); r_ind++)
            tmp(r_ind, i, j) *= vasp_r_centered_dmn::parameter_type::get_weights()[r_ind];

      MATH_ALGORITHMS::TRANSFORM<vasp_r_centered_dmn, bz_cut>::execute(tmp, H_0_dmft);

      //       H_0_dmft.print_fingerprint();

      { // compute the bands ...

        LIN_ALG::vector<             double , LIN_ALG::CPU> L_vec(o_dmft::dmn_size());
        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> H_mat(o_dmft::dmn_size());
        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> V_mat(o_dmft::dmn_size());

        for(int l=0; l<bz_cut::dmn_size(); l++)
          {
            for(int i=0; i<o_dmft::dmn_size(); i++)
              for(int j=0; j<o_dmft::dmn_size(); j++)
                H_mat(i,j) = H_0_dmft(l,i,j);

            LIN_ALG::GEEV<LIN_ALG::CPU>::execute('N', 'U', H_mat, L_vec, V_mat);

            for(int i=0; i<o_dmft::dmn_size(); i++)
              E_0_dmft(l, i) = L_vec[i];
          }
      }

      //       if(true)
      //         {
      //           Gnuplot plot_obj;

      //           for(int i=0; i<o_dmft::dmn_size(); i++)
      //             {
      //               std::vector<double> x,y;

      //               for(int ik=0; ik<bz_cut::dmn_size(); ik++ )
      //                 {
      //                   x.push_back(ik);
      //                   y.push_back(E_0_vasp(ik,ib));
      //                 }

      //               SHOW::plot_points(plot_obj, x, y, "SrVO3", "dmft-lines");
      //             }
      //         }
    }

    template<class parameters_type>
    void data<parameters_type>::check_bloch_hamiltonians()
    {
      cout << "\t" << __FUNCTION__ << endl;

      double xx, yy;

      for(int ib=0; ib<b_dmft::dmn_size(); ib++ ) {    // no x nb, zero
        for(int io=0; io<o_dmft::dmn_size(); io++ ) {
          p1_re(io,ib) = 0.;
          p1_im(io,ib) = 0.;
          p2_re(ib,io) = 0.;
          p2_im(ib,io) = 0.;
        }
      }

      for(int i=0; i<b_dmft::dmn_size(); i++ ) {       // nb x nb, diagonal unit
        for(int j=0; j<b_dmft::dmn_size(); j++ ) {
          nn_re(i,j) = 0.;
          nn_im(i,j) = 0.;
          if (i==j) nn_re(i,j) = 1.0;
        }
      }

      for(int i=0; i<o_dmft::dmn_size(); i++ ) {       // no x no, zero
        for(int j=0; j<o_dmft::dmn_size(); j++ ) {
          oo_re(i,j) = 0.;
          oo_im(i,j) = 0.;
          o_re(i,j) = 0.;
          o_im(i,j) = 0.;
        }
      }

      // p1 = p(k)
      // p2 = p_adjoint(k)
      // oo = p1 * ( nn(k) * p2 )

      // loop k
      for(int ik=0; ik<parameters.get_nKpoints(); ik++ ) {

        // -----------------------------------
        for(int ib=0; ib<b_dmft::dmn_size(); ib++ ) {         // p1 and p2
          for(int io=0; io<o_dmft::dmn_size(); io++ ) {

            p1_re(io,ib) =  projection_re(ik,ib,io);
            p1_im(io,ib) =  projection_im(ik,ib,io);

            p2_re(ib,io) =  p1_re(io,ib);
            p2_im(ib,io) = -p1_im(io,ib);
          }
        }

        // -----------------------------------
        for(int io=0; io<o_dmft::dmn_size(); io++ ) {         // p3 = nn(k) * p2
          for(int ib=0; ib<b_dmft::dmn_size(); ib++ ) {

            xx = 0.;
            yy = 0.;
            for(int l=0; l<b_dmft::dmn_size(); l++ ) {
              xx = xx + ( nn_re(ib,l)*p2_re(l,io) - nn_im(ib,l)*p2_im(l,io) );
              yy = yy + ( nn_im(ib,l)*p2_re(l,io) + nn_re(ib,l)*p2_im(l,io) );
            }
            p3_re(ib,io) = xx;
            p3_im(ib,io) = yy;

          }
        }

        // -----------------------------------
        for(int i=0; i<o_dmft::dmn_size(); i++ ) {           // oo = p1 * p3
          for(int j=0; j<o_dmft::dmn_size(); j++ ) {

            xx = 0.;
            yy = 0.;
            for(int l=0; l<b_dmft::dmn_size(); l++ ) {
              xx = xx + ( p1_re(i,l)*p3_re(l,j) - p1_im(i,l)*p3_im(l,j) );
              yy = yy + ( p1_im(i,l)*p3_re(l,j) + p1_re(i,l)*p3_im(l,j) );
            }
            oo_re(i,j) = xx;
            oo_im(i,j) = yy;

          }
        }

        // -----------------------------------  scale the projection
        double scale[o_dmft::dmn_size()];
        for(int io=0; io<o_dmft::dmn_size(); io++ )
          {
            scale[io] = sqrt( fabs( oo_re(io,io) ) );

            if (scale[io]<1.0e-8)
              scale[io]=1.0;
          }

        for(int io=0; io<o_dmft::dmn_size(); io++ ) {
          for(int ib=0; ib<b_dmft::dmn_size(); ib++ ) {
            projection_re(ik,ib,io) = projection_re(ik,ib,io) / scale[io] ;
            projection_im(ik,ib,io) = projection_im(ik,ib,io) / scale[io] ;
          }
        }

        // -----------------------------------
        for(int i=0; i<o_dmft::dmn_size(); i++ ) {           // k sum
          for(int j=0; j<o_dmft::dmn_size(); j++ ) {
            o_re(i,j) = o_re(i,j) + oo_re(i,j);
            o_im(i,j) = o_im(i,j) + oo_im(i,j);
          }
        }

      }  // loop k

      {
        cout << "\n\n\t checking it my way " << __FUNCTION__ << "\n";

        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> p_lhs("P_lhs");
        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> p_rhs("P_rhs");

        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> p_lhs_times_p_rhs("P_lhs x H_DFT(k) x P_rhs");

        p_lhs.resize(std::pair<int,int>(o_dmft::dmn_size(), b_dmft::dmn_size()));
        p_rhs.resize(std::pair<int,int>(b_dmft::dmn_size(), o_dmft::dmn_size()));

        p_lhs_times_p_rhs.resize(std::pair<int,int>(o_dmft::dmn_size(), o_dmft::dmn_size()));

        for(int ik=0; ik<parameters.get_nKpoints(); ik++ )
          {
            for(int ib=0; ib<b_dmft::dmn_size(); ib++ )         // p1 and p2
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                p_lhs(io,ib) = projector(ik,ib,io);// std::complex<double>(projection_re(ik,ib,io), projection_im(ik,ib,io));

            for(int io=0; io<o_dmft::dmn_size(); io++ )
              for(int ib=0; ib<b_dmft::dmn_size(); ib++ )         // p1 and p2
                p_rhs(ib,io) = std::conj(p_lhs(io,ib));

            LIN_ALG::GEMM<LIN_ALG::CPU>::execute(p_lhs, p_rhs, p_lhs_times_p_rhs);

            for(int ib=0; ib<b_dmft::dmn_size(); ib++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                if(abs(p_lhs_times_p_rhs(io,io))>1.e-6)
                  projector(ik,ib,io) /= sqrt(abs(p_lhs_times_p_rhs(io,io)));

            //cout << "\n\t checking projection_re\n\n";
            for(int ib=0; ib<b_dmft::dmn_size(); ib++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                if(abs(real(projector(ik,ib,io))-projection_re(ik,ib,io))>1.e-6)
                  throw std::logic_error(__FUNCTION__);

            //cout << "\n\t checking projection_im\n\n";
            for(int ib=0; ib<b_dmft::dmn_size(); ib++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                if(abs(imag(projector(ik,ib,io))-projection_im(ik,ib,io))>1.e-6)
                  throw std::logic_error(__FUNCTION__);
          }
      }
    }

    template<class parameters_type>
    void data<parameters_type>::downfold_bloch_hamiltonians()
    {
      cout << "\t" << __FUNCTION__ << endl;

      double xx, yy;

      for(int ik=0; ik<parameters.get_nKpoints(); ik++ ) {

        // p1 and p2
        for(int ib=0; ib<b_dmft::dmn_size(); ib++ ) {
          for(int io=0; io<o_dmft::dmn_size(); io++ ) {
            p1_re(io,ib) =  projection_re(ik,ib,io);
            p1_im(io,ib) =  projection_im(ik,ib,io);

            p2_re(ib,io) =  p1_re(io,ib);
            p2_im(ib,io) = -p1_im(io,ib);
          }
        }

        // p3 = BH(k) * p2
        for(int io=0; io<o_dmft::dmn_size(); io++ ) {
          for(int ib=0; ib<b_dmft::dmn_size(); ib++ ) {

            /*
              xx = 0.;
              yy = 0.;
              for(int l=0; l<b_dmft::dmn_size(); l++ ) {
              xx = xx + ( BlochHami(ik,ib,l)*p2_re(l,io) );
              yy = yy + ( BlochHami(ik,ib,l)*p2_im(l,io) );
              }
              p3_re(ib,io) = xx;
              p3_im(ib,io) = yy;
            */

            p3_re(ib,io) = BlochHami(ik,ib,ib)*p2_re(ib,io);
            p3_im(ib,io) = BlochHami(ik,ib,ib)*p2_im(ib,io);
          }
        }

        // dfBH(k) = p1 * p3
        for(int i=0; i<o_dmft::dmn_size(); i++ ) {
          for(int j=0; j<o_dmft::dmn_size(); j++ ) {
            xx = 0.;
            yy = 0.;
            for(int l=0; l<b_dmft::dmn_size(); l++ ) {
              xx = xx + ( p1_re(i,l)*p3_re(l,j) - p1_im(i,l)*p3_im(l,j) );
              yy = yy + ( p1_im(i,l)*p3_re(l,j) + p1_re(i,l)*p3_im(l,j) );
            }

            dfBlochHami_re(ik,i,j) = xx;
            dfBlochHami_im(ik,i,j) = yy;
          }
        }

      }

      {
        cout << "\n\n\t checking it my way " << __FUNCTION__ << "\n";

        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> p_lhs("P_lhs");
        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> p_rhs("P_rhs");

        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> H_k("H_k");

        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> p_lhs_x_H_k        ("P_lhs x H_DFT(k)");
        LIN_ALG::matrix<std::complex<double>, LIN_ALG::CPU> p_lhs_x_H_k_x_p_rhs("P_lhs x H_DFT(k) x P_rhs");

        p_lhs.resize(std::pair<int,int>(o_dmft::dmn_size(), b_dmft::dmn_size()));
        p_rhs.resize(std::pair<int,int>(b_dmft::dmn_size(), o_dmft::dmn_size()));

        H_k  .resize(std::pair<int,int>(b_dmft::dmn_size(), b_dmft::dmn_size()));

        p_lhs_x_H_k        .resize(std::pair<int,int>(o_dmft::dmn_size(), b_dmft::dmn_size()));
        p_lhs_x_H_k_x_p_rhs.resize(std::pair<int,int>(o_dmft::dmn_size(), o_dmft::dmn_size()));

        for(int ik=0; ik<parameters.get_nKpoints(); ik++ )
          {
            for(int ib=0; ib<b_dmft::dmn_size(); ib++ )         // p1 and p2
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                p_lhs(io,ib) = projector(ik,ib,io);// std::complex<double>(projection_re(ik,ib,io), projection_im(ik,ib,io));

            for(int io=0; io<o_dmft::dmn_size(); io++ )
              for(int ib=0; ib<b_dmft::dmn_size(); ib++ )         // p1 and p2
                p_rhs(ib,io) = std::conj(p_lhs(io,ib));

            // Bloch Hamiltonians are currently diagonal ...
            for(int jb=0; jb<b_dmft::dmn_size(); jb++ )
              for(int ib=0; ib<b_dmft::dmn_size(); ib++ )
                H_k(ib,jb) = BlochHami(ik,ib,jb);

            LIN_ALG::GEMM<LIN_ALG::CPU>::execute(p_lhs      , H_k  , p_lhs_x_H_k        );
            LIN_ALG::GEMM<LIN_ALG::CPU>::execute(p_lhs_x_H_k, p_rhs, p_lhs_x_H_k_x_p_rhs);

            for(int jo=0; jo<o_dmft::dmn_size(); jo++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                H_0(ik, io, jo) = p_lhs_x_H_k_x_p_rhs(io,jo);

            //cout << "\n\t checking projection_re\n\n";
            for(int jo=0; jo<o_dmft::dmn_size(); jo++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                if(abs(real(H_0(ik,io,jo))-dfBlochHami_re(ik,io,jo))>1.e-6)
                  throw std::logic_error(__FUNCTION__);

            //cout << "\n\t checking projection_im\n\n";
            for(int jo=0; jo<o_dmft::dmn_size(); jo++ )
              for(int io=0; io<o_dmft::dmn_size(); io++ )
                if(abs(imag(H_0(ik,io,jo))-dfBlochHami_im(ik,io,jo))>1.e-6)
                  throw std::logic_error(__FUNCTION__);
          }
      }
    }

    template<class parameters_type>
    void data<parameters_type>::construct_t_ij()
    {
      cout << "\t" << __FUNCTION__ << endl;

      //       std::vector<int>    vec_oi;
      //       std::vector<int>    vec_oj;
      //       std::vector<int>    vec_dRx;
      //       std::vector<int>    vec_dRy;
      //       std::vector<int>    vec_dRz;
      //       std::vector<double> vec_tijRe;
      //       std::vector<double> vec_tijIm;

      double KK=1.0/parameters.get_nKpoints();

      int dRx, dRy, dRz;

      std::complex<double> tij;

      // loop neighbor unit cells
      for (int a=-parameters.get_nr(); a<=parameters.get_nr(); a++) {
        for (int b=-parameters.get_nr(); b<=parameters.get_nr(); b++) {
          for (int c=-parameters.get_nr(); c<=parameters.get_nr(); c++) {

            dRx = a;
            dRy = b;
            dRz = c;

            // loop oi, oj, - orbital pairs between the 2 sites
            for (int oi=0; oi<o_dmft::dmn_size(); oi++) {
              for (int oj=0; oj<o_dmft::dmn_size(); oj++) {

                tij = std::complex<double>(0.0,0.0);
                for (int k=0; k<parameters.get_nKpoints(); k++) {
                  double kdotdR = 2.0 * M_PI * (kx(k)*dRx + ky(k)*dRy + kz(k)*dRz);      // k.dot.dR
                  tij = tij + ( std::complex<double>( cos(kdotdR), sin(kdotdR) ) *
                                std::complex<double>( dfBlochHami_re(k,oi,oj), dfBlochHami_im(k,oi,oj)) );
                }
                tij = tij * KK;

                t_ij_element t_ij;

                t_ij.oi = (oi);
                t_ij.oj = (oj);

                t_ij.dRx = (dRx);
                t_ij.dRy = (dRy);
                t_ij.dRz = (dRz);

                t_ij.value = tij;

                t_ij_vector.push_back(t_ij);

                // for SrVO3, one can compare some value with paper
                // if ( a==1 && b==0 && c==1 && oi==1 && oj==1 )
                //   cout << "\n ------- \n" << tij << "\n ------- \n" << endl;


              } // oj
            } // oi



          } // c
        } // b
      } // a


      std::sort(t_ij_vector.begin(), t_ij_vector.end());

      for(int l=0; l<10; l++)
        t_ij_vector[l].print();

    }

  }

}

#endif
