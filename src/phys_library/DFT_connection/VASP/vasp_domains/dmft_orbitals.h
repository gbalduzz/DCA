//-*-C++-*-

#ifndef DMFT_ORBITAL_DOMAIN_H
#define DMFT_ORBITAL_DOMAIN_H

namespace DFT
{
  namespace VASP
  {
    /*!
     *  \author Peter Staar
     */
    class dmft_orbital_domain
    {
    public:

      typedef int element_type;

    public:

      static int&                       get_size();
      static std::string                get_name();

      static std::vector<element_type>& get_elements();

      template<IO::FORMAT DATA_FORMAT>
      static void read(IO::reader<DATA_FORMAT>& reader);

      template<IO::FORMAT DATA_FORMAT>
      static void write(IO::writer<DATA_FORMAT>& writer);

      template<typename parameters_type>
      static void initialize(parameters_type& parameters);

      template<class stream_type>
      static void to_JSON(stream_type& ss);

    private:

      static std::vector<element_type>& initialize_elements();
    };

    int& dmft_orbital_domain::get_size()
    {
      static int size = 0;
      return size;
    }

    std::string dmft_orbital_domain::get_name()
    {
      static std::string name = "dmft-orbital-domain";
      return name;
    }

    std::vector<dmft_orbital_domain::element_type>& dmft_orbital_domain::get_elements()
    {
      static std::vector<element_type> elements(get_size());
      return elements;
    }

    template<IO::FORMAT DATA_FORMAT>
    void dmft_orbital_domain::write(IO::writer<DATA_FORMAT>& writer)
    {
      writer.open_group(get_name());
      writer.execute(get_elements());
      writer.close_group();
    }

    template<typename parameters_type>
    void dmft_orbital_domain::initialize(parameters_type& parameters)
    {
      get_size() = parameters.get_nb_dmft_orbitals();

      for(size_t i=0; i<get_elements().size(); ++i){
        get_elements()[i] = i;
      }
    }

  }

}

#endif