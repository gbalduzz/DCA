//-*-C++-*-

#ifndef COPY_FROM_GPU_CPU_H
#define COPY_FROM_GPU_CPU_H

namespace LIN_ALG {

  namespace COPY_FROM_GPU_to_CPU {

    template<typename scalartype>
    void memcopy_d_to_h(scalartype* target_ptr, scalartype* source_ptr, int size);
    
    template<typename scalartype>
    void memcopy_d_to_h_async(scalartype* target_ptr, scalartype* source_ptr, int size, 
			      int thread_id, int stream_id);

    template<typename scalartype>
    void memcopy_2D_d_to_h(scalartype* source_ptr, std::pair<int, int>& source_c_s, std::pair<int, int>& source_g_s, 
			   scalartype* target_ptr, std::pair<int, int>& target_c_s, std::pair<int, int>& target_g_s);

    template<typename scalartype>
    void memcopy_2D_d_to_h_async(scalartype* source_ptr, std::pair<int, int>& source_c_s, std::pair<int, int>& source_g_s, 
				 scalartype* target_ptr, std::pair<int, int>& target_c_s, std::pair<int, int>& target_g_s,
				 int thread_id, int stream_id);
  }

  /*!
   *  \brief structure to copy a matrix from the GPU to the CPU
   */
  template<>
  class COPY_FROM<GPU, CPU>
  {
  public:

    template<typename scalartype>
    static void execute(scalartype* ptr_gpu, scalartype* ptr_cpu, int size)
    {
      COPY_FROM_GPU_to_CPU::memcopy_d_to_h(ptr_cpu, ptr_gpu, size);
    }

    template<typename scalartype>
    static void execute(scalartype* ptr_gpu, scalartype* ptr_cpu, int size, int thread_id, int stream_id)
    {
      COPY_FROM_GPU_to_CPU::memcopy_d_to_h_async(ptr_cpu, ptr_gpu, size, thread_id, stream_id);
    }


    template<typename scalartype>
    static void execute(scalartype* source_ptr, std::pair<int, int>& source_c_s, std::pair<int, int>& source_g_s,
			scalartype* target_ptr, std::pair<int, int>& target_c_s, std::pair<int, int>& target_g_s)
    {
      assert(source_c_s==target_c_s);
      
      COPY_FROM_GPU_to_CPU::memcopy_2D_d_to_h(source_ptr, source_c_s, source_g_s, target_ptr, target_c_s, target_g_s);
    }

    template<typename scalartype>
    static void execute(scalartype* source_ptr, std::pair<int, int>& source_c_s, std::pair<int, int>& source_g_s,
			scalartype* target_ptr, std::pair<int, int>& target_c_s, std::pair<int, int>& target_g_s, 
			int thread_id, int stream_id)
    {
      assert(source_c_s==target_c_s);
    
      COPY_FROM_GPU_to_CPU::memcopy_2D_d_to_h_async(source_ptr, source_c_s, source_g_s, target_ptr, target_c_s, target_g_s, thread_id, stream_id);
    }



    template<typename gpu_matrix_type, typename cpu_matrix_type>
    static void execute(gpu_matrix_type& gpu_matrix, cpu_matrix_type& cpu_matrix)
    {
      if(gpu_matrix.get_global_size() == cpu_matrix.get_global_size())
	{
	  int size = gpu_matrix.get_global_size().first*gpu_matrix.get_global_size().second;
	  
	  execute(gpu_matrix.get_ptr(), cpu_matrix.get_ptr(), size);
	}
      else
	execute(gpu_matrix.get_ptr(), gpu_matrix.get_current_size(), gpu_matrix.get_global_size(),
		cpu_matrix.get_ptr(), cpu_matrix.get_current_size(), cpu_matrix.get_global_size());
    }

    template<typename gpu_matrix_type, typename cpu_matrix_type>
    static void execute(gpu_matrix_type& gpu_matrix, cpu_matrix_type& cpu_matrix, int thread_id, int stream_id)
    {
      if(gpu_matrix.get_global_size() == cpu_matrix.get_global_size())
	{
	  int size = gpu_matrix.get_global_size().first*gpu_matrix.get_global_size().second;
	  
	  execute(gpu_matrix.get_ptr(), cpu_matrix.get_ptr(), size, thread_id, stream_id);
	}
      else
	execute(gpu_matrix.get_ptr(), gpu_matrix.get_current_size(), gpu_matrix.get_global_size(),
		cpu_matrix.get_ptr(), cpu_matrix.get_current_size(), cpu_matrix.get_global_size(), 
		thread_id, stream_id);
    }

  };

}

#endif