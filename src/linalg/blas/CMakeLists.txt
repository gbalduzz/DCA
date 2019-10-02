# BLAS

if(DCA_HAVE_CUDA)
  CUDA_ADD_LIBRARY(blas_kernels kernels_gpu.cu tensorcore_gemm.cu)
  target_compile_definitions(blas_kernels PRIVATE DCA_HAVE_CUDA)
  target_link_libraries(blas_kernels ${CUDA_cublas_LIBRARY})
endif()
