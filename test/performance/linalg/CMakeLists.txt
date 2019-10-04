
if (DCA_HAVE_CUDA AND DCA_WITH_TENSOR_CORES)
    add_executable(tensorcore_performance_test tensorcore_performance_test.cpp)
    target_compile_definitions(tensorcore_performance_test PRIVATE DCA_HAVE_CUDA)
    target_link_libraries(tensorcore_performance_test PRIVATE blas_kernels cuda_utils)
    target_include_directories(tensorcore_performance_test  PRIVATE
            ${DCA_INCLUDE_DIRS};${PROJECT_SOURCE_DIR})
endif ()
