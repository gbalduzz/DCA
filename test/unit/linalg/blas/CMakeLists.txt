# linalg/blas
dca_add_gtest(tensorcore_gemm_test
        GTEST_MAIN
        CUDA
        INCLUDE_DIRS ${DCA_INCLUDE_DIRS}
        LIBS ${DCA_LIBS})
