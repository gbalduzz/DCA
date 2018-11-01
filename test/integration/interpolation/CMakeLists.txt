# Interpolation tests

dca_add_gtest(wannier_interpolation_test
  GTEST_MAIN
  INCLUDE_DIRS ${FFTW_INCLUDE_DIR};${SIMPLEX_GM_RULE_INCLUDE_DIR}
  LIBS coarsegraining;function;gaussian_quadrature;tetrahedron_mesh;
       ${LAPACK_LIBRARIES};${SIMPLEX_GM_RULE_LIBRARY})
