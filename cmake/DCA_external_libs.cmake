################################################################################
# External libraries
#
# TODO: - Write FindNFFT.cmake.
#       - Write FindSPGLIB.cmake.
#       - Write FindFFTW.cmake.
#       - Use static libraries for NFFT and SPGLIB?
#       - Set DCA_XXX_AVAILABLE to true after XXX was found?
################################################################################

# NFFT
find_library(NFFT_LIBRARY
  NAMES libnfft3.a nfft3
  PATHS ${NFFT_DIR}/lib
  NO_DEFAULT_PATH)

# SPGLIB
find_library(SPGLIB_LIBRARY
  NAMES libsymspg.a symspg
  PATHS ${SPGLIB_DIR}/lib
  NO_DEFAULT_PATH)

# Lapack
if (NOT DCA_LAPACK_IMPLICIT)
  find_package(LAPACK REQUIRED)
endif()

# HDF5
if (NOT DCA_HDF5_IMPLICIT)
  find_package(HDF5 REQUIRED COMPONENTS CXX)
endif()
  
# FFTW
if (NOT DCA_FFTW_IMPLICIT)
  find_library(FFTW_LIBRARY NAMES fftw3)
  get_filename_component(FFTW_LIB_DIR ${FFTW_LIBRARY} DIRECTORY)
  get_filename_component(FFTW_DIR     ${FFTW_LIB_DIR} DIRECTORY)
  set(FFTW_INCLUDE_DIR "${FFTW_DIR}/include" CACHE FILEPATH "Path to fftw3.h.")
endif()


set(DCA_EXTERNAL_LIBS
  ${LAPACK_LIBRARIES}
  ${HDF5_LIBRARIES}
  ${HDF5_CXX_LIBRARIES}
  ${NFFT_LIBRARY}
  ${FFTW_LIBRARY}
  ${SPGLIB_LIBRARY})

set(DCA_EXTERNAL_INCLUDES
  ${NFFT_DIR}/include
  ${SPGLIB_DIR}/include
  ${FFTW_INCLUDE_DIR}
  ${HDF5_INCLUDE_DIRS})

mark_as_advanced(
  MPI_LIBRARY MPI_EXTRA_LIBRARY
  NFFT_LIBRARY
  SPGLIB_LIBRARY
  FFTW_INCLUDE_DIR FFTW_LIBRARY
  HDF5_DIR)