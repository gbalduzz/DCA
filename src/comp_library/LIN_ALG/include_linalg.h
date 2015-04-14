//-*-C++-*-

// BLAS
//#include "C_wrappers_BLAS.h" 

//LAPACK
//#include "C_wrappers_LAPACK.h" 

#include "linalg_device_types.h"

#include "matrix_scalartype.h"

#include "cublas_thread_manager_tem.h"
#include "cublas_thread_manager_CPU.h"
#include "cublas_thread_manager_GPU.h"

#include "copy_from_tem.h"
#include "copy_from_CPU_CPU.h"
#include "copy_from_CPU_GPU.h"
#include "copy_from_GPU_CPU.h"
#include "copy_from_GPU_GPU.h"

#include "memory_management_tem.h"
#include "memory_management_CPU.h"
#include "memory_management_GPU.h"

#include "vector.h"
#include "matrix.h"

#include "LU_MATRIX_OPERATIONS.h"
#include "LU_MATRIX_OPERATIONS_CPU.h"
#include "LU_MATRIX_OPERATIONS_GPU.h"

#include "REMOVE_tem.h"
#include "REMOVE_CPU.h"
#include "REMOVE_GPU.h"

// BLAS 1

#include "BLAS_1_AXPY_tem.h"
#include "BLAS_1_AXPY_CPU.h"
#include "BLAS_1_AXPY_GPU.h"

#include "BLAS_1_COPY_tem.h"
#include "BLAS_1_COPY_CPU.h"
#include "BLAS_1_COPY_GPU.h"

#include "BLAS_1_SCALE_tem.h"
#include "BLAS_1_SCALE_CPU.h"
#include "BLAS_1_SCALE_GPU.h"

#include "BLAS_1_SWAP_tem.h"
#include "BLAS_1_SWAP_CPU.h"
#include "BLAS_1_SWAP_GPU.h"

// BLAS 2

#include "BLAS_2_GEMV_tem.h"
#include "BLAS_2_GEMV_CPU.h"

// BLAS 3

#include "BLAS_3_TRSM_tem.h"
#include "BLAS_3_TRSM_CPU.h"
#include "BLAS_3_TRSM_GPU.h"

#include "BLAS_3_GEMM_tem.h"
#include "BLAS_3_GEMM_CPU.h"
#include "BLAS_3_GEMM_GPU.h"


#include "LASET_tem.h"
#include "LASET_CPU.h"
#include "LASET_GPU.h"

#include "DOT_tem.h"
#include "DOT_CPU.h"
#include "DOT_GPU.h"

#include "GEMD_tem.h"
#include "GEMD_CPU.h"
#include "GEMD_GPU.h"

#include "TRSV_tem.h"
#include "TRSV_CPU.h"
#include "TRSV_GPU.h"


#include "BENNET_tem.h"
#include "BENNET_CPU.h"
#include "BENNET_GPU.h"

#include "GETRS_tem.h"
#include "GETRS_CPU.h"
#include "GETRS_GPU.h"

#include "GETRF_tem.h"
#include "GETRF_CPU.h"
#include "GETRF_GPU.h"

#include "GETRI_tem.h"
#include "GETRI_CPU.h"
#include "GETRI_GPU.h"

#include "GEINV_tem.h"

#include "GEEV_tem.h"
#include "GEEV_CPU.h"
#include "GEEV_GPU.h"

#include "GESV_tem.h"
#include "GESV_CPU.h"

#include "GESVD_tem.h"
#include "GESVD_CPU.h"
#include "GESVD_GPU.h"

#include "PSEUDO_INVERSE_tem.h"
#include "PSEUDO_INVERSE_CPU.h"

// performance_inspector

#include "performance_inspector_tem.h"
#include "performance_inspector_CPU.h"
#include "performance_inspector_GPU.h"