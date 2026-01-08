// Modern Accelerate BLAS API (ILP64)
// Compiler flags -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 are set in build settings
#include <Accelerate/Accelerate.h>
#include "BLASWrapper.h"

void BlasScopy(int n, const float* x, int incx, float* y, int incy) {
    cblas_scopy((__LAPACK_int)n, x, (__LAPACK_int)incx, y, (__LAPACK_int)incy);
}

void BlasSgemv(bool rowMajor, bool transpose, int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy) {
    cblas_sgemv(rowMajor ? CblasRowMajor : CblasColMajor,
                transpose ? CblasTrans : CblasNoTrans,
                (__LAPACK_int)m, (__LAPACK_int)n, alpha, A, (__LAPACK_int)lda, x, (__LAPACK_int)incx, beta, y, (__LAPACK_int)incy);
}

void BlasSgemm(bool rowMajor, bool transA, bool transB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
    cblas_sgemm(rowMajor ? CblasRowMajor : CblasColMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                (__LAPACK_int)m, (__LAPACK_int)n, (__LAPACK_int)k, alpha, A, (__LAPACK_int)lda, B, (__LAPACK_int)ldb, beta, C, (__LAPACK_int)ldc);
}
