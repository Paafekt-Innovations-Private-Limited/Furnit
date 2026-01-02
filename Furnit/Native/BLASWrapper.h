#pragma once
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

void BlasScopy(int n, const float* x, int incx, float* y, int incy);
void BlasSgemv(bool rowMajor, bool transpose, int m, int n, float alpha, const float* A, int lda, const float* x, int incx, float beta, float* y, int incy);
void BlasSgemm(bool rowMajor, bool transA, bool transB, int m, int n, int k, float alpha, const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc);

#ifdef __cplusplus
}
#endif
