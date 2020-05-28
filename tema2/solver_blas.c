#include "utils.h"
#include "matrix.h"

extern "C"
{
   #include <cblas.h>
}

// ============================================================================

matrix matrix::square()
{
	auto A2 = matrix(*this);
	auto N = size;

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                 CblasNonUnit, N, N, 1.0, data, N, A2.data, N);

	return A2;
}

// ============================================================================

matrix matrix::multiply1(const matrix& other)
{
	auto C = matrix(*this);
	auto N = size;

	cblas_dtrmm(CblasRowMajor, CblasRight, CblasUpper, CblasTrans,
                 CblasNonUnit, N, N, 1.0, other.data, N, C.data, N);

	return C;
}

// ============================================================================

matrix matrix::multiply2(const matrix& other)
{
	auto C = matrix(other);
	auto N = size;

	cblas_dtrmm(CblasRowMajor, CblasLeft, CblasUpper, CblasNoTrans,
                 CblasNonUnit, N, N, 1.0, data, N, C.data, N);

	return C;
}

// ============================================================================

matrix operator+(const matrix& A, const matrix& B)
{
	auto C = matrix(A.size);
	auto N = C.size * C.size;

	cblas_daxpy(N, 1.0, A.data, 1, C.data, 1);
	cblas_daxpy(N, 1.0, B.data, 1, C.data, 1);

	return C;
}

// ============================================================================

double* my_solver(int N, double* A, double* B)
{
	printf("BLAS SOLVER\n");
	auto X = matrix(A, N), Y = matrix(B, N);
	return (Y.multiply1(X) + X.square().multiply2(Y)).get();
}

// ============================================================================