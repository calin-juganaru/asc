#include "utils.h"
#include "matrix.h"

// ============================================================================

matrix matrix::transpose()
{
	auto At = matrix(size);

	for (auto i = 0; i < size; ++i)
		for (auto j = 0; j <= i; ++j)
			At[i * size + j] = data[j * size + i];

	return At;
}

// ============================================================================

matrix matrix::square()
{
	auto A2 = matrix(size);

	for (auto i = 0; i < size; ++i)
		for (auto j = i; j < size; ++j)
		{
			auto aux = 0.0;

			for (auto k = i; k <= j; ++k)
				aux += data[i * size + k] * data[k * size + j];

			A2[i * size + j] = aux;
		}

	return A2;
}

// ============================================================================

matrix matrix::multiply1(const matrix& other)
{
	auto C = matrix(size);

	for (auto i = 0; i < size; ++i)
		for (auto j = 0; j < size; ++j)
		{
			auto aux = 0.0;

			for (auto k = j; k < size; ++k)
				aux += data[i * size + k] * other[k * size + j];

			C[i * size + j] = aux;
		}

	return C;
}

// ============================================================================

matrix matrix::multiply2(const matrix& other)
{
	auto C = matrix(size);

	for (auto i = 0; i < size; ++i)
		for (auto j = 0; j < size; ++j)
		{
			auto aux = 0.0;

			for (auto k = i; k < size; ++k)
				aux += data[i * size + k] * other[k * size + j];

			C[i * size + j] = aux;
		}

	return C;
}

// ============================================================================

matrix operator+(const matrix& A, const matrix& B)
{
	auto C = matrix(A.size);

	for (auto i = 0; i < C.size * C.size; ++i)
		C[i] = A[i] + B[i];

	return C;
}

// ============================================================================

double* my_solver(int N, double* A, double* B)
{
	printf("NEOPT SOLVER\n");
	auto X = matrix(A, N), Y = matrix(B, N);
	return (Y.multiply1(X.transpose())
		  + X.square().multiply2(Y)).get();
}

// ============================================================================