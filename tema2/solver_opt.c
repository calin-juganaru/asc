#include "utils.h"
#include "matrix.h"

#define var register auto

// ============================================================================

matrix matrix::transpose()
{
	var At = matrix(size);

	for (var i = 0; i < size; ++i)
	{
		var At_i = At.data + i * size;
		var this_j = data + i;

		for (var j = 0; j <= i; ++j)
		{
			*At_i = *this_j;
			 ++At_i; this_j += size;
		}
	}

	return At;
}

// ============================================================================

matrix matrix::square()
{
	var A2 = matrix(size);
	var A2_i = A2.data;
	var this_i = data;

	for (var i = 0; i < size; ++i)
	{
		var this_i_k = this_i + i;
		var this_k = this_i;

		for (var k = i; k < size; ++k)
		{
			for (var j = k; j < size; ++j)
				A2_i[j] += *this_i_k * this_k[j];

			this_k += size; this_i_k++;
		}

		A2_i += size; this_i += size;
	}

	return A2;
}

// ============================================================================

matrix matrix::multiply1(const matrix& that)
{
	var C = matrix(size);

	for (var i = 0; i < size; ++i)
	{
		var this_i_k = data + i * size;
		var C_i = C.data + i * size;

		for (var k = 0; k < size; ++k, ++this_i_k)
		{
			var that_k_j = that.data + k * size;

			for (var j = 0; j <= k; ++j, ++that_k_j)
				C_i[j] += *this_i_k * *that_k_j;
		}
	}

	return C;
}

// ============================================================================

matrix matrix::multiply2(const matrix& that)
{
	var C = matrix(size);

	for (var i = 0; i < size; ++i)
	{
		var this_i_k = data + i * size + i;
		var C_i = C.data + i * size;

		for (var k = i; k < size; ++k, ++this_i_k)
		{
			var that_k_j = that.data + k * size;

			for (var j = 0; j < size; ++j, ++that_k_j)
				C_i[j] += *this_i_k * *that_k_j;
		}
	}

	return C;
}

// ============================================================================

matrix operator+(const matrix& A, const matrix& B)
{
	var size = A.size;
	var C = matrix(size);
	size *= size;

	var a = A.data;
	var b = B.data;
	var c = C.data;

	for (var i = 0; i < size; ++i)
	{
		*c = *a + *b;
		++a; ++b; ++c;
	}

	return C;
}

// ============================================================================

double* my_solver(int N, double* A, double* B)
{
	printf("OPT SOLVER\n");
	var X = matrix(A, N), Y = matrix(B, N);
	return (Y.multiply1(X.transpose())
		  + X.square().multiply2(Y)).get();
}

// ============================================================================