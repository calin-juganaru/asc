#pragma once

#include <iostream>
#include <algorithm>

using namespace std;

// ============================================================================

struct matrix
{
	double* data;
	int size;

	matrix(int N)
	{
		size = N;
		data = new double[size * size];
		fill(data, data + size * size, 0);
	}

	matrix(double* M, int N)
	{
		size = N;
		data = new double[size * size];
		move(M, M + size * size, data);
	}

	matrix(const matrix& other)
	{
		size = other.size;
		data = new double[size * size];
		copy(other.data, other.data + size * size, data);
	}

	~matrix() { if (data != nullptr) delete[] data; }

	double& operator[](size_t index)
	{ return data[index]; }
	const double& operator[](size_t index)
	const { return data[index]; }

    double* get()
    {
        auto aux = data;
        data = nullptr;
        return aux;
    }

	matrix transpose();
	matrix square();
	matrix multiply1(const matrix&);
    matrix multiply2(const matrix&);
};

// ============================================================================