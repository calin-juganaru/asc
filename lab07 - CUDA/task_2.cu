#include <stdio.h>
#include <math.h>
#include "utils/utils.h"

// ============================================================================

__global__ void add_arrays(const float* a, const float* b, float* c, int N)
{
	unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < N) c[i] = a[i] + b[i];
}

// ============================================================================

int main(void)
{
	cudaSetDevice(0);

	int N = 1 << 20;
	const size_t block_size = 256;
	size_t num_blocks = N / block_size;
    if (N % block_size)
		++num_blocks;

	float* host_array_a = 0;
	float* host_array_b = 0;
	float* host_array_c = 0;

	float* device_array_a = 0;
	float* device_array_b = 0;
	float* device_array_c = 0;

	// ------------------------------------------------------------------------

	host_array_a = (float*)malloc(N * sizeof(int));
	host_array_b = (float*)malloc(N * sizeof(int));
	host_array_c = (float*)malloc(N * sizeof(int));

	// ------------------------------------------------------------------------

	cudaMalloc(&device_array_a, N * sizeof(int));
    cudaMalloc(&device_array_b, N * sizeof(int));
	cudaMalloc(&device_array_c, N * sizeof(int));

	// ------------------------------------------------------------------------

	fill_array_float(host_array_a, N);
	fill_array_random(host_array_b, N);

	// ------------------------------------------------------------------------

	cudaMemcpy(device_array_a, host_array_a, N * sizeof(int),
                cudaMemcpyHostToDevice);

	cudaMemcpy(device_array_b, host_array_b, N * sizeof(int),
                cudaMemcpyHostToDevice);

	// ------------------------------------------------------------------------

	add_arrays<<<num_blocks, block_size>>>(device_array_a, device_array_b,
		                                    device_array_c, N);
	cudaDeviceSynchronize();

	// ------------------------------------------------------------------------

	cudaMemcpy(host_array_c, device_array_c, N * sizeof(int),
                cudaMemcpyDeviceToHost);

	check_task_2(host_array_a, host_array_b, host_array_c, N);

	// ------------------------------------------------------------------------

	free(host_array_a);
	free(host_array_b);
	free(host_array_c);

	cudaFree(device_array_a);
    cudaFree(device_array_b);
	cudaFree(device_array_c);

	return 0;
}

// ============================================================================