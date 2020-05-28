#include <stdio.h>
#include "utils/utils.h"

#define NMAX (1 << 20)

// ============================================================================

__global__ void kernel_parity_id(int *a, int N)
{
	unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < N) a[i] %= 2;
}

// ============================================================================

__global__ void kernel_block_id(int* a, int N)
{
	unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < N) a[i] = blockIdx.x;
}

// ============================================================================

__global__ void kernel_thread_id(int* a, int N)
{
	unsigned i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < N) a[i] = threadIdx.x;
}

// ============================================================================

int main(void)
{
    int nDevices, *host_a, *device_a;
    cudaGetDeviceCount(&nDevices);

    // ------------------------------------------------------------------------

    for (int i = 0; i < nDevices; ++i)
    {
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, i);

		printf("Device number: %d\n", i);
		printf("    Device name: %s\n", properties.name);
		printf("    Total memory: %zu\n", properties.totalGlobalMem);
		printf("    Memory Clock Rate (KHz): %d\n", properties.memoryClockRate);
		printf("    Memory Bus Width (bits): %d\n", properties.memoryBusWidth);
    }

    // ------------------------------------------------------------------------

    host_a = (int*)malloc(NMAX * sizeof(int));
	cudaMalloc(&device_a, NMAX * sizeof(int));

    fill_array_int(host_a, NMAX);
	cudaMemcpy(device_a, host_a, NMAX * sizeof(int),
		                cudaMemcpyHostToDevice);

    // ------------------------------------------------------------------------

    kernel_parity_id<<<NMAX / 4, 4>>>(device_a, NMAX);

	cudaDeviceSynchronize();

	cudaMemcpy(host_a, device_a, NMAX * sizeof(int),
		        cudaMemcpyDeviceToHost);

	check_task_1(3, host_a);

    // ------------------------------------------------------------------------

	kernel_block_id<<<NMAX / 4, 4>>>(device_a, NMAX);

	cudaDeviceSynchronize();

	cudaMemcpy(host_a, device_a, NMAX * sizeof(int),
		        cudaMemcpyDeviceToHost);

	check_task_1(4, host_a);

    // ------------------------------------------------------------------------

	kernel_thread_id<<<NMAX / 4, 4>>>(device_a, NMAX);

	cudaDeviceSynchronize();

	cudaMemcpy(host_a, device_a, NMAX * sizeof(int),
		        cudaMemcpyDeviceToHost);

	check_task_1(5, host_a);

    // ------------------------------------------------------------------------

	free(host_a);
	cudaFree(device_a);

    return 0;
}

// ============================================================================