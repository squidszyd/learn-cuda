#include <stdio.h>
#include "../common/book.h"

#define N 10000

__global__ void add(int* a, int* b, int* c) {
	/*
	 *	blockIdx is the built-in variable which contains the value of
	 *	the block index for whichever block is currently running the
	 *	device code.
	 *	Another thing to be explained is the '.x' domain. In CUDA C,
	 *	blocks are actually defined in two-dimensions
	 */
	int tid = blockIdx.x;
	if(tid < N)	c[tid] = a[tid] + b[tid];
}

int main(void) {
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;

	HANDLE_ERROR(cudaMalloc((void**)&dev_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, N * sizeof(int)));

	for(int i = 0; i < N; ++i) {
		a[i] = -i;
		b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
 	HANDLE_ERROR(cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	/*
	 *	Note the <<<N, 1>>> here:
	 *	The first N represents the number of parallel blocks in which we
	 *	we would like the device to execute out kernel. The runtime will
	 *	create N copies of the kernel and running them in parallel.
	 */
	add<<<N, 1>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for(int i = 0; i < N; ++i) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
 	cudaFree(dev_b);
 	cudaFree(dev_c);

	return 0;
}
