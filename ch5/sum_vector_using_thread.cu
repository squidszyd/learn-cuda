#include <stdio.h>
#include "../common/book.h"

#define N 10

__global__ void add(int* a, int* b, int* c) {
	/*
	 *	threadIdx is a built-in variable representing the index
	 *	of thread. In CUDA, thread is also two dimensional, so
	 *	we can fetch the index using .x
	 */
	int tid = threadIdx.x;
	if(tid < N) c[tid] = a[tid] + b[tid];
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
	 *	Up to now, we have never set the second number in <<< ?, ? >>>
	 *	It is the first time we use this number here. The second value
	 *	indicates how many thread will we launch in a block. Relationship
	 *	between block and thread can be understood using the following
	 *	char-graph:
	 *	
	 *		BLOCK 0	| Thread 0 | Thread 1 | Thread 2 |
	 *		BLOCK 1	| Thread 0 | Thread 1 | Thread 2 |
	 *		BLOCK 2	| Thread 0 | Thread 1 | Thread 2 |
	 *		BLOCK 3	| Thread 0 | Thread 1 | Thread 2 |
	 *	
	 *	which means, each block has a number of threads. We can thus 
	 *	index data using:
	 *		int tid = threadIdx.x + blockIdx.x * blockDim.x
	 *	where blockDim is another built-in variable which is constant for
	 *	all blocks, indicating the number of threads along each dimension
	 *	of a block.
	 */
	add<<<1, N>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	for(int i = 0; i < N; ++i) {
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}

	cudaFree(dev_a);
 	cudaFree(dev_b);
 	cudaFree(dev_c);

	return 0;
}
