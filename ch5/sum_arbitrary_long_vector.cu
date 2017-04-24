#include <stdio.h>
#include "../common/book.h"

#define N (33 * 1024)

/*
 *	As we learned before, 65535 is the max number of blocks that we
 *	can launch in CUDA. This raises questions if we are going to sum
 *	vector whose length is larger than 65556 * number of threads.
 *	This can be solved quite simply by modifying the kernel code.
 *	We can iterare the data index by increment of blockDim.x * 
 *	gridDim.x. This is the number of threads per block multiplied by
 *	the number of blocks in the grid.
 */
__global__ void add(int* a, int* b, int* c) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += (blockDim.x * gridDim.x);
	}
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
	 *	Here, we launch 128 blocks and 128 threads in each block.
	 */
	add<<<128, 128>>>(dev_a, dev_b, dev_c);

	HANDLE_ERROR(cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	bool success = true;
	for(int i = 0; i < N; ++i) {
		if(a[i] + b[i] != c[i]) {
			success = false;
			printf("Error %d + %d != %d\n", a[i], b[i], c[i]);
		}
	}
	if(success) {
		printf("We did it!\n");
	}

	cudaFree(dev_a);
 	cudaFree(dev_b);
 	cudaFree(dev_c);

	return 0;
}
