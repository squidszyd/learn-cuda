#include <stdio.h>
#include "../common/book.h"

__global__ void add(int a, int b, int* c) {
	*c = a + b;
}

int main(void) {
	int c;
	int* dev_c;
	/*
	 * ** DO NOT ** dereference the pointer returned by cudaMalloc()
	 * from code that executes on the host !!
	 * RESTRICTIONS:
	 * <1>. Pass pointers allocated with cudaMalloc() to functions that
	 *		execute on the device is allowed.
	 * <2>. It is allowed to read or write the pointers allocated with
	 *		cudaMalloc() as long as they are run on the device.
	 * <3>. Pointers allocated with cudaMalloc() can be passed to
	 *		functions execute on the host.
	 * <4>.	As the ** DO NOT ** says at line 12
	 */
	HANDLE_ERROR(cudaMalloc((void**)&dev_c, sizeof(int)));

	add<<<1, 1>>>(2, 7, dev_c);

	/*
	 *	cudaMemcpyDeviceToHost		|
	 *	cudaMemcpyHostToDevice		┠ As their names tell
	 *	cudaMemcpyDeviceToDevice	|
	 */
	HANDLE_ERROR(cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost));

	printf("2 + 7 = %d\n", c);

	/*
	 * We must use cudaFree to free the memory allocated by cudaMalloc,
	 * but not C free.
	 */
	cudaFree(dev_c);

	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));
	printf("Device count: %d\n", count);

	/*
	 *	cudaDeviceProp is a structure contains information abount our device
	 *	char name[256], size_t totalGlobalMem and many more. [Page 28]
	 */
	cudaDeviceProp prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	printf("Device name: %s\n", prop.name);

	return 0;
}
