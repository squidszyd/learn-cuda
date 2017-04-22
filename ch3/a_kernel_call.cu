#include <stdio.h>

/*
 * '__global__' alerts the compiler that a function should be compiled
 * to run on a device instead of the host
 */
__global__ void kernel( void ) {
}

int main( void ) {
	/*
	 * <<<?, ?>>> will be run on device, and the '?' in these angle brackets
	 * are parameters that will influence how the runtime will launch our 
	 * device code.
	 */
	kernel<<<1, 1>>>();
	printf("Hello World!\n");
	return 0;
}
