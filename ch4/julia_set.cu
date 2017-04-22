#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 100
#define SCALE 1.5
#define CONST_CX -0.8
#define CONST_CY 0.156
#define ITER_NUM 200
#define DIVERGENCE_TH 1000

/*
 *	Our complext number structure
 */
struct cuComplex {
	float r;
	float i;
	__device__ cuComplex(float real, float imag):r(real), i(imag){}
	__device__ float magnitude2(void){
		return r * r + i * i;
	}
	__device__ cuComplex operator*(const cuComplex& rhs){
		return cuComplex(r * rhs.r - i * rhs.i, i * rhs.r + r * rhs.i);
	}
	__device__ cuComplex operator+(const cuComplex& rhs){
		return cuComplex(r + rhs.r, i + rhs.i);
	}
};

__device__ int julia(int x, int y){
	// the complex plane is centered at (DIM/2, DIM/2)
	float jx = SCALE * (float)(DIM/2 - x)/(DIM/2);
	float jy = SCALE * (float)(DIM/2 - y)/(DIM/2);

	cuComplex C(CONST_CX, CONST_CY);
	cuComplex a(jx, jy);

	int i = 0;
	for(i = 0; i < ITER_NUM; ++i){
		a = a * a + C;	
		if(a.magnitude2() > DIVERGENCE_TH)
			return 0;
	}
	return 1;
}

__global__ void julia_kernel(unsigned char* ptr){
	int x = blockIdx.x;
	int y = blockIdx.y;
	/*
	 *	gridDim is another built-in variable, representing the dimension of
	 *	computation blocks
	 */
	int offset = x + y * gridDim.x;

	int julia_val = julia(x, y);
	ptr[offset*4 + 0] = 255 * julia_val;
	ptr[offset*4 + 1] = 0;
	ptr[offset*4 + 2] = 0;
	ptr[offset*4 + 3] = 255;
}

int main(void) {
	CPUBitmap bitmap(DIM, DIM);
	unsigned char* dev_bitmap;
	
	HANDLE_ERROR(cudaMalloc((void**)&dev_bitmap, bitmap.image_size()));
	
	/*
	 *	dim3 is a built-in type, representing a three dimension tuple.
	 *	But why is grid(DIM, DIM) three dimensional? Isn't it two
	 *	dimensional? Actually, three dimensional is what the cuda
	 *	runtime expects, though it is not supported by now. The third
	 *	dimension will be initialized as 1 by default.
	 */
	dim3 grid(DIM, DIM);
	julia_kernel<<<grid, 1>>>(dev_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap, 
				bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();

	HANDLE_ERROR(cudaFree(dev_bitmap));

	return 0;
}
