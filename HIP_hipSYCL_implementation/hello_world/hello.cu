#include <iostream>
#include <cuda.h>

__global__ void hello(){
	printf("Hello from gpu\n");
}

int main(){


	hello<<<1,1>>>();
	cudaDeviceSynchronize();

	return 0;
}
