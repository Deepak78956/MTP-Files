#include <iostream>
#include <hip/hip_runtime.h>

__global__ void hello(){
	printf("Hello from gpu\n");
}

int main(){


	hello<<<1,1>>>();
	hipDeviceSynchronize();

	return 0;
}
