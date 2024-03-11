#include <iostream>
#include <hip/hip_runtime.h>

__global__ void hello(){
	printf("Hello from gpu\n");
}

int main(){
    
	std::cout << "here" << std::endl;
	hipLaunchKernelGGL(hello, dim3(1), dim3(1), 0, 0);
	hipError_t syncStatus = hipDeviceSynchronize();
	if (syncStatus != hipSuccess) {
		std::cerr << "Device synchronization failed: " << hipGetErrorString(syncStatus) << std::endl;
		return -1;
	}

	return 0;
}

