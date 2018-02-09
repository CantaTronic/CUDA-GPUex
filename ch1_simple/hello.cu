
#include <cuda_runtime.h>
//#include <iostream>
#include <cstdio>

__global__ void test(/* const char * name -- CUDA cannot handle an array directly without cudaMemCpy */) {
  printf ("Hello\n");
//  printf ("Hello, %s\n", name);
//  std::cout<<std::endl; // CUDA cannot handle std::cout inside a kernel
}

int main(){
  test<<<1,3>>>(/* "Test" */);
  cudaDeviceSynchronize();
  return 0;
}

