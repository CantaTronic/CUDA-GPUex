
#include <cuda_runtime.h> 
#include <iostream>

__global__ void test(char * name) {
//   printf ("Hello, %s", name);
  std::cout<<std::endl;
}

int main(){
  test<<<1,3>>>("Test");
  cudaDeviceSynchronize();
  return 0;
}

