
// #include <iostream>
#include <cuda_runtime.h>
// #include <string>
#include <stdio.h>

using namespace std;

__global__ void test(/*string name*/) {
  printf("test_gpu\n");
}

int main(){
  test<<<1,1>>>(/*"Test"*/);
  cudaDeviceSynchronize();
}
