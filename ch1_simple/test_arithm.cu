
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void test_add(int a, int b) { // added parameters int a, int b
   a += b;
}

int main() {
//   cout<<(test_add<<<1,1>>>(4,5))<<endl;
  test_add<<<1,1>>>(4,5);
  cudaDeviceSynchronize(); // was CudaDeviceSinchronize
  cudaDeviceReset(); // was CudaDeviceReset
  return 0;
}
