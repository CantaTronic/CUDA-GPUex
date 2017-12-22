
#include <cuda_runtime.h>
#include <iostream>

using namespace std;

__global__ void test_add() {
//   a += b;
}

int main() {
//   cout<<(test_add<<<1,1>>>(4,5))<<endl;
  test_add<<<1,1>>>(4,5);
  CudaDeviceSinchronize();
  CudaDeviceReset();
  return 0;
} 
