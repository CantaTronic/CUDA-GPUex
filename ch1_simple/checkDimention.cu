
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void checkIndex() {
  printf("threadIdx = ( %d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z);
  printf("blockDim = ( %d, %d, %d)\n", blockDim.x, blockDim.y, blockDim.z);
  printf("gridDim = ( %d, %d, %d)\n", gridDim.x, gridDim.y, gridDim.z);
}

int main() {
  int nElem = 6;
  
  dim3 block(3);
  dim3 grid((nElem - block.x - 1)/block.x);
  
  printf("block = ( %d, %d, %d)\n", block.x, block.y, block.z);
  printf("grid = ( %d, %d, %d)\n", grid.x, grid.y, grid.z);
  
  
  checkIndex <<<block, grid>>> ();
  // check
  printf("==== test ====\n");
  checkIndex <<<2, 3>>> ();
  cudaDeviceSynchronize();
  return 0;
}
