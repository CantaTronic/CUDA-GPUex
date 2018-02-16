
#include <stdio.h>
#include "cuda_runtime.h"

int gridSize(int nElem, int blockSz) {
  return (nElem + blockSz - 1)/blockSz;
}

void gridResize(dim3 block, dim3 grid, int nElem, int blockSz){
  block.x = blockSz;
  grid.x = gridSize(nElem, block.x);
  printf("Blocks: %d, grids: %d\n", block.x, grid.x);
}

int main() {
  //define total data elements
  int nElem = 1024;
  
  //inite block and grid sizes
  dim3 block(1024);
  dim3 grid (gridSize(nElem, block.x));
  printf("Initial block size: %d, grid size: %d\n", block.x, grid.x);
  
  //variate block sizes
  for (int i = nElem; i>1; i/=2)
    gridResize(block, grid, nElem, i);
  
  cudaDeviceSynchronize();
  cudaDeviceReset();
  
  return 0;
}
