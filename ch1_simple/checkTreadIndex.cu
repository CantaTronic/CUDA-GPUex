
#include <stdio.h>
#include "cuda_runtime.h"

#define CHECK(call) {\
const cudaError_t err = call;\
if (err != cudaSuccess) {\
printf("Error №%d in %s:%d\n", err, __FILE__, __LINE__);\
printf("Reason: %s \n", err, cudaGetStringError(err));\
exit(1);\
}\
}

__global__ void printThreadIndex(int *matr, int nx, int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  
  unsigned idx = iy * nx + ix;
  
  printf("ThreadIdx (%d, %d)\nBlockIdx (%d, %d)\nCoordinate (%d, %d)\nGlobal index %d, value is %2d\n====================================================\n", threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, matr[idx]);
}

void initialInt(int * ip, unsigned sz){
  for (unsigned i = 0; i < sz; i++)
    ip[i] = i;
}

void printMatrix(int *matr, unsigned nx, unsigned ny){
//печатаем матрицу matr, развернутую в одномерный вектор
  int *my_matr = matr;
  printf("Matrix %d x %d\n", nx, ny);
  for (unsigned i = 0; i < nx; i++) {
    for (unsigned j = 0; j < ny; j++){
      int ind = j * nx + i;
      printf("%3d", matr[ind]);
    }
    printf("\n");
  }
  printf("\n");
}

void SetDevice(int dev = 0) {
  cudaDeviceProp devProp;
//   int dev = 0;
  cudaGetDeviceProperties(&devProp, dev);
  printf ("Using device %d: %s\n", dev, devProp.name);
  cudaSetDevice(dev);
}

int main() {
  SetDevice();
  
  //setup matrix dimentions
  int nx = 6; //number of digits in column 
  int ny = 8; //number of digits in row
  unsigned nxy = nx * ny; //whole matrix size
  int nBytes = nxy * sizeof(float);
  
  //malloc host memory
  int * h_A = (int *) malloc(nBytes);
  //initialize host matrix
  initialInt(h_A, nxy);
  //check out initialization
  printMatrix(h_A, nx, ny);

  //malloc device memory
  int * dev_A;
  cudaMalloc((void **) &dev_A, nBytes);
  
  //transfer data from host to device
  //куда, что, сколько, направление
  cudaMemcpy(dev_A, h_A, nBytes, cudaMemcpyHostToDevice);
  
  //setup block execution generation
  dim3 block (4, 2);
  dim3 grid ((nx + block.x -1)/block.x, (ny + block.y -1)/block.y);
  
  printThreadIndex <<<block, grid>>> (dev_A, nx, ny);
  
  cudaDeviceSynchronize();
  
  //free host and device memory
  cudaFree(dev_A);
  free(h_A);
  
  cudaDeviceReset();
  printf("Success!\n");
  return 0;
}
