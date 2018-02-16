#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>


#define CHECK(call)\
{ \
const cudaError_t error = call;\
if (error != cudaSuccess)\
{\
printf("Error: %s:%d, ", __FILE__, __LINE__); \
printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
exit(1); \
}\
}

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void initialInt(int *ip, int size) {
  for (int i=0; i<size; i++) {
    ip[i] = i;
  }
}

void checkResult(float *hostRef, float *gpuRef, const int N) {
  double epsilon = 1.0E-8;
  bool match = 1;
  for (int i=0; i<N; i++) {
    if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
      match = 0;
      printf("Arrays do not match!\n");
      printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
      break;
    }
  }
  if (match) printf("Arrays match.\n\n");
}

void initialData(float *ip,int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned) time(&t));
  for (int i=0; i<size; i++) {
    ip[i] = (float)( rand() & 0xFF )/10.0f;
  }
}

void printMatrix(int *C, const int nx, const int ny) {
  int *ic = C;
  printf("\nMatrix: (%d.%d)\n", nx, ny);
  for (int iy=0; iy<ny; iy++) {
    for (int ix=0; ix<nx; ix++) {
      printf("%3d",ic[ix]);
    }
    ic += nx;
    printf("\n");
  }
  printf("\n");
}

void sumMatrixOnHost (float *A, float *B, float *C, const int nx, const int ny) {
  float *ia = A;
  float *ib = B;
  float *ic = C;
  for (int iy=0; iy<ny; iy++) {
    for (int ix=0; ix<nx; ix++) {
      ic[ix] = ia[ix] + ib[ix];
    }
    ia += nx; ib += nx; ic += nx;
  }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny) {
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nx + ix;
  if (ix < nx && iy < ny)
    MatC[idx] = MatA[idx] + MatB[idx];
}


__global__ void printThreadIndex(int *A, const int nx, const int ny) {
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nx + ix;
  printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
  "global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
  blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv) {
  printf("%s Starting...\n", argv[0]);
  // set up device
  int dev = 0;
  cudaDeviceProp deviceProp;
  CHECK(cudaGetDeviceProperties(&deviceProp, dev));
  printf("Using Device %d: %s\n", dev, deviceProp.name);
  CHECK(cudaSetDevice(dev));
  // set up date size of vectors
  int nElem = 1<<24;
  printf("Vector size %d\n", nElem);
  // malloc host memory
  size_t nBytes = nElem * sizeof(float);
  float *h_A, *h_B, *hostRef, *gpuRef;
  h_A
  = (float *)malloc(nBytes);
  h_B
  = (float *)malloc(nBytes);
  hostRef = (float *)malloc(nBytes);
  gpuRef = (float *)malloc(nBytes);
  double iStart,iElaps;
  // initialize data at host side
  iStart = cpuSecond();
  initialData (h_A, nElem);
  initialData (h_B, nElem);
  iElaps = cpuSecond() - iStart;
  memset(hostRef, 0, nBytes);
  memset(gpuRef, 0, nBytes);
  // add vector at host side for result checks
  iStart = cpuSecond();
  sumArraysOnHost (h_A, h_B, hostRef, nElem);
  iElaps = cpuSecond() - iStart;
  // malloc device global memory
  float *d_A, *d_B, *d_C;
  cudaMalloc((float**)&d_A, nBytes);
  cudaMalloc((float**)&d_B, nBytes);
  cudaMalloc((float**)&d_C, nBytes);
  // transfer data from host to device
  cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
  // invoke kernel at host side
  int iLen = 1024;
  dim3 block (iLen);
  dim3 grid ((nElem+block.x-1)/block.x);
  iStart = cpuSecond();
  sumArraysOnGPU <<<grid, block>>>(d_A, d_B, d_C,nElem);
  cudaDeviceSynchronize();
  iElaps = cpuSecond() - iStart;
  printf("sumArraysOnGPU <<<%d,%d>>> Time elapsed %f" \
  "sec\n", grid.x, block.x, iElaps);
  // copy kernel result back to host side
  cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
  // check device results
  checkResult(hostRef, gpuRef, nElem);
  // free device global memory
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  // free host memory
  free(h_A);
  free(h_B);
  free(hostRef);
  free(gpuRef);

  return 0;
}

