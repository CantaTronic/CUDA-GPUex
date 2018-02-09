
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

__global__ void sumOnDevice(float *A, float *B, float *C, unsigned N) {
  unsigned i = threadIdx.x;
  if (i<N) {
    C[i] = A[i] + B[i];
  }
}

void sumOnHost(float *A, float *B, float *C, unsigned N) {
  for (unsigned i = 0; i<N; i++) {
    C[i] = A[i] + B[i];
  }
}

void initData(float * arr, unsigned sz) {
  srand((unsigned) time(NULL));
  for (unsigned i = 0; i < sz; i++) {
    arr[i] = (float) (( rand() & 0xFF)/10.0f);
  }
}

void printArr(float * arr, unsigned sz) {
  for (unsigned i = 0; i < sz-1; i++) {
    printf("%f\t", arr[i]);
  }
  printf("%f\n", arr[sz-1]);
}

int main(){
  unsigned N = 10;
  float A[N], B[N], C[N];
  initData(A, N);
  sleep(1);
  initData(B, N);
  printArr(A, N);
  printArr(B, N);
  //CPU calculation
  sumOnHost(A, B, C, N);
  printArr(C, N);
  
  float d_A[N], d_B[N], d_C[N];
  unsigned byteSize = N * sizeof(float);
  cudaMalloc((float **) &d_A, byteSize);
  cudaMalloc((float**) &d_B, byteSize);
  cudaMalloc((float**) &d_C, byteSize);
  
  cudaMemcpy(d_A, A, byteSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, byteSize, cudaMemcpyHostToDevice);
  
  sumOnDevice<<<1,1>>>(d_A, d_B, d_C, N);
  
  cudaMemcpy(C, d_C, byteSize, cudaMemcpyDeviceToHost);
  printArr(C, N);
  
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  
  return 0;
}
