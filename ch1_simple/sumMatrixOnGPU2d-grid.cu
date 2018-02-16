
#include <stdio.h>
#include "cuda_runtime.h"
#include <sys/time.h>

double cpuSecond() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void printMatrix(float *C, const int nx, const int ny) {
  float *ic = C;    //бережем оригинальный массив от изменения
  for (int iy = 0; iy < ny; iy++) {
    for (int ix = 0; ix < nx; ix++) {
      printf("%f\t", ic[ix]);
    }
    ic += nx;   //переставляем указатель
    printf("\n");   //печатаем новую строку
  }
  printf("============================================================================\n");
}

void checkResult(float * hostRes, float * devRes, const int nxy) {
  double eps = 1*0E-8;
  bool match = 1;
  int idx = 0;
  while (match && idx < nxy) {
    if (abs(hostRes[idx] - devRes[idx]) > eps) {
      match = 0;
      printf("Results do not match!\n");
    }
    idx++;
  }
  if (idx == nxy) {
    printf("Success: results match!\n");
  }
}
  void sumMatrixOnHost(float * const A, float * const B, float * C, const int nx, const int ny ) {
  //verification function
    float * ia = A;
    float * ib = B;
    float * ic = C;
    for (int iy = 0; iy < ny; iy++) {     //ny раз
      for (int ix = 0; ix < nx; ix++) {     //каждый элемент в отдельно взятой строке матрицы 
        ic[ix] = ia[ix] + ib[ix];
//         printf("%f\t", ic[ix]);
      }
      ia += nx; ib += nx; ic += nx;   //вот так вот просто переставляем указатели на следующую строку
//       printf("\n");
    }
//     printf("============================================================================\n");
  }

  __global__ void sumMatrixOnGPU(float * devA, float * devB, float * devC, const int nx, const int ny) {
    //подчитываем глобальный индекс
    int ix = threadIdx.x + blockIdx.x * blockDim.x;   //по x
    int iy = threadIdx.y + blockIdx.y * blockDim.y;   //по y
    //и совсем уже глобальный индекс нити:
    int ixy = iy * nx + ix;
    
    if (ix < nx && iy < ny) {
      devC[ixy] = devA[ixy] + devB[ixy];
    }
  }

  void initialFlt(float * ip, unsigned sz){
    for (unsigned i = 0; i < sz; i++)
      ip[i] = (float) (i);
  }

  void testSumOnGPU(float * const A, float * const B, float * C, const int nx, const int ny) {  
    int gpu = 0;
    double iStart = cpuSecond();
    cudaSetDevice(gpu);
    
    int nBytes = nx * ny * sizeof(float);
    //init device pointers
    float * devA, * devB, * devC;
    //не забывать правильно выделять память под массивы на устройстве!
    cudaMalloc((void **) &devA, nBytes);
    cudaMalloc((void **) &devB, nBytes);
    cudaMalloc((void **) &devC, nBytes);
    //copy data to GPU memory
    cudaMemcpy(devA, A, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B, nBytes, cudaMemcpyHostToDevice);
    
    //setup grid paramters
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1)/block.x, (ny + block.y -1)/block.y);
    
    //invoke kernel
    sumMatrixOnGPU<<<grid, block>>>(devA, devB, devC, nx, ny);
    cudaDeviceSynchronize();
    //copy result from GPU to host 
    //куда на хотсте, откуда на девайсе, размер, направление
    cudaMemcpy(C, devC, nBytes, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    double iEnd = cpuSecond();
    printf ("Test sumMatrixOnGPU<<<(%d, %d), (%d, %d)>>>. \nEsplaced time: %f\n", grid.x, grid.y, block.x, block.y, (iEnd - iStart));
    
    //check out result:
//     printMatrix(C, nx, ny);
    
    //set memory free
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    
    cudaDeviceReset();
  }

int main() {
    //==================настраиваем среду для теста===================
    //создадим две матрицы (точнее, указателя на них)
    float *A, *B, *hostRes, *devRes;
    // размерность складываемых матриц
    const int nx = 1 << 12;
    const int ny = 1 << 12;
    const int nxy = nx*ny;
    
    //узнаем размер матриц в байтах
    int nBytes = nxy * sizeof(float);
    
    //выделяем память под матрицы
    A = (float *) malloc(nBytes);
    B = (float *) malloc(nBytes);
    hostRes = (float *) malloc(nBytes);
    devRes = (float *) malloc(nBytes);
    
    //инициализируем матрицы
    initialFlt(A, nxy);
//     printMatrix(A, nx, ny);
    
    initialFlt(B, nxy);
//     printMatrix(B, nx, ny);
    
    //==================тестирование и тайминг ===================
    double iStart, iEnd;
    
    //==================тестирование на хосте===================
    iStart = cpuSecond();
    sumMatrixOnHost(A, B, hostRes, nx, ny);
    iEnd = cpuSecond();
    printf ("Test sumMatrixOnHost. \nEsplaced time: %f\n", (iEnd - iStart));
//     printMatrix(hostRes, nx, ny);
    
    //==================тестирование на девайсе===================
//     int dimx = 32;
//     int dimy = 32;
//     iStart = cpuSecond();
    testSumOnGPU(A, B, devRes, nx, ny);
//     iEnd = cpuSecond();
//     printf ("Test sumMatrixOnGPU<<<>>>. \nEsplaced time: %f\n", (iEnd - iStart));
    
    //==================проверяем совпадение результата===================
    checkResult(hostRes, devRes, nxy);

    //set memory free
    free(A);
    free(B);
    free(hostRes);
    free(devRes);
    
    return 0;
}
