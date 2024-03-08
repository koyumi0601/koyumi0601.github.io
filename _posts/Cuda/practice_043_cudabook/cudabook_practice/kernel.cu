
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <iostream>
#define NUM_DATA 1024


__global__ void helloCUDA(void)
{
  printf("Hello CUDA from GPU! Grid(%d, %d, %d) Block(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

__global__ void printInt1(void)
{
  printf("Hello CUDA from GPU! Grid(%d, %d, %d) Block(%d, %d, %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}

void checkDeviceMemory(void)
{
  size_t free, total;
  cudaMemGetInfo(&free, &total);
  printf("Device memory (free/total) = %lld/%lld bytes\n", free, total);
}

__global__ void printData(int* _dDataPtr)
{
  printf("%d", _dDataPtr[threadIdx.x]);
}

__global__ void setData(int* _dDataPtr)
{
  _dDataPtr[threadIdx.x] = 2;
}

__global__ void vecAdd(int* _a, int* _b, int* _c, int size)
{
  int tID = threadIdx.x;
  if (tID < size)
  {
    _c[tID] = _a[tID] + _b[tID]; // 포인터를 받은 후 []연산자를 통해서 그 주소의 값에 접근할 수 있다. 해당 주소로부터 i만큼 떨어진 곳의 값을 가져온다.
    //printf("Thread ID: %d, Result: %d, %d, %d\n", tID, _c[tID], _a[tID], _b[tID]);
  }
}

int main()
{
  
  // (1)
  //dim3 gridDim(1, 1, 1);
  //dim3 blockDim(32, 1);
  //helloCUDA <<<gridDim, blockDim >>> ();

  // (2)
  //int* dDataPtr;
  //cudaError_t errorCode;
  //checkDeviceMemory();
  //errorCode = cudaMalloc(&dDataPtr, sizeof(int) * 1024 * 1024);
  //printf("cudaMalloc - %s\n", cudaGetErrorName(errorCode));
  //checkDeviceMemory();
  //errorCode = cudaMemset(dDataPtr, 0, sizeof(int) * 1024 * 1024);
  //printf("cudaMemset - %s\n", cudaGetErrorName(errorCode));
  //errorCode = cudaFree(dDataPtr);
  //printf("cudaFree - %s\n", cudaGetErrorName(errorCode));
  //checkDeviceMemory();

  // (3)
  //int data[10] = { 0 };
  //for (int i = 0; i < 10; i++)
  //{
  //  data[i] = 1;
  //}
  //int* dDataPtr;
  //cudaMalloc(&dDataPtr, sizeof(int) * 10);
  //cudaMemset(dDataPtr, 0, sizeof(int) * 10);
  //printf("Data in device: ");
  //printData <<<1, 10 >>> (dDataPtr);
  //cudaMemcpy(dDataPtr, data, sizeof(int) * 10, cudaMemcpyHostToDevice);
  //printf("\nHost -> Device: ");
  //printData <<<1, 10 >>> (dDataPtr);
  //setData <<<1, 10 >>> (dDataPtr);
  //cudaMemcpy(data, dDataPtr, sizeof(int) * 10, cudaMemcpyDeviceToHost);
  //printf("\nDevice -> Host: ");
  //for (int i = 0; i < 10; i++)
  //{
  //  printf("%d", data[i]);
  //}
  //cudaFree(dDataPtr);

  // (4) cpu, memory copy
  //int* a, * b, * c;
  //int memSize = sizeof(int) * NUM_DATA;
  //a = new int[NUM_DATA]; memset(a, 0, memSize);
  //b = new int[NUM_DATA]; memset(b, 0, memSize);
  //c = new int[NUM_DATA]; memset(c, 0, memSize);
  //for (int i = 0; i < NUM_DATA; i++)
  //{
  //  a[i] = rand() % 10;
  //  b[i] = rand() % 10;
  //}

  //for (int i = 0; i < NUM_DATA; i++)
  //{
  //  c[i] = a[i] + b[i];
  //  printf("c[i] = a[i] + b[i] = %d = %d + %d\n", c[i], a[i], b[i]);
  //}
  //delete[] a; delete[] b; delete[] c;

  //int* a, * b, * c, * hc;
  //printf("%d elements, memSize = %d bytes\n", NUM_DATA, memSize);

  //// Memory allocation on the host-side
  //a = new int[NUM_DATA]; memset(a, 0, memSize);
  //b = new int[NUM_DATA]; memset(b, 0, memSize);
  //c = new int[NUM_DATA]; memset(c, 0, memSize);
  //hc = new int[NUM_DATA]; memset(hc, 0, memSize);

  std::vector<int> a(NUM_DATA);
  std::vector<int> b(NUM_DATA);
  std::vector<int> c(NUM_DATA);
  std::vector<int> hc(NUM_DATA);

  // Data generation
  for (int i = 0; i < NUM_DATA; i++)
  {
    a[i] = rand() % 10;
    b[i] = rand() % 10;
  }

  // Vector sum on host (for performance comparison)
  for (int i = 0; i < NUM_DATA; i++)
  {
    hc[i] = a[i] + b[i];
  }


  ////////////////////////// device side
  std::vector<int> da(a.begin(), a.end());
  std::vector<int> db(b.begin(), b.end());
  std::vector<int> dc(NUM_DATA);
  
  int memSize = sizeof(int) * NUM_DATA;
  int* d_a, * d_b, * d_c; // CUDA 디바이스 메모리 주소를 저장하기 위한 포인터
  cudaMalloc(&d_a, NUM_DATA * sizeof(int));
  cudaMalloc(&d_b, NUM_DATA * sizeof(int));
  cudaMalloc(&d_c, NUM_DATA * sizeof(int));
  // vector: ptr, vector.data(): value
// std::vector<int> V = {1, 2, 3, 4, 5} 값을 통하여 초기화
// std::vector<int> V(values.begin(), values.end()); 주소를 통하여 값 할당
// []를 통하여 요소에 접근 가능

  //// data copy: Host -> Device
  cudaMemcpy(d_a, da.data(), NUM_DATA * sizeof(int), cudaMemcpyHostToDevice); // cuda device memory에 host의 memory 주소를 알려주고, 그 뒤로 사이즈 NUM_DATA * sizeof(int)만큼 복사해오기.
  cudaMemcpy(d_b, db.data(), NUM_DATA * sizeof(int), cudaMemcpyHostToDevice);


  //// kernel call
  //vecAdd << <1, NUM_DATA >> > (da, db, dc);
  // vecAdd <<<1, NUM_DATA>>> (da, db, dc); 와 같이, 직접 vector를 전달하는 것은 불가능하다.
  // da.data()에는 주소가, 역참조 *da.data()에는 값이 들어있다.
  //std::cout << "da.data() address: " << da.data() << std::endl; // 00000215167CCE40
  //std::cout << "da.data() value: " << *da.data() << std::endl; // 1


  vecAdd << <1, NUM_DATA >> > (d_a, d_b, d_c, NUM_DATA); // cuda kernel은 포인터에 대해서 동작함

  cudaDeviceSynchronize();
  //// copy result: Deviec -> Host
  //cudaMemcpy(c, dc, memSize, cudaMemcpyDeviceToHost);

  cudaMemcpy(dc.data(), d_c, memSize, cudaMemcpyDeviceToHost);


  //// release device memory
  //cudaFree(da); cudaFree(db); cudaFree(dc);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

  // Check result
  bool result = true;
  for (int i = 0; i < NUM_DATA; i++)
  {
    // if (hc[i] != c[i])
    if (hc[i] != dc[i])
    {
      printf("[%d] The result is not matched! (%d, %d)\n", i, hc[i], c[i]);
      result = false;
    }
  }
  if (result)
    printf("GPU works well!\n");


  //// Release host memory
  //delete[] a; delete[] b; delete[] c;



    return 0;
}
