#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
// #define DLLEXPORT extern "C" __declspec(dllexport) // window


__global__ void mul_const_kernel(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] * dconst;

    
    return ;
}


__global__ void add_const_kernel(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] + dconst;
    
    return ;

}


__global__ void mul_mat_kernel(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] * pdsrc2[id];

    return ;

}


__global__ void add_mat_kernel(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;

    if (idz >= nz || idy >= ny || idx >= nx) return;

    id = ny * nx * idz + nx * idy + idx;
    pddst[id] = pdsrc[id] + pdsrc2[id];
    
    return ;

}

// cpu interface
// DLLEXPORT void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // window only expression
extern "C" void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // available on linux, window
    float *gpddst = 0;
    float *gpdsrc = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));

    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(float));

    cudaMemcpy(gpdsrc, pdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    mul_const_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc, dconst, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc = 0;
    gpnsz = 0;

    return ;
}

// DLLEXPORT void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // window only
extern "C" void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz){ // available on window and linux 
    float *gpddst = 0;
    float *gpdsrc = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));

    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(float));

    cudaMemcpy(gpdsrc, pdsrc, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    add_const_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc, dconst, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc = 0;
    gpnsz = 0;

    return ;
}

// DLLEXPORT void mul_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // window only
extern "C" void mul_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // available on window and linux
    float *gpddst = 0;
    float *gpdsrc1 = 0;
    float *gpdsrc2 = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));


    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc1, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc2, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    mul_mat_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc1, gpdsrc2, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc1);
    cudaFree(gpdsrc2);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc1 = 0;
    gpdsrc2 = 0;
    gpnsz = 0;

    return ;
}


// DLLEXPORT void add_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // window only
extern "C" void add_mat(float *pddst, float *pdsrc1, float *pdsrc2, int *pnsz){ // available on window and linux
    float *gpddst = 0;
    float *gpdsrc1 = 0;
    float *gpdsrc2 = 0;
    int *gpnsz = 0;

    cudaMalloc((void **)&gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMalloc((void **)&gpnsz, 3*sizeof(int));


    cudaMemset(gpddst, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc1, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpdsrc2, 0, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float));
    cudaMemset(gpnsz, 0, 3*sizeof(int));

    cudaMemcpy(gpdsrc1, pdsrc1, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpdsrc2, pdsrc2, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyHostToDevice); // destination, source, memory size, direction
    cudaMemcpy(gpnsz, pnsz, 3*sizeof(int), cudaMemcpyHostToDevice); // destination, source, memory size, direction

    int nthread = 8;
    dim3 nblock(nthread, nthread, nthread);
    dim3 ngrid ((pnsz[0] + nthread - 1) / nthread,
                (pnsz[1] + nthread - 1) / nthread,
                (pnsz[2] + nthread - 1) / nthread);

    add_mat_kernel<<<ngrid, nblock>>>(gpddst, gpdsrc1, gpdsrc2, gpnsz);

    cudaMemcpy(pddst, gpddst, pnsz[0]*pnsz[1]*pnsz[2]*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(gpddst);
    cudaFree(gpdsrc1);
    cudaFree(gpdsrc2);
    cudaFree(gpnsz);

    gpddst = 0;
    gpdsrc1 = 0;
    gpdsrc2 = 0;
    gpnsz = 0;

    return ;
}




// nvcc -Xcompiler -fPIC math_cu.cu -shared -o libmath_cu.so
// nvcc -Xcompiler -fPIC math_cu.cu -shared -o libmath_cu_win.so
// nvcc error   : 'cudafe++' died with status 0xC0000005 (ACCESS_VIOLATION)