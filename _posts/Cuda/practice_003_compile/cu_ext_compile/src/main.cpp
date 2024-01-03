#include "matmul.cuh"

#include <chrono>
#include <cmath>

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv)
{
    int m, n, k;
    if (argc < 3)
    {
        m = 500;
        n = 600;
        k = 700;
    }
    else
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    printf("matrix size A(%d, %d), B(%d, %d), C(%d, %d)\n", m, k, k, n, m, n);

    int *A = new int[m * k];
    int *B = new int[k * n];
    int *C = new int[m * n];
    int *C2 = new int[m * n];
    for (int i = 0; i < m * k; ++i)
    {
        A[i] = rand() % 100;
    }

    for (int i = 0; i < k * n; ++i)
    {
        B[i] = rand() % 100;
    }

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            int sum = 0;
            for (int d = 0; d < k; ++d)
            {
                sum += A[i * k + d] * B[n * d + j];
            }
            C2[n * i + j] = sum;
        }
    }
    std::chrono::duration<double> timeCpuMatmul = std::chrono::system_clock::now() - start;

    matMulWrapper(A, B, C, m, n, k);

    bool matrixCompare = true;
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            if (C[n * i + j] != C2[n * i + j])
            {
                printf("wrong value at (%d, %d) C1 = %d C2 = %d\n", i, j, C[n * i + j], C2[n * i + j]);
                matrixCompare = false;
            }
        }
    }

    printf("cpu matmul elapsed : %lf(ms)\n", timeCpuMatmul * 1000);

    if (matrixCompare == true)
    {
        printf("matmul results using CPU and GPU are matched\n");
    }

    delete[] A, B, C, C2;
    return 0;
}