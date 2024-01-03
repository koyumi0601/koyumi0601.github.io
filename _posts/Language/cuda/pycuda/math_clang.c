#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

void mul_const(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] * dconst;
            }
        }
    }
    
    return ;
}


void add_const(float *pddst, float *pdsrc, float dconst, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] + dconst;
            }
        }
    }
    
    return ;

}


void mul_mat(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] * pdsrc2[id];
            }
        }
    }
    
    return ;

}


void add_mat(float *pddst, float *pdsrc, float *pdsrc2, int *pnsz) {
    int nx = pnsz[0];
    int ny = pnsz[1];
    int nz = pnsz[2];
    int id = 0;

    for (int idz = 0; idz < nz; idz++) {
        for (int idy = 0; idy < ny; idy++) {
            for (int idx = 0; idx < nx; idx++) {
                id = ny * nx * idz + nx * idy + idx;
                pddst[id] = pdsrc[id] + pdsrc2[id];
            }
        }
    }
    
    return ;

}

// gcc -c -fPIC math_clang.c -o math_clang.o
// gcc -shared math_clang.o -o libmath_clang.so