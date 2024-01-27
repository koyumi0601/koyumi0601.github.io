//// mydll.cpp : Defines the exported functions for the DLL.
//#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
//#include <utility>
//#include <limits.h>
//#include "mydll.h"
//#include <cstdlib> // malloc
//
//int* my_list_add(int* a, int* b, int cnt) {
//  int* ret = (int*)malloc(sizeof(int) * cnt);
//  for (int i = 0; i < cnt; i++) {
//    ret[i] = a[i] + b[i];
//  }
//  return ret;
//}




// mydll.cpp : Defines the entry point for the DLL application.
#include "pch.h"
#include <utility>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "mydll.h"

/*
 * Multiplies each element of the source array by a constant value.
 *
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to be multiplied.
 * @param dconst    Constant value to multiply each element.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 *
 * p represents pointer
 * d represents data
 */
void mul_const(float* pddst, float* pdsrc, float dconst, int* pnsz) {
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

  return;
}

/*
 * Adds a constant value to each element of the source array.
 *
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to which the constant is added.
 * @param dconst    Constant value to be added to each element.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void add_const(float* pddst, float* pdsrc, float dconst, int* pnsz) {
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

  return;

}

/*
 * Multiplies two arrays element-wise and stores the result in the destination array.
 *
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to be multiplied with another source array.
 * @param pdsrc2    Second source array for element-wise multiplication.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void mul_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz) {
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

  return;

}

/*
 * Adds two arrays element-wise and stores the result in the destination array.
 *
 * @param pddst     Destination array to store the result.
 * @param pdsrc     Source array to which the second source array is added.
 * @param pdsrc2    Second source array for element-wise addition.
 * @param pnsz      Integer array containing the dimensions [nx, ny, nz].
 */
void add_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz) {
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

  return;

}

// gcc -c -fPIC math_clang.c -o math_clang.o
// gcc -shared math_clang.o -o libmath_clang.so