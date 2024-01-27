// mydll.cpp : Defines the exported functions for the DLL.
#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
//#include <utility>
//#include <limits.h>
#include "mydll.h"
//#include <cstdlib> // malloc

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

//
//int* my_list_add(int* a, int* b, int cnt) {
//  int* ret = (int*)malloc(sizeof(int) * cnt);
//  for (int i = 0; i < cnt; i++) {
//    ret[i] = a[i] + b[i];
//  }
//  return ret;
//}


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