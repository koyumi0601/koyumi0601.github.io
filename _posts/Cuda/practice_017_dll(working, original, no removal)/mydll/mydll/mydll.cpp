// mydll.cpp : Defines the exported functions for the DLL.
#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include <utility>
#include <limits.h>
#include "mydll.h"
#include <cstdlib> // malloc

int* my_list_add(int* a, int* b, int cnt) {
  int* ret = (int*)malloc(sizeof(int) * cnt);
  for (int i = 0; i < cnt; i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}