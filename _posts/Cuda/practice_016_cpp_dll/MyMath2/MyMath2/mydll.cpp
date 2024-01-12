//// mydll.cpp : Defines the exported functions for the DLL.
//#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
//#include <utility>
//#include <limits.h>
//#include "mydll.h"
////#include <stdlib.h>


// mydll.cpp : Defines the exported functions for the DLL.
#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include <utility>
#include <limits.h>
#include "mydll.h"
#include <cstdlib> // 또는 #include <stdlib.h>


int* my_list_add(int* a, int* b, int cnt) {
  int* ret = (int*)malloc(sizeof(int) * cnt);
  for (int i = 0; i < cnt; i++) {
    ret[i] = a[i] + b[i];
  }
  return ret;
}

//int* my_list_add(int* a, int* b, int cnt) {
//  int* ret = (int*)malloc(sizeof(int) * cnt);
//  if (ret != NULL) {
//    // 할당된 메모리를 0으로 초기화
//    memset(ret, 0, sizeof(int) * cnt);
//
//    for (int i = 0; i < cnt; i++) {
//      ret[i] = a[i] + b[i];
//    }
//  }
//
//  return ret;
//}