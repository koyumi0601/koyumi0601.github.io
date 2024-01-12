// mydll.h - Contains declarations of math functions
#pragma once

#ifdef MATHLIBRARY_EXPORTS
#define MATHLIBRARY_API __declspec(dllexport)
#else
#define MATHLIBRARY_API __declspec(dllimport)
#endif

//extern "C" MATHLIBRARY_API int* my_list_add(int* a, int* b, int cnt);
extern "C" MATHLIBRARY_API void mul_const(float* pddst, float* pdsrc, float dconst, int* pnsz);