#pragma once

#ifdef MATHLIBRARY_EXPORTS
#define MATHLIBRARY_API __declspec(dllexport)
#else
#define MATHLIBRARY_API __declspec(dllimport)
#endif

extern "C" MATHLIBRARY_API void mul_const(float* pddst, float* pdsrc, float dconst, int* pnsz);
extern "C" MATHLIBRARY_API void add_const(float* pddst, float* pdsrc, float dconst, int* pnsz);
extern "C" MATHLIBRARY_API void mul_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz);
extern "C" MATHLIBRARY_API void add_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz);