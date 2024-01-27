#pragma once
// mydll.h

#ifndef MYDLL_H // 헤더가드, 인클루전 가드, 같은 헤더파일이 포함되어도 컴파일 오류가 나지 않음.
#define MYDLL_H

void mul_const(float* pddst, float* pdsrc, float dconst, int* pnsz);
void add_const(float* pddst, float* pdsrc, float dconst, int* pnsz);
void mul_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz);
void add_mat(float* pddst, float* pdsrc, float* pdsrc2, int* pnsz);

#endif // MATH_OPERATIONS_H