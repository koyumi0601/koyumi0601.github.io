// // Kernel definition
// __global__ void VecAdd(float* A, float* B, float* C)
// {
//     int i = threadIdx.x;
//     C[i] = A[i] + B[i];
// }

// int main()
// {
//     // Kernel invocation with N threads
//     VecAdd<<<1, N>>>(A, B, C);
// }

#include <iostream>

int main() {
    int num1, num2;

    // 사용자로부터 두 정수 입력 받기
    std::cout << "첫 번째 정수를 입력하세요: ";
    std::cin >> num1;

    std::cout << "두 번째 정수를 입력하세요: ";
    std::cin >> num2;

    // 두 정수의 합 계산
    int sum = num1 + num2;

    // 합을 화면에 출력
    std::cout << "두 정수의 합은 " << sum << " 입니다." << std::endl;

    return 0;
}