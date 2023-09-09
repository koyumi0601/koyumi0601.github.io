#include <iostream>
#include "power.h" // 헤더 파일 포함

int main() {
    int base = 2;
    int exponent = 3;
    
    // power 함수 호출
    int result = power(base, exponent);

    std::cout << base << "의 " << exponent << " 제곱은 " << result << "입니다." << std::endl;

    return 0;
}