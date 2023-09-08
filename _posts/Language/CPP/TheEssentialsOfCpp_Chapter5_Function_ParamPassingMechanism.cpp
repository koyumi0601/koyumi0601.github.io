// Pass by value

// #include <iostream>

// void modifyValue(int x) {
//     x = x * 2;
// }

// int main() {
//     int originalValue = 10;
//     modifyValue(originalValue);
//     std::cout << "originalValue: " << originalValue << std::endl; // 10
//     return 0;
// }


// Pass by Reference

// #include <iostream>

// void modifyValue(int& x) {
//     x = x * 2;
// }

// int main() {
//     int originalValue = 10;
//     modifyValue(originalValue);
//     std::cout << "originalValue: " << originalValue << std::endl; // 20
//     return 0;
// }


// Pass by Pointer

// #include <iostream>

// void modifyValue(int* x) {
//     *x = *x * 2;
// }

// int main() {
//     int originalValue = 10;
//     modifyValue(&originalValue);
//     std::cout << "originalValue: " << originalValue << std::endl; // 20
//     return 0;
// }




// #include <iostream>

// // 상수 참조를 사용하여 두 수의 합을 계산하는 함수
// int add(const int& a, const int& b) {
//     return a + b;
// }

// int main() {
//     int num1 = 5;
//     int num2 = 7;

//     // add 함수를 호출하고 결과를 출력
//     int sum = add(num1, num2);

//     std::cout << "Sum: " << sum << std::endl;

//     // 원래 변수는 수정되지 않음
//     std::cout << "num1: " << num1 << std::endl;
//     std::cout << "num2: " << num2 << std::endl;

//     return 0;
// }



// #include <iostream>

// void modifyValue(const int& x) {
//     // 아래 코드는 에러를 발생시킬 것이므로 주석 처리
//     // x = x * 2;
// }

// int main() {
//     int originalValue = 10;
//     modifyValue(originalValue);
//     std::cout << "originalValue: " << originalValue << std::endl;
//     return 0;
// }