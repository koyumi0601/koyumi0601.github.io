
// textbook example 1, previous expression before C++11
/*
#define NULL 0
// ...
void main()
{
    int *p = NULL, *q = NULL;
    // ...
    if (p == NULL) p = new integer;
    // ...
}
*/


// practical example 1
/*
#include <iostream> // 헤더 파일을 포함해야 합니다.

int main() { // main 함수의 반환 유형은 int 여야 합니다.
    int *p = nullptr, *q = nullptr; // nullptr을 사용하여 포인터를 초기화합니다.
    
    // p가 nullptr인 경우에만 할당
    if (p == nullptr) {
        p = new int; // int를 가리키는 포인터를 할당합니다.
        *p = 42; // 포인터가 가리키는 곳에 값을 할당합니다.
    }

    // 포인터 사용
    if (p != nullptr) {
        std::cout << "Value pointed by p: " << *p << std::endl;
    }

    // 메모리 해제
    if (p != nullptr) {
        delete p; // 동적으로 할당된 메모리를 해제합니다.
        p = nullptr; // 포인터를 다시 nullptr로 설정합니다.
    }

    return 0; // main 함수의 반환 값
}
*/


// #include <iostream>

// int main() {
    
//     // int *x = nullptr;
    
//     int y = 40;
//     std::cout << "&y : " << &y << std::endl;

//     // int *x = &y; // Works! x 선언, y의 주소를 포인터 변수 x에 할당한다.

//     // std::cout << "*x : " << *x << std::endl; // 포인터 변수 x가 가리키는 곳에는 값 40이 들어 있다.
//     // std::cout << "x : " << x << std::endl; // 포인터 변수 자체는 주소 값을 갖는다

//     // int *x = y; // Error! 포인터 변수에 정수를 할당할 수 없다. 타입이 서로 다르다.
    
//     // int *x; // 선언을 분리하는 경우,
//     // // *x = &y; // Error! *x는 메모리 위치 값에 값을 저장하라는 의미이다. 주소 값을 넣을 수 없다.
//     // *x = y; // Works! *x의 메모리 위치 값에, y의 값을 넣는다. 주소를 변경은 시킨 적이 없다.
    
//     // std::cout << "*x : " << *x << std::endl; // *x의 메모리 위치 값에 40이 들어가 있다
//     // std::cout << "x : " << x << std::endl; // x의 메모리 주소는 y와 서로 다르다

//     int *x; // 선언을 분리하는 경우,
//     x = &y; // Works! 포인터 변수에 y의 주소를 저장한다.

//     std::cout << "*x : " << *x << std::endl; // *x 메모리 위치는 y의 위치를 가리키고 있으므로, 서로 같은 값을 보여준다
//     std::cout << "x : " << x << std::endl; // 주소가 같으므로, 같은 값을 보여준다.

//     // int *x; 
//     // x = y; // Error! 포인터 변수에 y의 값을 저장할 수 없다.

//     return 0;
// }

#include <iostream>
int main(){
    int *x = nullptr; // 포인터 변수 생성. 초기 값은 아무것도 지시하지 않도록 한다. 아직 메모리를 점유하고 있지 않는다.
    int y = 40;
    std::cout << "y : " << y << std::endl; // y : 40
    std::cout << "&y : " << &y << std::endl; // &y : 0x7ffc66873a0c
    x = new int(10); // 메모리를 동적으로 할당. 초기 값은 10으로 설정.
    std::cout << "x : " << x << std::endl; // x : 0x5587797f82c0
    std::cout << "*x : " << *x << std::endl; // *x : 10
    *x = y; // y의 값만 받아온다. 메모리는 동적으로 할당되어 있다. 만약 x = &y라고 쓰면, y의 주소를 '정적'으로 할당하는 셈.
    std::cout << "x : " << x << std::endl; // 메모리는 변경되지 않았다. x : 0x5587797f82c0, &y : 0x7ffc66873a0c
    std::cout << "*x : " << *x << std::endl; // 값은 y의 값을 갖는다. *x : 40
    delete x; // x의 동적 메모리를 해제했다. 
    /*
    메모리 해제: 
     메모리 관리자(메모리 관리 유틸리티, 예: 힙 관리자) 혹은 운영체제 수준에서 처리
     힙 관리자는 할당된 메모리 블록을 추적하고, 해당 메모리를 반환하여 메모리 관리를 수행
     프로그램에서 delete 또는 free를 호출하면, 메모리 관리자는 해당 메모리 블록을 해제하고, 그 메모리를 다시 사용 가능한 상태로 표시
     메모리 해제하면, 해당 메모리 블록은 더 이상 해당 프로그램에 할당되지 않은 상태가 됨. 즉, 다시 할당하려면, 다시 할당 요청해야 함.
    */
    x = nullptr; // 포인터를 null pointer로 설정. 포인터 무효화 
    /*
     단순히 포인터 변수의 값을 바꾸는 것. 유효한 메모리를 가리키지 않도록 한다.
     메모리 관리자와 직접적인 상호작용 없음.
     포인터를 사용하려고 할 때, 예기치 않은 동작을 방지한다.
    */ 
    std::cout << "x : " << x << std::endl; // Error! 유효하지 않은 메모리 주소에 접근하려고 시도.
    std::cout << "*x : " << *x << std::endl; // Error! 유효하지 않은 메모리 주소에 접근하려고 시도.
}