---
layout: single
title: "The essentials of C++, Chapter 04, Further study and Practice"
categories: language
tags: [language, programming, cpp, The essentials of C++]
toc: true
author_profile: false
sidebar:
    nav: "docs"
search: true

---

*The Essentials of C++, David hunter*



# Chapter 4. Pointer and Reference Types 포인터와 참조 유형


## 4.1 Pointer Variables 포인터 변수

##### 대략의 설명

- [포인터가 뭐고 왜 쓰는 건가요? Start 2:00](https://www.youtube.com/watch?v=u65F4ECaKaY)

##### 함수 내에서 포인터를 이용해서 접근했을 때

- 예제 [C언어 기초 프로그래밍 강좌 13강 - 포인터 (Pointer) Start 4:53](https://www.youtube.com/watch?v=JsS1A0xwozo&t=311s)

```cpp
#include <stdio.h>
void swap(int *x, int *y) // 포인터로 메모리 어느 위치에 있는 것인지 알려주고, 함수 내에서 직접 접근하여 변경한다.
{
    int temp;
    temp = *x;
    *x = *y;
    *y = temp;
}

int main(void)
{
    int x = 1;
    int y = 2;
    swap(&x, &y);
    printf("x = %d\n y = %d\n", x, y); // 원본 변수에 swap 동작이 반영되어 있다.
    return 0;
}
```

```cpp
x = 2
y = 1
```

- 포인터로 접근했기 때문에 **main에서 swap을 실행했을 때**, 입력 변수의 값이 변경되었다.
- 값을 복사해서 입력한 것도 아니고 return하지도 않았으므로, 메모리를 효율적으로 사용할 수 있었다


```cpp
#include <stdio.h>
void swap(int x, int y) // 입력 변수를 카피해서 또다른 메모리에 가지고 있는다. 이것을 함수에 넣어준다.
{
    int temp;
    temp = x;
    x = y;
    y = temp;
}

int main(void)
{
    int x = 1;
    int y = 2;
    swap(x, y);
    printf("x = %d\n y = %d\n", x, y); // 원본 변수는 영향받지 않았다.
    return 0;
}
```

```cpp
x = 1
y = 2
```

- main에서 swap을 실행시켰지만, 바깥에서는 값이 변경되지 않았다. return하면 동작은 할 것.

- 만약에, return 받아서 값을 변경하고 싶다면 아래와 같이 짤 수도 있다. return을 두 개할 수 없어서 구조체를 추가했다.

```cpp
#include <stdio.h>

// 구조체 정의
struct IntPair {
    int x;
    int y;
};

// swap 함수는 IntPair 구조체를 반환하도록 수정
IntPair swap(int x, int y)
{
    IntPair result;
    result.x = y; // x와 y 값을 교환
    result.y = x;
    
    return result; // 변경된 값들을 IntPair 구조체로 반환
}

int main(void)
{
    int x = 1;
    int y = 2;
    
    // swap 함수를 한 번 호출하여 결과를 변수 pair에 저장
    IntPair pair = swap(x, y);
    
    x = pair.x; // 구조체의 x 값을 x에 저장
    y = pair.y; // 구조체의 y 값을 y에 저장

    printf("x = %d\ny = %d\n", x, y);
    return 0;
}
```

##### 포인터를 사용한 것과, 구조체 + return을 활용한 예제의 장단점 비교
- 포인터 사용 예제:
    - 장점:
        - 메모리 사용량이 적음. 추가적인 구조체 반환을 필요로 하지 않음.
    - 단점:
        - 원본 변수 x와 y를 변경하기 위해 포인터를 사용하므로 변수에 대한 직접적인 조작이 가능하다.
        - 코드의 가독성이 떨어질 수 있음.
- 구조체 + return 사용 예제:
    - 장점:
        - 코드의 가독성이 높아짐. 구조체를 통해 반환 값들을 명확하게 표현.
        - 원본 변수 x와 y의 값을 변경하지 않고 새로운 값을 반환하므로 예기치 않은 부작용을 방지.
    - 단점:
        - 구조체를 반환하기 위해 추가적인 메모리 할당 및 복사가 필요할 수 있어서 약간의 오버헤드가 발생할 수 있음.

- 효율성 관점에서는 두 가지 방법 모두 거의 동일한 성능을 제공할 것으로 예상됩니다. 그러나 가독성과 코드의 안전성 측면에서 구조체 + return 방식이 더 나은 선택일 수 있습니다. 코드가 복잡해지거나 함수의 동작이 더 많은 변수와 데이터를 다루는 경우 구조체를 사용하여 결과를 반환하는 것이 코드의 이해와 유지보수를 쉽게 만들 수 있습니다.

##### 그럼 언제 포인터를 사용하나?

- 동적 메모리 할당: 
    - 프로그램 실행 중에 메모리를 동적으로 할당 - 배열, 객체. 필요한 만큼 메모리를 할당하고 해제.
- 데이터 구조: 
    - 연결 리스트, 이진 트리, 그래프 등
    - 데이터 구조에서 노드 간의 연결 유지, 다른 데이터 구조를 구현
- 함수 인자 전달: 
    - 함수에 대한 인자로 변수를 전달할 때 포인터를 사용하여 변수의 값을 변경하거나 원본 데이터를 수정
    - "참조에 의한 전달"와 유사한 방식으로 동작
- 동적 데이터 관리: 
    - 동적으로 생성된 데이터나 객체를 관리
    - 예를 들어, 동적으로 생성된 문자열, 배열, 또는 복잡한 데이터 구조
- 자료 구조와 알고리즘: 
    - 다양한 자료 구조와 알고리즘을 구현하고 최적화
    - 예를 들어, 배열 조작, 포인터 배열
- 하드웨어 접근: 
    - 하드웨어와 상호작용
    - 하드웨어 레지스터에 접근, 메모리 매핑 디바이스와 통신
- 다차원 배열: 
    - 다차원 배열의 배열 요소에 접근하고 반복
- 최적화: 
    - 메모리 효율성

#### 내 연습

##### 예제

```cpp
int x = 70;
int *y = x; // compile error. pointer variable needs address!
```

```cpp
int x = 70;
int *y = &x;

printf("y = %p\n", (void *)y); // 포인터 변수 y의 값, 즉 x의 주소를 출력
printf("*y = %d\n", *y);       // 포인터 변수 y가 가리키는 주소의 값, 즉 x의 값 출력
```

```cpp
y = 0x7ffd9f26c31c // 포인터 변수 y의 값, x의 주소
*y = 70           // 포인터 변수 y가 가리키는 주소의 값, x의 값
```


##### Case variation

- case 1. 선언과 할당을 동시에 수행하는 경우


```cpp
int y = 40;
int *x = &y; // Works! 포인터 변수 x에 y의 주소 값을 할당한다.
int *x = y; // Error! 포인터 변수 x에 y의 값을 할당하려고 시도했다. 서로 타입(포인터 변수, 값)이 다르므로 오류.
```

- case 2. 선언 따로

```cpp
int y = 40;
int *x;
*x = &y; // Error! 포인터 변수 x가 가리키는 메모리의 위치의 '값'에 y의 주소 값을 할당하려고 시도했다. 서로 타입(값, 주소)이 다르므로 오류.
*x = y; // Works! 포인터 변수 x가 가리키는 메모리의 위치의 '값'에 y의 값을 할당하려고 시도했다. 서로 타입(값)이 같으므로 동작이 수행된다. 주소 값에 대한 할당이 아니므로, x 주소와 y의 주소는 서로 다르고 값만 같다.
x = &y; // Works! 포인터 변수 x가 가리키는 메모리의 위치에, y의 주소 값을 할당했다. 주소가 같으므로 값도 같다.
x = y; // Error! 포인터 변수 x가 가리키는 메모리의 위치에, y의 값을 할당하려고 시도했다. 서로 타입(주소, 값)이 다르므로 오류.
```


##### 메모리 동적할당

- 좋은 코딩 습관

```cpp
int *p = nullptr; // 포인터 변수를 생성한다. 초기 값은 nullptr. 이 때에는 메모리를 점유하고 있지 않으며, 할당하기 전까지 정의되지 않은 동작하는 것을 방지한다. (안전)
p = new int(42); // 메모리를 할당하고 초기값을 설정한다. 설정하지 않으면 쓰레기 값이 들어가 있다.
std::cout << *p; // 포인터의 값에 접근하여 출력한다
delete p; // 포인터 메모리를 해제한다 -> 메모리 관리
p = nullptr; // 포인터를 다시 nullptr로 할당한다 -> 명시적으로 표현해줘야 한다. 
// 안정성, 버그 예방, 런타임 오류 방지. nullptr을 사용하려고 하면 런타임 에러가 발생하므로 프로그램이 예상치 못한 동작을 방지할 수 있다.
```

```cpp
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

```


- smart pointer를 사용하여 자동으로 관리 가능하다

## 4.2 Dynamic Arrays

## 4.3 Reference Variables

## 4.4 Recursive Structures

- Linked list
- 자료구조 참조








