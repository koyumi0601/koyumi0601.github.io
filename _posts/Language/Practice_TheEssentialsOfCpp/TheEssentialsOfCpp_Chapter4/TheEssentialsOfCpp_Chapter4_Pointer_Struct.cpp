//   #include <iostream>

//   struct Person {
//       std::string name;
//       int age;
//   };

//   int main() {
//       Person person1;
//       person1.name = "Alice";
//       person1.age = 30;

//       // 포인터를 사용하여 구조체 필드에 접근
//       Person* ptr = &person1;
//       std::cout << "이름: " << ptr->name << ", 나이: " << ptr->age << std::endl;

//       // 동일한 동작을 하는 코드 (p->f를 (*p).f로 대체)
//       std::cout << "이름: " << (*ptr).name << ", 나이: " << (*ptr).age << std::endl;

//       return 0;
//   }

#include <iostream>
using namespace std;

struct Person {
    string name;
    int age;
};

int main() {
    Person person1;
    person1.name = "Alice";
    person1.age = 30;
    cout << person1.age << endl; 
    // 직접 구조체의 멤버변수에 접근. 구조체 객체가 스택에 할당되어 있을 때 사용

    Person* ptr; // 구조체의 포인터 변수 선언
    ptr = &person1; // 포인터 변수 ptr을 인스턴스 person1로 초기화

    // Person* ptr = &person1; 로 간략히 쓸 수 있음.

    cout << "포인터를 사용한 나이 출력: " << ptr->age << endl; 
    // 포인터 ptr을 사용하여 구조체 person1의 멤버변수 age에 접근. -> 연산자를 사용하여 접근할 수 있다.
    // 구조체에 동적으로 접근하거나, 구조체가 힙에 할당되어 있을 때 사용
    return 0;
}