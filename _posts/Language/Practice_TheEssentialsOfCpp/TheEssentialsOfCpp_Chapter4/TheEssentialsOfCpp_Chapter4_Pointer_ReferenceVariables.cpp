#include <iostream>
#include <string>

using namespace std;

class Person {
public:
    string name;
    int age;

    Person(const string& n, int a) : name(n), age(a) {} 
    
    // Person(string& n, int a) : name(n), age(a) {} 
    // 클래스 생성자. 클래스와 이름이 같다. 객체 생성시 자동 호출되며, 객체를 초기화
    // 입력변수 2개 - const string& n, int a
    // reference variable를 썼다. 문자열 복사가 발생하지 않고 효율적으로 문자열을 전달할 수 있다.
    // Person(const string n, int a) : name(n), age(a) {} 라고 작성해도 동작한다. 문자열 복사가 발생한다.
    // Person(string n, int a) : name(n), age(a) {} 라고 작성해도 동작한다. 함수 내에서 string 변경이 가능하여 예기치 않은 부작용이 발생할 수 있다.
    // name(n), age(a): member initializer. 생성자 위에 string name, int age가 있는데, 생성시에 인자를 전달 받아서 멤버 변수를 초기화 해준다.

    void introduce() {
        
        cout << "My name is " << name << " and I am " << age << " years old." << endl;
    }
};

int main() {
    Person person1("Alice", 30);
    Person person2("Bob", 25);

    cout << "Before modification:" << endl;
    person1.introduce();
    person2.introduce();

    // 참조 변수를 사용하여 객체 수정
    Person& ref = person1; // Person class의 객체를 참조할 수 있는 참조 변수 ref를 선언하고, person1 객체를 참조한다.
    ref.name = "Alicia";
    ref.age = 31;

    cout << "\nAfter modification:" << endl;
    person1.introduce();
    person2.introduce();

    return 0;
}

