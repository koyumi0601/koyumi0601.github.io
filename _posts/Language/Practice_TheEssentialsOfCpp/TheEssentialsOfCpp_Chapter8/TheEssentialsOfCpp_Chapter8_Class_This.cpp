#include <iostream>

class MyClass {
public:
    MyClass(int value) : data(value) {}

    // 메서드 내에서 this 포인터 사용
    void printAddress() {
        std::cout << "Address of the current object: " << this << std::endl;
    }

    // *this를 사용하여 현재 객체의 데이터 출력
    void printData() {
        std::cout << "Data of the current object: " << this->data << std::endl;
    }

    // *this를 반환하는 메서드
    MyClass doubleData() {
        this->data *= 2;
        return *this;
    }

private:
    int data;
};

int main() {
    MyClass obj1(42);
    MyClass obj2(17);

    std::cout << "Address of obj1: " << &obj1 << std::endl;
    std::cout << "Address of obj2: " << &obj2 << std::endl;

    obj1.printAddress();
    obj2.printAddress();

    // *this를 사용하여 객체 내의 데이터를 두 배로 만듭니다.
    obj1.doubleData();
    obj2.doubleData();    

    obj1.printData();
    obj2.printData();

    obj1.doubleData().doubleData(); // 메서드 체이닝
    obj2.doubleData().doubleData(); // 메서드 체이닝

    obj1.printData();
    obj2.printData();


    return 0;
}