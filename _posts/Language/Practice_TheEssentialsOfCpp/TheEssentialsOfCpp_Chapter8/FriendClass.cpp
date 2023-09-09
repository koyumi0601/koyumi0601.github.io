#include <iostream>

class vehicle; // Forward declaration
void reset(vehicle&, int, int); // Function declaration
// 전방 선언은 코드의 가독성을 향상시키고 순환 참조 문제를 방지하며, 클래스나 함수의 접근 권한을 명시하는 데 도움을 줍니다. 

class vehicle {
    friend void reset(vehicle&, int, int);

public:
    vehicle(float in_weight, int in_wheels = 4) : wheels(in_wheels), weight(in_weight) {}

    void showInfo() {
        std::cout << "Wheels: " << wheels << ", Weight: " << weight << std::endl;
    }

private:
    int wheels;
    float weight;
};

void reset(vehicle& v, int in_wheels, int in_weight) {
    // Note access to protected members of vehicle
    v.wheels = in_wheels;
    v.weight = in_weight;
}

int main() {
    vehicle myCar(1500.0, 4);

    std::cout << "Before Reset: ";
    myCar.showInfo();

    reset(myCar, 6, 2000);

    std::cout << "After Reset: ";
    myCar.showInfo();

    return 0;
}