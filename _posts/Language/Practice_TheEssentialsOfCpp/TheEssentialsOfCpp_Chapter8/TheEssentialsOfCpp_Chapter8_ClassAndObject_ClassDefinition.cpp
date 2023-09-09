#include <iostream>

class Vehicle {
public:
    int wheels;
    float weight;
    void initialize(float in_weight, int in_wheels = 4);
    float wheel_loading();
    void honk();
};

void Vehicle::initialize(float in_weight, int in_wheels) {
    weight = in_weight;
    wheels = in_wheels;
}

float Vehicle::wheel_loading() {
    std::cout << "Whel loading, temp" << std::endl;
    return 1.2;
}

void Vehicle::honk() {
    std::cout << "Beep Beep!" << std::endl;
}

int main() {
    Vehicle s1, s2;
    s1.initialize(2000);
    std::cout << s1.wheel_loading() << std::endl;
    s2.initialize(25, 2);
    s2.honk();
    // ...
    return 0;
}