// complier directives always begin with # and are typically placed at the top of a file

// include the I/O functions
#include <iostream>

// define a constant
#define PI 3.1415927

// A function that, given a radius, returns the circumference
float circumf(float radius) {
    return 2 * PI * radius;
}

int main() { // The main function
    float r; // Declare r
    char c = 'y'; // Declare c and set to 'y'
    while (c == 'y') { // While c is equal to 'y', do ...
        // Output newline and prompt
        std::cout << "\nEnter the radius of the circle: ";
        std::cin >> r; // Input r
        std::cout << "Circumference: " << circumf(r) << '\n'; // Output circumference

        // Output prompt
        std::cout << "Enter y to continue, n to exit: ";
        std::cin >> c; // Input c
    } // End loop
    
    return 0;
}
