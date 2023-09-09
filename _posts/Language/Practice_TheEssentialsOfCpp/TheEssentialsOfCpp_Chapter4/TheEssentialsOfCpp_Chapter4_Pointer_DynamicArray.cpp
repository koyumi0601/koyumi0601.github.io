// #include <iostream>

// int main() {
//     int *a, size;
//     std::cout << "How many data points are there? ";
//     std::cin >> size;
//     a = new int[size];
//     for (int i = 0; i < size; ++i) {
//         std::cin >> a[i];
//         std::cout << "a[i]: " << a[i] << std::endl;
//     }

//     delete[] a; 
//     return 0;
// }


// #include <iostream>

// int main()
// {
//     int a[2], size;
//     std::cout << "How many data points are there? ";
//     std::cin >> size;
//     for (int i = 0; i < size; ++i) {
//         std::cin >> a[i];
//         std::cout << "a[i]: " << a[i] << std::endl;
//     }
//     // stack smaching detected: terminated.
//     // Aborted (core dumped)
//     return 0;
// }

#include <iostream>

int main() {
    int *a, size;
    std::cout << "How many data points are there? ";
    std::cin >> size;
    a = new int[size];
    for (int i = 0; i < size; ++i) {
        std::cin >> a[i];
        std::cout << "a[i]: " << a[i] << std::endl;
        std::cout << "&a[i]: " << &a[i] <<std::endl;
    }

    delete[] a; 
    return 0;
}