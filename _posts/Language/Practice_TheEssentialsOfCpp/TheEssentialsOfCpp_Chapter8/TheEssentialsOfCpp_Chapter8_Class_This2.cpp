#include <iostream>

using namespace std;

class MyClass {
public:
    void PrintThis() {
        cout << "Address: " << this << endl;
    }
};

int main() {
    MyClass a, b;

    cout << "Address a: " << &a << endl;
    cout << "Address b: " << &b << endl;

    a.PrintThis();
    b.PrintThis();
}

// without this

// #include <iostream>

// using namespace std;

// class MyClass {
// public:
//     void PrintThis(MyClass *ptr ) {
//         cout << "Address: " << ptr << endl;
//     }
// };

// int main() {
//     MyClass a, b;

//     cout << "Address a: " << &a << endl;
//     cout << "Address b: " << &b << endl;

//     a.PrintThis(&a);
//     b.PrintThis(&b);
// }