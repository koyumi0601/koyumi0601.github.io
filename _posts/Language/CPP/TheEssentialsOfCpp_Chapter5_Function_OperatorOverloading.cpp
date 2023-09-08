#include <iostream>
using namespace std;

struct fraction {
    int numerator, denominator;
};

fraction operator*(fraction f1, fraction f2)
{
    fraction result;
    result.numerator = f1.numerator * f2.numerator;
    result.denominator = f1.denominator * f2.denominator;
    return result;
}

int main()
{
    fraction a = {2, 3}, b = {3, 5}, c;
    c = a * b; // c is 6/15
    cout << "c.numerator: " << c.numerator << " c.denominator: " << c.denominator << endl;
    return 0;
}