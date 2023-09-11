#include <iostream>

class Fraction {
public:
    Fraction(int top, int bot) : numerator(top), denominator(bot) {}
    Fraction(int top) : numerator(top), denominator(1) {}
    Fraction() : numerator(0), denominator(1) {}
    int top() { return numerator; }
    int bot() { return denominator; }

    friend Fraction operator*(const Fraction& f1, const Fraction& f2) {
        return Fraction(f1.numerator * f2.numerator, f1.denominator * f2.denominator);
    }

private:
    int numerator, denominator;
};

int main() {
    Fraction frac1(1, 2);
    Fraction frac2(3, 4);
    Fraction result = frac1 * frac2;
    std::cout << "Result: " << result.top() << "/" << result.bot() << std::endl;
    return 0;
}