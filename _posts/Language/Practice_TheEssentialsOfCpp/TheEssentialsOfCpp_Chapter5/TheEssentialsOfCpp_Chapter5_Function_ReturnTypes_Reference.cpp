#include <iostream>
using namespace std;

// 참조 형식 반환, int& max{}
// int& max(int a, int b) { // q, r, s, t의 값은 변하지 않는다
// int& max(int& a, int& b) { // 원본예제 
// int max(int a, int b) { // ++max, max(s, t) = 3에서 lvalue required as left operand of assignment
// int max(int& a, int& b) { // ++max, max(s, t) = 3에서 lvalue required as left operand of assignment
int& max(int& a, int& b) {
    return (a > b) ? a : b;
}

int main() {
    int q = 5, r = 10, s = 15, t = 20;
    cout << "q: " << q << ", r: " << r << ", s: " << s << ", t: " << t << endl;
    // q: 5, r: 10, s: 15, t: 20

    ++max(q, r); // r을 증가시킴, max(q, r)는 r의 별칭(alias)이다.
    cout << "q: " << q << ", r: " << r << ", s: " << s << ", t: " << t << endl;
    // q: 5, r: 11, s: 15, t: 20

    max(s, t) = 3; // t를 3으로 설정함, max(s, t)는 t의 별칭(alias)이다.
    cout << "q: " << q << ", r: " << r << ", s: " << s << ", t: " << t << endl;
    // q: 5, r: 11, s: 15, t: 3

    int x = max(q, s); // 함수 호출과 같이 동작하여 x를 15로 설정함.
    cout << "q: " << q << ", r: " << r << ", s: " << s << ", t: " << t << ", x: " << x << endl;
    // q: 5, r: 11, s: 15, t: 3, x: 15

    return 0;
}






