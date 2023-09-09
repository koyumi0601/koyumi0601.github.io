#include <iostream>
using namespace std;

struct student_record {
    int id;
    char classification;
    double gpa;
};

student_record read_record_method1()
// return a student record
{
    student_record r;
    cout << "id(int): ";
    cin >> r.id;
    cout << "classification(char): ";
    cin >> r.classification; 
    cout << "pga(double): ";
    cin >> r.gpa;
    return r;
}

student_record* read_record_method2()
// return a pointer to a student record
{
    student_record* r = new student_record;
    cout << "id(int): ";
    cin >> r->id;
    cout << "classification(char): ";
    cin >> r->classification;
    cout << "pga(double): ";
    cin >> r->gpa;
    return r;
}

int main()
{
    student_record test1, *test2;
    // student_record test1;
    test1 = read_record_method1();
    test2 = read_record_method2();
    // cout << test1.id << test2->id;
    cout << "test1.id: " << test1.id << endl;
    cout << "test2.id: " << test2->id << endl;
    delete test2;

    return 0;
}