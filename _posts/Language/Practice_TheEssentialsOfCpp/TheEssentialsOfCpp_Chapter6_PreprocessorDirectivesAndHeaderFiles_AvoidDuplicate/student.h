// File "student.h"
#ifndef STUDENT_H
#define STUDENT_H
struct student_record {
    int id;
    char classification;
    float gpa;
};
void read_student(student_record&);
// ...
#endif // STUDENT_H