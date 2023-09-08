// File "class.h"
#ifndef CLASS_H
#define CLASS_H
#include "student.h"
struct class_record {
    student_record students[100];
    int enrollment;
};
// ...
#endif // CLASS_H