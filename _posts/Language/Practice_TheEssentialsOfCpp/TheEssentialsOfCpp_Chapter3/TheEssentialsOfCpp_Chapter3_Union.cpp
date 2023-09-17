// #include <iostream>
// #include <cstring> 

// // 공용체 선언
// union MyUnion {
//     int intValue;
//     float floatValue;
//     char stringValue[20];
// };

// int main() {
//     MyUnion myData; // 공용체 변수 생성

//     // 멤버에 값 할당
//     myData.intValue = 42;
//     std::cout << "Integer Value: " << myData.intValue << std::endl;

//     // 다른 멤버에 값 할당
//     myData.floatValue = 3.1415f;
//     std::cout << "Float Value: " << myData.floatValue << std::endl;

//     // 다른 멤버에 값 할당
//     strcpy(myData.stringValue, "Hello, C++");
//     std::cout << "String Value: " << myData.stringValue << std::endl;

//     // 무효화된 값에 접근
//     std::cout << "Float Value: " << myData.floatValue << std::endl;

//     return 0;
// }


#include <iostream>

union grade_value {
    float score;
    char letter;
};

enum grade_type { letter_grade, numerical_grade };

struct result_record {
    grade_type type;
    grade_value value;
};

int main() {
    result_record results[25];
    results[1].type = letter_grade;
    results[1].value.letter = 'A'; 
    std::cout << "results[1].value.letter:" << results[1].value.letter << std::endl; 
    // ...
    results[20].type = numerical_grade;
    results[20].value.score = 85; 
    std::cout << "results[20].value.score:" << results[20].value.score << std::endl; 
    // ...
    return 0;
}
