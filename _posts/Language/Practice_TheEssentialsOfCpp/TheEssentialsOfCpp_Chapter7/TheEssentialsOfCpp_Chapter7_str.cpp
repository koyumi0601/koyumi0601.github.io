#include <cstring>
#include <iostream>
using namespace std;

int main()
{
    // char *s = "This is it", *delim = " ", *tok; // 원본
    char s[] = "This is it";  // 문자열을 수정 가능한 배열로 복사
    const char *delim = " ";
    char *tok;

    // set up strtok for processing s and return pointer to first token
    tok = strtok(s, delim);

    do {
        cout << tok << '\n';
        // get pointer to the next token, nullptr if none
        tok = strtok(nullptr, delim);
    } while (tok != nullptr);
    return 0;
}