#ifndef OCR_H // 헤더가드. 라이브러리 중복 포함 방지
#define OCR_H

#include <iostream>
#include <../src/logger/logger.h>


class ocr 
{
    public:
        ocr();
        ~ocr();
        void printOcr();
};


#endif // OCR_H

