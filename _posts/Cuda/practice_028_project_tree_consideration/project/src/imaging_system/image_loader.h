#ifndef IMAGE_LOADER_H // 헤더가드. 라이브러리 중복 포함 방지
#define IMAGE_LOADER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <../src/logger/logger.h>

class imgLoader
{
    public:
        imgLoader();
        ~imgLoader();
        void printImageLoader();   
};

#endif // IMAGE_LOADER_H

void loadImg();
void runOCR_in_imageloader();

