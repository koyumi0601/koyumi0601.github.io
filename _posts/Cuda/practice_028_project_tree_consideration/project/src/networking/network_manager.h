#ifndef NETWORK_MANAGER_H // 헤더가드. 라이브러리 중복 포함 방지
#define NETWORK_MANAGER_H

#include <iostream>
#include <../src/logger/logger.h>
#include <string>
#include "httplib.h" // downloaded and header copied (httplib.h)
#include <thread>


using namespace httplib;

class networkManager
{
    public:
        networkManager();
        ~networkManager();
        void printNetworkManager();
        void openWeb();

};


#endif // NETWORK_MANAGER_H

