#ifndef NETWORK_MANAGER_H // 헤더가드. 라이브러리 중복 포함 방지
#define NETWORK_MANAGER_H

#include <iostream>
#include <../src/logger/logger.h>
#include <string>
#include "cpphttplib/httplib.h"


using namespace httplib;

class networkManager
{
    public:
        networkManager();
        ~networkManager();
        void printNetworkManager();
        void openWeb();
    private:
        void handle_get(const Request& req, Response& res);
};


#endif // NETWORK_MANAGER_H

