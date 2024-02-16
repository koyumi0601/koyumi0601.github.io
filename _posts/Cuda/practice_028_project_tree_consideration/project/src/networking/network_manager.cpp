#include "network_manager.h" // <network_manager.h>에서는 안되지만 "network_manager.h"는 동일 경로의 헤더파일임을 알려준다.


networkManager::networkManager()
{
}

networkManager::~networkManager()
{
}

void networkManager::printNetworkManager()
{
    try
    {
        std::cout << "This is printNetworkManager" << std::endl;
        //throw std::runtime_error("Simulation of an exception"); // exception simulation code
    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
}