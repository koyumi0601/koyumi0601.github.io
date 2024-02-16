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


void networkManager::handle_get(const Request& req, Response& res) {
    std::string content = "<html><body><h1>Hello, World!</h1></body></html>";
    res.set_content(content, "text/html");
}


void networkManager::openWeb()
{
    try
    {
        std::cout << "This is printHostname" << std::endl;
        Server svr;
        svr.Get("/", handle_get);
        std::cout << "Server started at http://localhost:8080" << std::endl;
        svr.listen("localhost", 8080);

    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
    
};