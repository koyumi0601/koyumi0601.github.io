#include "ocr.h"

ocr::ocr()
{
}

ocr::~ocr()
{
}

void ocr::printOcr()
{
    try
    {
        std::cout << "This is printOcr" << std::endl;
        //throw std::runtime_error("Simulation of an exception"); // exception simulation code
    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
}
