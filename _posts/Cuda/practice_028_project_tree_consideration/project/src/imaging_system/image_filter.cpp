#include "image_filter.h"


imageFilter::imageFilter() 
{
}


imageFilter::~imageFilter()
{
}


void imageFilter::printImageFilter()
{
    try
    {
        std::cout << "This is printImageFilter" << std::endl;
        //throw std::runtime_error("Simulation of an exception"); // exception simulation code
    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
}