#include "mainwindow.h"


mainwindow::mainwindow()
{
}

mainwindow::~mainwindow()
{
}

void mainwindow::printMainwindow()
{
    try
    {
        std::cout << "This is printMainwindow" << std::endl;
        //throw std::runtime_error("Simulation of an exception"); // exception simulation code
    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
}