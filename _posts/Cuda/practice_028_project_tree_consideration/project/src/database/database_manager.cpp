#include "database_manager.h"


databaseManager::databaseManager()
{
} 


databaseManager::~databaseManager()
{
}


void databaseManager::printDatabaseManager()
{
    try
    {
        std::cout << "This is print database manager" << std::endl;
        // throw std::runtime_error("Simulation of an exception"); // exception simulation code
    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
}
