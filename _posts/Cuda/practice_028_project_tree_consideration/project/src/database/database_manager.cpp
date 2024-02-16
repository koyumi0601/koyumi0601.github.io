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
    }
    catch(const std::exception& e)
    {
        std::cout << "Error in "<< __func__ << std::endl;
    }
}
