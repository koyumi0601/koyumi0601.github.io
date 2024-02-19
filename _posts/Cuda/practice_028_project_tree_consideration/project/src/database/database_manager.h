#ifndef DATABASE_MANAGER_H // 헤더가드. 라이브러리 중복 포함 방지
#define DATABASE_MANAGER_H

#include <iostream>
#include <../src/logger/logger.h>
#include <sqlite3.h>


class databaseManager 
{
    public:
        databaseManager();
        ~databaseManager();
        void printDatabaseManager();
        void createDatabase();
        void viewDatabase();
        void insertDatabase();
};


#endif // DATABASE_MANAGER_H


