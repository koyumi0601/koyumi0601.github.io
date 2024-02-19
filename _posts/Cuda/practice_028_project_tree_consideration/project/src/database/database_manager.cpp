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


void databaseManager::createDatabase()
{
    sqlite3* db;
    char* err_msg = nullptr;

    // SQLite 데이터베이스 파일 경로
    const char* db_file = "myDb.db";

    // SQLite 데이터베이스 열기 또는 생성
    int rc = sqlite3_open_v2(db_file, &db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, nullptr);
    if (rc != SQLITE_OK) {
        std::cerr << "Can't open or create database: " << sqlite3_errmsg(db) << std::endl;
        sqlite3_close(db);
        return;
    }

    // 사용자 테이블 생성
    const char* create_table_query = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)";
    rc = sqlite3_exec(db, create_table_query, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << err_msg << std::endl;
        sqlite3_free(err_msg);
    }

    // 데이터베이스 닫기
    sqlite3_close(db);
}


void databaseManager::viewDatabase()
{
    sqlite3* db;
    char* err_msg = nullptr;

    // SQLite 데이터베이스 파일 경로
    const char* db_file = "myDb.db";

    // SQLite 데이터베이스 열기
    int rc = sqlite3_open(db_file, &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
    }

    // 사용자 테이블 조회 쿼리 실행
    const char* query = "SELECT * FROM users";
    rc = sqlite3_exec(db, query, [](void* data, int argc, char** argv, char** /* azColName */) -> int {
        for (int i = 0; i < argc; i++) {
            std::cout << argv[i] << " ";
        }
        std::cout << std::endl;
        return 0;
    }, nullptr, &err_msg);
    
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << err_msg << std::endl;
        sqlite3_free(err_msg);
    }

    // 데이터베이스 닫기
    sqlite3_close(db);
}


void databaseManager::insertDatabase()
{
    sqlite3* db;
    char* err_msg = nullptr;

    // SQLite 데이터베이스 파일 경로
    const char* db_file = "myDb.db";

    // SQLite 데이터베이스 열기
    int rc = sqlite3_open(db_file, &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // 사용자 테이블 생성
    const char* create_table_query = "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)";
    rc = sqlite3_exec(db, create_table_query, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        sqlite3_close(db);
        return;
    }

    // 사용자 데이터 삽입
    const char* insert_data_query = "INSERT INTO users (name) VALUES ('John')";
    rc = sqlite3_exec(db, insert_data_query, nullptr, nullptr, &err_msg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << err_msg << std::endl;
        sqlite3_free(err_msg);
    }

    // 데이터베이스 닫기
    sqlite3_close(db);
}