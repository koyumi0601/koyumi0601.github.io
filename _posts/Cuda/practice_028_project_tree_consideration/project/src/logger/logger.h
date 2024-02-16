#ifndef LOGGER_H // 헤더가드. 라이브러리 중복 포함 방지
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#include <experimental/filesystem>
#include <algorithm> // replace

namespace fs // forward declaration
{
    using namespace std::experimental::filesystem;
}


enum class LogLevel 
{
    INFO,
    WARNING,
    ERROR
};


class logger
{
    public:
        logger(std::string filename);
        ~logger();
        void printLogger();
        void log(LogLevel level, const std::string& message, const std::string& function);

    private:
        std::string getCurrentTimestamp();
        std::string logLevelToString(LogLevel level);

    private:
        std::string memberFilename;
        fs::path logsFilePath;
        std::ofstream logFile;     
};


#endif // LOGGER_H
