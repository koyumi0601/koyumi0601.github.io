#ifndef LOGGER_H // 헤더가드. 라이브러리 중복 포함 방지
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <ctime>
#include <string>
#ifdef _WIN32
#include <filesystem>
namespace fs = std::filesystem;
#else
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#endif

#include <algorithm> // replace

enum class LogLevel 
{
    INFO,
    WARNING,
    ERROR
};


class logger
{
    public:
        static logger& getInstance();
        void printLogger();
        void log(LogLevel level, const std::string& message, const std::string& function);

    private:
        logger(std::string filename);
        ~logger();
        std::string getCurrentTimestamp();
        std::string logLevelToString(LogLevel level);

    private:
        std::string memberFilename;
        fs::path logsFilePath;
        std::ofstream logFile;     
};


#endif // LOGGER_H
