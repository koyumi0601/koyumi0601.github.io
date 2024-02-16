#include "logger.h"


logger& logger::getInstance() {
    static logger instance("log");
    return instance;
}


logger::logger(std::string filename):memberFilename(filename) // init member in cpp only. 헤더에서는 초기화하지 않는다.
{
    fs::path logsFolderPath = fs::current_path().parent_path() / "logs";
    if (!fs::exists(logsFolderPath))
    {
        fs::create_directory(logsFolderPath);
    }
    std::string timestamp = getCurrentTimestamp();
    std::replace(timestamp.begin(), timestamp.end(), '-', '_');
    std::replace(timestamp.begin(), timestamp.end(), ':', '_');
    std::replace(timestamp.begin(), timestamp.end(), ' ', '_');
    logsFilePath = logsFolderPath / (memberFilename + "_" + timestamp + ".txt");
    logFile.open(logsFilePath, std::ios::app); //  std::ios::app = append mode
}


logger::~logger()
{
    if (logFile.is_open()) 
    {
        logFile.close();
    }
}


void logger::printLogger()
{
    std::cout << "This is logger" << std::endl;
}


void logger::log(LogLevel level, const std::string& message, const std::string& function) 
{
    if (logFile.is_open()) 
    {
        logFile << "[" << getCurrentTimestamp() << "] ";
        logFile << "[" << logLevelToString(level) << "] ";
        logFile << "[" << function << "] ";
        logFile << message << std::endl;
    }
}


std::string logger::getCurrentTimestamp()
{
    std::time_t now = std::time(nullptr);
    char timestamp[20];
    std::strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", std::localtime(&now));
    return timestamp;
}


std::string logger::logLevelToString(LogLevel level) 
{
    switch (level) 
    {
        case LogLevel::INFO:
            return "INFO";
        case LogLevel::WARNING:
            return "WARNING";
        case LogLevel::ERROR:
            return "ERROR";
        default:
            return "UNKNOWN";
    }
}

