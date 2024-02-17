#include "network_manager.h" // <network_manager.h>에서는 안되지만 "network_manager.h"는 동일 경로의 헤더파일임을 알려준다.


networkManager::networkManager()
{
}

networkManager::~networkManager()
{
}

void networkManager::printNetworkManager()
{
    try
    {
        std::cout << "This is printNetworkManager" << std::endl;
        //throw std::runtime_error("Simulation of an exception"); // exception simulation code
    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
}



void networkManager::openWeb()
{
    try
    {
        std::cout << "This is networkManager::openWeb" << std::endl;
        httplib::Server svr;
        svr.Get("/", [](const httplib::Request& req, httplib::Response& res) 
        {
            std::ifstream t("../src/networking/index.html"); 
            if (!t.is_open()) 
            {
                res.status = 404;
                res.set_content("File not found", "text/plain");
            }
            std::string str((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
            res.set_content(str, "text/html");
        });

        // calculate function
        svr.Post("/calculate", [](const httplib::Request& req, httplib::Response& res) 
        {
            int num1 = std::stoi(req.get_param_value("num1"));
            int num2 = std::stoi(req.get_param_value("num2"));
            int result = num1 + num2;
            std::string response = "Result: " + std::to_string(result);
            res.set_content(response, "text/plain");
        });

        // Handle POST request for the slider endpoint
        svr.Post("/calcSlider", [](const httplib::Request& req, httplib::Response& res) 
        {
            int sliderValue = std::stoi(req.get_param_value("sliderValue"));
            int result = sliderValue + 200;
            res.set_content(std::to_string(result), "text/plain");
        });

        // shutdown funtion
        svr.Post("/shutdown", [&](const httplib::Request& req, httplib::Response& res) 
        {
            res.set_content("Server is shutting down...", "text/plain");
            svr.stop();
        });

                // Handle GET request for the image endpoint
        svr.Get("/image", [](const httplib::Request& req, httplib::Response& res) 
        {
            std::ifstream imageFile("../resources/images/image_00003_smile.png", std::ios::binary);
            if (!imageFile.is_open()) 
            {
                res.status = 404;
                res.set_content("Image file not found", "text/plain");
                return;
            }

            std::string imageData((std::istreambuf_iterator<char>(imageFile)), std::istreambuf_iterator<char>());
            res.set_content(imageData, "image/png");
        });


        std::cout << "Server started at http://localhost:8080" << std::endl;
        svr.listen("localhost", 8080);
        

    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
    
};