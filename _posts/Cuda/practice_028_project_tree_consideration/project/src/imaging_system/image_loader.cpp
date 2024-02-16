#include "image_loader.h"



void runOCR_in_imageloader()
{
    std::cout << "This is runOCR_in_imageloader!\n";
}


imgLoader::imgLoader()
{
}


imgLoader::~imgLoader()
{
}


void imgLoader::printImageLoader()
{
    try
    {
        std::cout << "This is printImageLoader" << std::endl;
        //throw std::runtime_error("Simulation of an exception"); // exception simulation code
    }
    catch(const std::exception& e)
    {
        logger& loggerInstance = logger::getInstance();
        loggerInstance.log(LogLevel::ERROR, "An exception occured: " + std::string(e.what()), __PRETTY_FUNCTION__);
    }
}


void imgLoader::loadAndShow()
{
    std::cout << "This is image_loader! \n";
    std::string imagePath = "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_028_project_tree_consideration/project/resources/images/image_00003_smile.png";
    cv::Mat image = cv::imread(imagePath);
    if (image.empty()) {
        std::cerr << "Error: Failed to load image" << std::endl;
    }
    cv::imshow("Image", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}