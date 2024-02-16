#include "image_loader.h"


void loadImg() {
    // 이미지 파일 경로

    std::cout << "This is image_loader! \n";
    std::string imagePath = "/home/ko/Documents/GitHub/koyumi0601.github.io/_posts/Cuda/practice_028_project_tree_consideration/project/resources/images/image_00003_smile.png";

    // 이미지 로드
    cv::Mat image = cv::imread(imagePath);

    // 이미지가 성공적으로 로드되었는지 확인
    if (image.empty()) {
        std::cerr << "Error: Failed to load image" << std::endl;
    }

    // 이미지 창에 이미지 표시
    cv::imshow("Image", image);

    // 키 입력 대기 (키를 누를 때까지 대기)
    cv::waitKey(0);

    // OpenCV 창 닫기
    cv::destroyAllWindows();

}

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