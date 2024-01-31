#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>


struct UserData {
    cv::Mat orgImg;              // not changed
    cv::Mat updatedImg;          // could be changed such as adding Rect
    cv::Rect rectInfo;           // left-top and right-bottom coordinates of selected sub-ROI
    std::string text_string;     // text string for displaying
    cv::Point cv_text_position;  // string poistion for diplaying
    bool selecting;              // whether select something or not
};


// mouse click callback event
void onMouse(int event, int x, int y, int flags, void* userdata) {
    UserData* data = static_cast<UserData*>(userdata);
    switch (event) {
        case cv::EVENT_LBUTTONDOWN: // when push LB
            // flush rectInfo info at every user rectInfo
            data->rectInfo.width = 0;
            data->rectInfo.height = 0;
            data->updatedImg = data->orgImg.clone();
            data->rectInfo.x = x;
            data->rectInfo.y = y;
            // draw red box
            cv::putText(data->updatedImg, data->text_string, data->cv_text_position, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Select ROI to get updatedImg's depth...", data->updatedImg);
            break;
        case cv::EVENT_LBUTTONUP: // when release LB
            data->selecting = true;
            data->rectInfo.width = x - data->rectInfo.x;
            data->rectInfo.height = y - data->rectInfo.y;
            // draw red box
            cv::rectangle(data->updatedImg, data->rectInfo, cv::Scalar(0, 0, 255), 1);
            cv::putText(data->updatedImg, data->text_string, data->cv_text_position, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
            cv::imshow("Select ROI to get updatedImg's depth...", data->updatedImg);
            break;
    }
}


int main(int argc, char* argv[]) {

    std::string windowName = "Select ROI to get updatedImg's depth...";
    cv::Point text_position(5, 50);
    std::string instruct_text_1 = cv::format("Drag the mouse to draw a rectangular sub-ROI ...");

    // get arguments
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    std::string outText, imPath = argv[1];

    // read an updatedImg using CV
    cv::Mat img = cv::imread(imPath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Failed to read the updatedImg." << std::endl;
        return 1;
    }

    // init a status of user
    UserData userdata;
    userdata.orgImg = img.clone();
    userdata.updatedImg = img.clone();
    userdata.selecting = false;
    userdata.text_string = instruct_text_1;
    userdata.cv_text_position = text_position;

    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::resizeWindow(windowName, 1024, 608);
    cv::putText(userdata.updatedImg, instruct_text_1, text_position, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 0, 255), 2);
    cv::imshow(windowName, userdata.updatedImg);

    // callback of mouse click
    cv::setMouseCallback(windowName, onMouse, &userdata);
    while (true) {
        int key = cv::waitKey(10);
        if (key == 'q') {
            break;
        }
        else if (key == 13) {  // enter=13
            if (userdata.selecting) {
                cv::Mat tempImage = userdata.orgImg.clone();
                cv::rectangle(tempImage, userdata.rectInfo, cv::Scalar(0, 255, 0), 2);
                std::string instruct_text_2 = cv::format("Sub-ROI determined... x=%d, y=%d, width=%d, height=%d",
                 userdata.rectInfo.x, userdata.rectInfo.y, userdata.rectInfo.width, userdata.rectInfo.height);
                cv::putText(tempImage, instruct_text_2, text_position, cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(0, 255, 0), 2);
                cv::imshow(windowName, tempImage);
                cv::waitKey(3000);
                break;
            }
        }
    }
    cv::destroyAllWindows();

    cv::Mat subRoi_img = userdata.orgImg(userdata.rectInfo); 

    // inference of an updatedImg using tesseract
    // init Tesseract OCR
    tesseract::TessBaseAPI* ocr_api = new tesseract::TessBaseAPI();
    if (ocr_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY) != 0) {
        std::cerr << "Could not initialize Tesseract." << std::endl;
        return 1;
    }
    ocr_api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr_api->SetVariable("tessedit_char_whitelist", "0123456789.");
    ocr_api->SetImage(subRoi_img.data, subRoi_img.cols, subRoi_img.rows, 3, subRoi_img.step);
    ocr_api->SetSourceResolution(250); // updatedImg resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
    outText = std::string(ocr_api->GetUTF8Text());

    // print inferred result out
    std::cout << outText;

    // remove objects
    ocr_api->End();

    return 0;
}
