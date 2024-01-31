#include <string>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <image_path>" << endl;
        return 1;
    }

    string outText, imPath = argv[1];


    Mat im = cv::imread(imPath, IMREAD_COLOR);

    if (im.empty()) {
        cout << "Failed to read the image." << endl;
        return 1;
    }

    tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);

    ocr->SetPageSegMode(tesseract::PSM_AUTO);

    ocr->SetImage(im.data, im.cols, im.rows, 3, im.step);

    ocr->SetSourceResolution(197); // image resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)

    outText = string(ocr->GetUTF8Text());

    cout << outText;

    ocr->End();
    
    return 0;
}
