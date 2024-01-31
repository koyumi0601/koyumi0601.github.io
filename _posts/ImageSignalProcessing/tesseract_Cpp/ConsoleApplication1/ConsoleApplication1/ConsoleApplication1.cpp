#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <iostream>
#include <string>
#include <windows.h>
// https://maloveforme.tistory.com/181
// https://github.com/UB-Mannheim/tesseract/wiki
// https://github.com/tesseract-ocr/tessdata
int main()
{
  SetConsoleOutputCP(CP_UTF8); // 
  setvbuf(stdout, nullptr, _IOFBF, 1000); //
  char* outText;
  tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();

  if (api->Init(NULL, "kor")) {
    std::cerr << "Could not initialize tesseract." << std::endl;
    exit(1);
  }

  //Pix* image = pixRead("D:\Downloads\opencv_images\Letter_A.jpg");
  Pix* image = pixRead("D:\\Downloads\\opencv_images\\045.png");
  api->SetImage(image);

  outText = api->GetUTF8Text();
  std::cout << "OCR output:\n" << outText << std::endl;

  api->End();
  delete[] outText;
  pixDestroy(&image);

  // wait user input
  std::cout << "Press Enter to exit..." << std::endl;
  std::cin.get();

}