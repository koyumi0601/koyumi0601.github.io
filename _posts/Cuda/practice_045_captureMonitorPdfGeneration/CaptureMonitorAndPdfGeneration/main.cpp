#include <iostream>      // Input/output stream
#include <vector>        // Dynamic array container
#include <windows.h>     // Windows API header
#include <gdiplus.h>     // Microsoft GDI+ header
#include <shlwapi.h>     // Shell Light-weight Utility Functions header
#include <sstream>       // String stream processing
#include <gdiplusheaders.h> // Microsoft GDI+ header (declarations)
#include <Magick++.h>    // ImageMagick++ header

#pragma comment(lib, "gdiplus.lib") // Linker directive for GDI+ library
#pragma comment(lib, "shlwapi.lib") // Linker directive for Shell Light-weight API library


/**
 * @brief 지정된 폴더에서 특정 확장자를 가진 파일 목록을 검색합니다.
 *
 * @param folder 검색할 폴더 경로 (유니코드 문자열)
 * @param extension 검색할 확장자 (유니코드 문자열)
 * @return std::vector<std::string> 검색된 파일 목록 (유니코드 문자열을 ANSI 문자열로 변환하여 반환)
 */
std::vector<std::string> findFilesWithExtension(const std::wstring& folder, const std::wstring& extension) {
  std::vector<std::string> files;
  WIN32_FIND_DATAW findFileData;
  HANDLE hFind = FindFirstFileW((folder + L"\\*" + extension).c_str(), &findFileData);
  if (hFind != INVALID_HANDLE_VALUE) {
    do {
      if (!(findFileData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
        std::wstring wideFilePath = folder + L"\\" + findFileData.cFileName;
        std::string filePath(wideFilePath.begin(), wideFilePath.end());
        files.push_back(filePath);
      }
    } while (FindNextFileW(hFind, &findFileData) != 0);
    FindClose(hFind);
  }
  else {
    std::cerr << "Error: Failed to find files in directory." << std::endl;
  }

  return files;
}

/**
 * @brief 지정된 이미지 형식에 대한 인코더의 CLSID를 검색합니다.
 *
 * @param format 이미지 형식을 지정하는 문자열 포인터 (널 종료 유니코드 문자열)
 * @param pClsid 검색된 인코더의 CLSID를 저장할 CLSID 포인터
 * @return int CLSID 검색 결과를 나타내는 정수 값 (-1: 실패, 0 이상: 성공 및 인덱스)
 */
int GetEncoderClsid(const WCHAR* format, CLSID* pClsid) {
  UINT num = 0;          // Number of image encoders
  UINT size = 0;         // Size of the image encoder array in bytes
  // Get the size of the image encoder array
  Gdiplus::GetImageEncodersSize(&num, &size);
  if (size == 0)
    return -1;  // Failure
  // Create a buffer to hold the encoder array
  Gdiplus::ImageCodecInfo* imageCodecInfo = (Gdiplus::ImageCodecInfo*)(malloc(size));
  if (imageCodecInfo == NULL)
    return -1;  // Failure
  // Get the image encoder array
  GetImageEncoders(num, size, imageCodecInfo);
  // Find the CLSID for the specified format
  for (UINT i = 0; i < num; ++i) {
    if (wcscmp(imageCodecInfo[i].MimeType, format) == 0) {
      *pClsid = imageCodecInfo[i].Clsid;
      free(imageCodecInfo);
      return i;   // Success
    }
  }
  free(imageCodecInfo);
  return -1;  // Failure
}

/**
 * @brief 지정된 화면 좌표에 마우스를 클릭하는 함수입니다.
 *
 * @param x 클릭할 위치의 x 좌표
 * @param y 클릭할 위치의 y 좌표
 */
void ClickMouse(int x, int y) {
  INPUT input = { 0 };
  input.type = INPUT_MOUSE;
  input.mi.dwFlags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_MOVE;
  input.mi.dx = x * (65536 / GetSystemMetrics(SM_CXSCREEN));
  input.mi.dy = y * (65536 / GetSystemMetrics(SM_CYSCREEN));
  SendInput(1, &input, sizeof(INPUT));
  input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
  SendInput(1, &input, sizeof(INPUT));
  Sleep(10);
  input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
  SendInput(1, &input, sizeof(INPUT));
}

/**
 * @brief 지정된 영역을 캡처하고 각 페이지의 이미지를 저장하는 함수입니다.
 *
 * @param folder 이미지를 저장할 폴더의 경로
 */
void CaptureRegionFor(const std::wstring& folder) {
  RECT region; // 캡처할 영역의 좌표를 저장하는 구조체
  std::wstringstream ss; // 문자열 스트림 생성
  LPCWSTR filename = ss.str().c_str(); // 파일 이름을 LPCWSTR로 변환
  region.left = 280; // 캡처 영역의 왼쪽 좌표
  region.top = 80; // 캡처 영역의 상단 좌표
  region.right = 1700; // 캡처 영역의 오른쪽 좌표
  region.bottom = 980; // 캡처 영역의 하단 좌표
  Gdiplus::GdiplusStartupInput gdiplusStartupInput; // GDI+ 초기화 입력 구조체 생성
  ULONG_PTR gdiplusToken; // GDI+ 토큰 변수
  Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL); // GDI+ 초기화
  for (int j = 0; j < 156; j++) { // 각 페이지에 대한 반복문
    std::cout << "page pair number: " << j << std::endl; // 페이지 번호 출력
    int width = region.right - region.left; // 캡처 영역의 너비
    int height = region.bottom - region.top; // 캡처 영역의 높이
    HDC hdcScreen = GetDC(NULL); // 화면의 디바이스 컨텍스트 핸들 가져오기
    HDC hdcMem = CreateCompatibleDC(hdcScreen); // 호환되는 메모리 DC 생성
    HBITMAP hBitmap = CreateCompatibleBitmap(hdcScreen, width, height); // 호환되는 비트맵 생성
    HGDIOBJ hOldBitmap = SelectObject(hdcMem, hBitmap); // 메모리 DC에 비트맵 선택
    BitBlt(hdcMem, 0, 0, width, height, hdcScreen, region.left, region.top, SRCCOPY); // 화면에서 비트맵에 이미지 복사
    Gdiplus::Bitmap bitmap(hBitmap, NULL); // GDI+ 비트맵 객체 생성
    CLSID clsid; // 이미지 인코더의 CLSID
    GetEncoderClsid(L"image/png", &clsid); // PNG 이미지의 인코더 CLSID 가져오기
    CHAR exePath[MAX_PATH]; // 실행 파일 경로를 저장할 문자열 버퍼
    GetModuleFileNameA(NULL, exePath, MAX_PATH); // 실행 파일 경로 가져오기
    PathRemoveFileSpecA(exePath); // 경로에서 파일 이름 제거
    WCHAR imagePath[MAX_PATH]; // 이미지 파일 경로를 저장할 유니코드 문자열 버퍼
    swprintf_s(imagePath, MAX_PATH, L"%s\\_%03d.png", folder.c_str(), j); // 이미지 파일 경로 생성
    bitmap.Save(imagePath, &clsid); // 이미지를 지정된 경로에 저장
    SelectObject(hdcMem, hOldBitmap); // 비트맵 선택 해제
    DeleteObject(hBitmap); // 비트맵 객체 삭제
    DeleteDC(hdcMem); // 메모리 DC 삭제
    ReleaseDC(NULL, hdcScreen); // 화면 DC 해제
    Sleep(1000); // 1초 대기
    int clickX = 1880; // 마우스 클릭 위치의 x 좌표
    int clickY = 540; // 마우스 클릭 위치의 y 좌표
    ClickMouse(clickX, clickY); // 마우스 클릭 함수 호출
    Sleep(1000); // 1초 대기
  }
  Gdiplus::GdiplusShutdown(gdiplusToken); // GDI+ 종료
  std::cout << "Capture Complete\n"; // 캡처 완료 메시지 출력
}

/**
 * @brief 여러 이미지 파일을 하나의 PDF 파일로 변환하는 함수입니다.
 *
 * @param imageFiles 변환할 이미지 파일들의 경로를 포함하는 벡터
 * @param pdfFilename PDF 파일의 이름 및 경로
 */
void imagesToPdf(const std::vector<std::string>& imageFiles, const std::string& pdfFilename) {
  Magick::InitializeMagick(nullptr); // Magick++ 초기화
  std::vector<Magick::Image> images; // Magick++ 이미지 객체를 담을 벡터 생성
  for (const auto& file : imageFiles) { // 각 이미지 파일에 대한 반복문
    Magick::Image image; // Magick++ 이미지 객체 생성
    image.read(file); // 이미지 파일을 읽어들임
    images.push_back(image); // 벡터에 이미지 추가
  }
  Magick::writeImages(images.begin(), images.end(), pdfFilename); // 이미지들을 PDF 파일로 작성
  std::cout << "PDF file created successfully: " << pdfFilename << std::endl; // PDF 파일 생성 완료 메시지 출력

}

/**
 * @brief 프로그램의 메인 함수입니다.
 *
 * @details bookviewer의 영역을 캡쳐하고, 다음 페이지 버튼을 클릭한 후, png 파일로 만듭니다. 모든 png 파일을 pdf로 합칩니다.
 * 
 * @return 프로그램 종료 코드
 */
int main() {
  std::wstring folderPath = L".\\output"; // 이미지 파일이 저장된 폴더 경로
  CaptureRegionFor(folderPath); // 화면 캡처를 통해 이미지를 생성하는 함수 호출
  std::wstring extension = L".png"; // 찾고자 하는 이미지 파일의 확장자
  std::vector<std::string> pngFiles = findFilesWithExtension(folderPath, extension); // 지정된 폴더에서 확장자가 .png인 파일을 찾아 벡터로 반환
  std::string pdfFilename = "output_new.pdf"; // 생성할 PDF 파일의 이름
  imagesToPdf(pngFiles, pdfFilename); // 이미지 파일을 PDF 파일로 변환하는 함수 호출
  return 0; // 프로그램 종료
}