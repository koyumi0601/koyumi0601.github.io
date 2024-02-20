//#include <iostream>
//#include <opencv2/opencv.hpp>
//#include "kernel.cuh"
//
//using namespace cv;
//using namespace std;
//
//
//int main()
//{
//  Mat image = imread("D:/Downloads/opencv_images/Eagle2.jpg");
//
//  if (image.empty()) // Check for failure
//  {
//    cout << "Could not open or find the image" << endl;
//    system("pause"); //wait for any key press
//    return -1;
//  }
//
//  String windowName = "My HelloWorld Window"; //Name of the window
//
//  namedWindow(windowName); // Create a window
//
//  imshow(windowName, image); // Show our image inside the created window.
//
//  waitKey(0); // Wait for any keystroke in the window
//
//  destroyWindow(windowName); //destroy the created window
//
//
//  const int arraySize = 5;
//  int a[arraySize] = { 1, 2, 3, 4, 5 };
//  int b[arraySize] = { 10, 20, 30, 40, 50 };
//  int c[arraySize] = { 0 };
//
//  cudaAdd(c, a, b, arraySize);
//
//  std::cout << "Result: ";
//  for (int i = 0; i < arraySize; ++i)
//  {
//    std::cout << c[i] << " ";
//  }
//  std::cout << std::endl;
//
//
//
//  return 0;
//
//
//}