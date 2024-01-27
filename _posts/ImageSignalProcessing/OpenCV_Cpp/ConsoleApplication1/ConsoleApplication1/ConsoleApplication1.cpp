#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
  // Read the image file
  Mat image = imread("D:/Downloads/opencv_images/Eagle2.jpg");

  if (image.empty()) // Check for failure
  {
    cout << "Could not open or find the image" << endl;
    system("pause"); //wait for any key press
    return -1;
  }

  String windowName = "My HelloWorld Window"; //Name of the window

  namedWindow(windowName); // Create a window

  imshow(windowName, image); // Show our image inside the created window.

  waitKey(0); // Wait for any keystroke in the window

  destroyWindow(windowName); //destroy the created window

  return 0;
}