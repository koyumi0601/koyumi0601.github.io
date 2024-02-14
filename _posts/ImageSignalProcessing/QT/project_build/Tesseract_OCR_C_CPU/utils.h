#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <QPoint>
#include <QPointF>

namespace Utils
{
    std::vector<std::string> getAllPNGFiles(const std::string& directoryPath);
    bool andArray(const std::vector<bool> array);
    bool orArray(const std::vector<bool> array);
    int findFirstFalse(const std::vector<bool> array);
    int signInt(const int value);
    std::tuple<std::vector<int>, std::vector<int>, int> calculateProfileData(const cv::Mat& inputImage, int minBrightnessForThreshold);
    int findFirstNonZeroIndex(const std::vector<int>& arr);
    int findFirstZeroIndex(const std::vector<int>& arr);
    int findLastNonZeroIndex(const std::vector<int>& arr);
    int findFirstMaxIndex(const std::vector<int>& arr);
    int findLastMaxIndex(const std::vector<int>& arr);
    QPointF intersectionPointof2Lines(const QPoint topLeftPoint, const QPoint bottomLeftPoint, const QPoint topRightPoint, const QPoint bottomRightPoint);
    qreal intersectionPointOfLineAndCircle(qreal pos_x, const QPointF& apexPoint, qreal radius);
    int findFirstNonMinIndex(const std::vector<int>& arr);
    int findLastNonMinIndex(const std::vector<int>& arr);

    void plot(std::vector<int> data);
    template <typename T>
    T clip(const T& value, const T& minValue, const T& maxValue);
    void printPixelValue(const cv::Mat& image, int row, int col);
    void printGrayscaleImagePixelValue(const cv::Mat& grayscaleImage);
    std::pair<int, int> find2SmallestIndices(const std::vector<float>& array);
}

#endif
