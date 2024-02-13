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
    std::pair<std::vector<int>, std::vector<int>> calculateProfileData(const cv::Mat& inputImage);
    int findFirstNonZeroIndex(const std::vector<int>& arr);
    int findLastNonZeroIndex(const std::vector<int>& arr);
    int findFirstMaxIndex(const std::vector<int>& arr);
    int findLastMaxIndex(const std::vector<int>& arr);
    QPointF intersectionPoint(const QPoint topLeftPoint, const QPoint bottomLeftPoint, const QPoint topRightPoint, const QPoint bottomRightPoint);

    void plot(std::vector<int> data);
    template <typename T>
    T clip(const T& value, const T& minValue, const T& maxValue);
}

#endif
