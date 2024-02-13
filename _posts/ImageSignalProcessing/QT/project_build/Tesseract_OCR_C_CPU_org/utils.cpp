#include "utils.h"
#include <filesystem>
#include <iostream>
#include <algorithm>

namespace fs = std::filesystem;

std::vector<std::string> Utils::getAllPNGFiles(const std::string& directoryPath)
{
    std::vector<std::string> pngFiles;
    try
    {
        for (const auto& entry : fs::directory_iterator(directoryPath))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".png")
            {
                pngFiles.push_back(entry.path().string());
            }
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
    }
    std::sort(pngFiles.begin(), pngFiles.end());
    return pngFiles;
}


bool Utils::andArray(const std::vector<bool> array)
{
    int size = static_cast<int>(array.size());
    for (int i = 0; i < size; ++i) {
        if (!array[i]) {
            return false;
        }
    }
    return true;
}


bool Utils::orArray(const std::vector<bool> array)
{
    int size = static_cast<int>(array.size());
    for (int i = 0; i < size; ++i) {
        if (array[i]) {
            return true;
        }
    }
    return false;
}


int Utils::findFirstFalse(const std::vector<bool> array)
{
    int size = static_cast<int>(array.size());
    for (int i = 0; i < size; ++i) {
        if (!array[i]) {
            return i;
        }
    }
    return -1;
}


int Utils::signInt(const int value)
{
    return static_cast<int>((0 < value) - (value < 0));
}


std::pair<std::vector<int>, std::vector<int>> Utils::calculateProfileData(const cv::Mat& inputImage)
{
    double minBrightnessForThreshold;
    cv::minMaxLoc(inputImage, &minBrightnessForThreshold, NULL, NULL, NULL);
    cv::Mat binaryImage;
    cv::threshold(inputImage, binaryImage, static_cast<int>(minBrightnessForThreshold), 1, cv::THRESH_BINARY);
    cv::Mat widthProfile, heightProfile;
    cv::reduce(binaryImage, widthProfile, 0, cv::REDUCE_SUM, CV_32S);
    cv::reduce(binaryImage, heightProfile, 1, cv::REDUCE_SUM, CV_32S);
    std::vector<int> widthProfileData(widthProfile.begin<int>(), widthProfile.end<int>());
    std::vector<int> heightProfileData(heightProfile.begin<int>(), heightProfile.end<int>());
    return std::make_pair(widthProfileData, heightProfileData);
}


int Utils::findFirstNonZeroIndex(const std::vector<int>& arr)
{
    auto firstNonZeroIter = std::find_if(arr.begin(), arr.end(), [](int val) { return val != 0; });
    return (firstNonZeroIter != arr.end()) ? std::distance(arr.begin(), firstNonZeroIter) : -1;
}


int Utils::findLastNonZeroIndex(const std::vector<int>& arr)
{
    auto lastNonZeroIter = std::find_if(arr.rbegin(), arr.rend(), [](int val) { return val != 0; });
    return (lastNonZeroIter != arr.rend()) ? arr.size() - 1 - std::distance(arr.rbegin(), lastNonZeroIter) : -1;
}


int Utils::findFirstMaxIndex(const std::vector<int>& arr)
{
    auto firstMaxIter = std::max_element(arr.begin(), arr.end());
    return (firstMaxIter != arr.end()) ? std::distance(arr.begin(), firstMaxIter) : -1;
}


int Utils::findLastMaxIndex(const std::vector<int>& arr)
{
    auto lastMaxIter = std::max_element(arr.rbegin(), arr.rend());
    return (lastMaxIter != arr.rend()) ? arr.size() - 1 - std::distance(arr.rbegin(), lastMaxIter) : -1;
}


QPointF Utils::intersectionPoint(const QPoint topLeftPoint, const QPoint bottomLeftPoint, const QPoint topRightPoint, const QPoint bottomRightPoint)
{
    qreal x1 = (qreal)topLeftPoint.x();
    qreal y1 = (qreal)topLeftPoint.y();
    qreal x2 = (qreal)bottomLeftPoint.x();
    qreal y2 = (qreal)bottomLeftPoint.y();
    qreal x3 = (qreal)topRightPoint.x();
    qreal y3 = (qreal)topRightPoint.y();
    qreal x4 = (qreal)bottomRightPoint.x();
    qreal y4 = (qreal)bottomRightPoint.y();
    qreal denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4));
    if (qFuzzyIsNull(denominator)) // no intersection point
    {
        return QPointF();
    }
    qreal numeratorX = ((x1 * y2 - y1 * x2) * (x3 - x4)) - ((x1 - x2) * (x3 * y4 - y3 * x4));
    qreal numeratorY = ((x1 * y2 - y1 * x2) * (y3 - y4)) - ((y1 - y2) * (x3 * y4 - y3 * x4));
    qreal x = numeratorX / denominator;
    qreal y = numeratorY / denominator;
    return QPointF(x, y);
}





void Utils::plot(std::vector<int> data)
{
    cv::Mat graph(100, 200, CV_8UC3, cv::Scalar(255, 255, 255));
    cv::Scalar line_color(0, 0, 255);
    for (int i = 1; i < data.size(); ++i) {
        cv::line(graph,
                 cv::Point((i - 1) * 4, 100 - data[i - 1]),
                 cv::Point(i * 4, 100 - data[i]),
                 line_color,
                 2);
    }
    cv::imshow("tmp plot", graph);
    cv::waitKey(0);
}


template <typename T>
T Utils::clip(const T& value, const T& minValue, const T& maxValue)
{
    return std::min(std::max(value, minValue), maxValue);
}
