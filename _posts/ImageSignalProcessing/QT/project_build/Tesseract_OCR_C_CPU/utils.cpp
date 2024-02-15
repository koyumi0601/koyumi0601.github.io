#include "utils.h"
#include <filesystem>
#include <iostream>
#include <algorithm>
#include <QDebug>
#include <QPointF>
#include <cmath>

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


std::tuple<std::vector<int>, std::vector<int>, int> Utils::calculateProfileData(const cv::Mat& inputImage, int minBrightnessForThreshold)
{
    int minimumThreshold = 16;
    if (minBrightnessForThreshold == -1)
    {
        double minBrightness;
        cv::minMaxLoc(inputImage, &minBrightness, NULL, NULL, NULL);
        minBrightnessForThreshold = static_cast<int>(minBrightness);
    }
    if (minBrightnessForThreshold == 0)
    {
        minBrightnessForThreshold = minimumThreshold;
    }
    cv::Mat binaryImage;
    cv::threshold(inputImage, binaryImage, minBrightnessForThreshold, 1, cv::THRESH_BINARY);
    cv::Mat widthProfile, heightProfile;
    cv::reduce(binaryImage, widthProfile, 0, cv::REDUCE_SUM, CV_32S);
    cv::reduce(binaryImage, heightProfile, 1, cv::REDUCE_SUM, CV_32S);
    std::vector<int> widthProfileData(widthProfile.begin<int>(), widthProfile.end<int>());
    std::vector<int> heightProfileData(heightProfile.begin<int>(), heightProfile.end<int>());
    return std::make_tuple(widthProfileData, heightProfileData, minBrightnessForThreshold);
}


int Utils::findFirstNonZeroIndex(const std::vector<int>& arr)
{
    auto firstNonZeroIter = std::find_if(arr.begin(), arr.end(), [](int val) { return val != 0; });
    return (firstNonZeroIter != arr.end()) ? std::distance(arr.begin(), firstNonZeroIter) : -1;
}


int Utils::findFirstZeroIndex(const std::vector<int>& arr)
{
    auto firstZeroIter = std::find_if(arr.begin(), arr.end(), [](int val) { return val == 0; });
    return (firstZeroIter != arr.end()) ? std::distance(arr.begin(), firstZeroIter) : -1;
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

int Utils::findFirstNonMinIndex(const std::vector<int>& arr) {
    auto minValue = *std::min_element(arr.begin(), arr.end());
    auto firstNonMinIter = std::find_if(arr.begin(), arr.end(), [minValue](int val) { return val != minValue; });
    return (firstNonMinIter != arr.end()) ? std::distance(arr.begin(), firstNonMinIter) : -1;
}

int Utils::findLastNonMinIndex(const std::vector<int>& arr) {
    auto minValue = *std::min_element(arr.begin(), arr.end());
    auto lastNonMinIter = std::find_if(arr.rbegin(), arr.rend(), [minValue](int val) { return val != minValue; });
    return (lastNonMinIter != arr.rend()) ? arr.size() - 1 - std::distance(arr.rbegin(), lastNonMinIter) : -1;
}


QPointF Utils::intersectionPointof2Lines(const QPoint topLeftPoint, const QPoint bottomLeftPoint, const QPoint topRightPoint, const QPoint bottomRightPoint)
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


qreal Utils::intersectionPointOfLineAndCircle(qreal pos_x, const QPointF& apexPoint, qreal radius)
{
    return std::sqrt((radius * radius) - (pos_x - apexPoint.x())*(pos_x - apexPoint.x())) + + apexPoint.y();
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

void Utils::printPixelValue(const cv::Mat& image, int row, int col)
{
    cv::Vec3b pixelValue = image.at<cv::Vec3b>(row, col);
    int blue = pixelValue[0];
    int green = pixelValue[1];
    int red = pixelValue[2];
    qDebug() << "BGR: (" << blue << " " << green << " " << red << ")";
}

void Utils::printGrayscaleImagePixelValue(const cv::Mat& grayscaleImage)
{
    for (int row = 0; row < grayscaleImage.rows; ++row) {
        for (int col = 0; col < grayscaleImage.cols; ++col) {
            uchar pixelValue = grayscaleImage.at<uchar>(row, col);
            std::cout << "Pixel Value at (" << row << ", " << col << "): " << static_cast<int>(pixelValue) << std::endl;
        }
    }
}


std::pair<int, int> Utils::find2SmallestIndices(const std::vector<float>& array)
{
    int smallestIndex = -1;
    int secondSmallestIndex = -1;
    int minVal = std::numeric_limits<int>::max();
    for (int i = 0; i < array.size(); ++i) {
        if (array[i] < minVal) {
            secondSmallestIndex = smallestIndex;
            smallestIndex = i;
            minVal = array[i];
        } else if (array[i] < array[secondSmallestIndex] || secondSmallestIndex == -1) {
            secondSmallestIndex = i;
        }
    }
    return {smallestIndex, secondSmallestIndex};
}

