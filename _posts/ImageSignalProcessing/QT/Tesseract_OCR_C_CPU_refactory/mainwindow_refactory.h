#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCoreApplication>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QImage>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


struct OcrData
{
    int positionX1, positionY1;                 // position1 (x, y) of the roibox
    int positionX2, positionY2;                 // position2 (x, y) of the roibox
    std::vector<QGraphicsRectItem*> rectItems;  // drawn rectangular object on window
    bool isRoiCandidated;                       // whether of candidated roi or not
    bool isRoiDetermined;                       // whether of determined roi or not
    bool isReadyToDraw;                         // whether of ready to draw or not
    cv::Rect rectInfo;
    cv::Rect recoveredRectInfo;
};


struct StencilData
{
    int selectedIdx;
    int updateTargetIdx;
    std::vector<std::vector<std::vector<int>>> positionA;   // dim1=x,y coordinate; dim2=position1,2 for making rec; dim3=4 corners of us image;
    std::vector<QGraphicsRectItem*> rectItems;              // drawn rectangular object on window
    std::vector<bool> isRoiCandidated;                      // whether of candidated roi or not
    bool isFourRoiReady;                                    // whether of determined roi or not
    bool isRoiDetermined;                                   // whether of determined roi or not
    bool isReadyToDraw;                                     // whether of ready to draw or not
    bool isMaskExist;                                       // whether of exist stencil mask or not
    bool isStencilApplied;                                  // whether of applying stencil mask on mainwindow or not
    std::vector<cv::Rect> rectInfo;
    std::vector<cv::Rect> recoveredRectInfo;
    int prevPositionX;
    int prevPositionY;
};


struct StructData
{
    std::string systemName;
    std::string probeName;
    std::string probeType;
    std::vector<std::string> pngFileNames;                      // Png file name list
    int currentDrawingIndex;                                    // drowing image index
    cv::Mat orgImg, scaledOrgImg, orgImgMask, scaledOrgImgMask; // orgImg = no-changed image, scaledOrgImg = scaled image
    float widthFactor;                                          // scale factor of width
    float heightFactor;                                         // scale factor of height
    Qt::MouseButton mouseButtonStatus;                          // mouse button status
    OcrData ocrdata;
    StencilData stencildata;
};


QT_BEGIN_NAMESPACE
namespace Ui
{
    class MainWindow;
}
QT_END_NAMESPACE


class MainWindow : public QMainWindow
{
    Q_OBJECT

    public:
        MainWindow(const QStringList& arguments, QWidget *parent = nullptr);
        ~MainWindow();

    protected:
        bool eventFilter(QObject* obj, QEvent* event) override;
        void onOcrRoiPushButtonClicked();
        void onConfirmStencilRoiPushButtonClicked();
        void onGenerateStencilRoiPushButtonClicked();
        void onProbeTypeChanged();
        void onApplyStencilStateChanged();
        void redrawWindow();
        void initOCR();
        std::string runOCR(const cv::Mat& inputImage);
        void handleLeftMouseButtonEvent(QMouseEvent* mouseEvent);
        void handleRightMouseButtonEvent(QMouseEvent* mouseEvent);
        void startOCRBoxSelection(const QPoint& startPos);
        void endOCRBoxSelection(const QPoint& endPos);
        void calculateOCRBox();
        void startStencilBoxSelectionOrRemoveStencilBox(const QPoint& pos);
        int findStencilBoxIndexAtPoint(const QPoint& point);
        int countActiveStencilBoxes();
        void startNewStencilBoxAt(const QPoint& startPos);
        void endStencilBoxSelection(const QPoint& startPos);




    private:
        Ui::MainWindow *ui;
        tesseract::TessBaseAPI* ocr_api;
        QGraphicsPixmapItem* pixmapItem;
        QGraphicsRectItem* rectItem;
        QTimer *updateTimer;
};

#endif // MAINWINDOW_H
