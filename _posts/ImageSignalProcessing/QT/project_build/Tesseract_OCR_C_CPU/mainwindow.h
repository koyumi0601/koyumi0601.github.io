#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QCoreApplication>
#include <QGraphicsPixmapItem>
#include <QGraphicsRectItem>
#include <QTableWidget>
#include <QImage>
#include <QTimer>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


struct OcrData
{
    int positionX1, positionY1;                 // position1 (x, y) of the roibox
    int positionX2, positionY2;                 // position2 (x, y) of the roibox
    bool isRoiCandidated;                       // whether of candidated roi or not
    bool isRoiDetermined;                       // whether of determined roi or not
    bool isReadyToDraw;                         // whether of ready to draw or not
    cv::Rect rectInfo;
    cv::Rect recoveredRectInfo;
    int numSteadyFrames;
    cv::Mat prevSubRoiImg;
    std::string depth;
};


struct StencilData
{
    int updateTargetIdx;
    std::vector<std::vector<std::vector<int>>> positionA;   // dim1=x,y coordinate; dim2=position1,2 for making rec; dim3=4 corners of us image;
    std::vector<bool> isRoiCandidated;                      // whether of candidated roi or not
    bool isFourRoiReady;                                    // whether of determined roi or not
    bool isRoiDetermined;                                   // whether of determined roi or not
    bool isReadyToDraw;                                     // whether of ready to draw or not
    bool isMaskExist;                                       // whether of exist stencil mask or not
    bool isStencilApplied;                                  // whether of applying stencil mask on mainwindow or not
    int existingBoxIndex;
    std::vector<cv::Rect> rectInfo;
    std::vector<cv::Rect> recoveredRectInfo;

    //output
    float topLeftX;
    float topLeftY;
    float topRightX;
    float topRightY;
    float midLeftX;
    float midLeftY;
    float midRightX;
    float midRightY;
    float bottomLeftX;
    float bottomLeftY;
    float bottomRightX;
    float bottomRightY;
    cv::Rect boundRect;
};


struct StructData
{
    std::string systemName;
    std::string probeName;
    std::string probeType;
    std::string toolModeName;                                   // "stream", "emulate"
    std::vector<std::string> pngFileNames;                      // Png file name list
    int currentDrawingIndex;                                    // drowing image index
    cv::Mat orgImg, scaledOrgImg, orgImgMask, scaledOrgImgMask; // orgImg = no-changed image, scaledOrgImg = scaled image
    float widthFactor;                                          // scale factor of width
    float heightFactor;                                         // scale factor of height
    Qt::MouseButton mouseButtonStatus;                          // mouse button status
    OcrData ocrdata;
    StencilData stencildata;
    //QPoint scaledMaskTopLeftPointInt, scaledMaskBottomLeftPointInt, scaledMaskTopRightPointInt, scaledMaskBottomRightPointInt;
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
        // mainwindow
        std::tuple<std::string, std::vector<std::string>, cv::Mat> getImageAndOthers(QStringList args);
        cv::Mat openStreamAndGetFrame();
        std::vector<std::string> grapEmulationPngList(const std::string& directoryPath);
        cv::Mat resizeCvImage(cv::Mat img);
        void initMainGraphicsViewAndScene(cv::Mat img);
        void connectCallbacks();
        void initStructData(std::string toolModeName, std::vector<std::string> pngFiles, cv::Mat cvOrgImage, cv::Mat cvScalsedImage);
        void initPrivateVariables();
        // ocr
        void initOcr();
        std::string runOcr(const cv::Mat& img);

        // redraw
        void redrawWindow();
        cv::Mat reloadCvImage();
        void redrawScene(const cv::Mat& image);
        void drawOcrRectRubberBand();
        void drawStencilRectRubberBands();
        //void drawCrossLines(const QList<QPoint>& points);

        // eventFilter
        bool eventFilter(QObject* obj, QEvent* event) override;
        void leftMouseButtonEvent(QMouseEvent* mouseEvent);
        void startSelectOcrRubberBand(const QPoint& startPos);
        void endSelectOcrRubberBand(const QPoint& endPos);
        void updateOcrRubberBandInfo();
        void rightMouseButtonEvent_NEW(QMouseEvent* mouseEvent);
        void getMouseMoveStartPos(const QPoint& pos);
        void generateStencilBox(const QPoint& pos);
        void moveStencilBox(const QPoint& pos);
        void rightMouseButtonEvent(QMouseEvent* mouseEvent);
        void startSelectStencilRubberBand(const QPoint& pos);
        int searchAlreadyExistStencilRubberBandIndex(const QPoint& point);
        void updateSelectStencilRubberBand(const QPoint& pos);
        void endSelectStencilRubberBand(const QPoint& pos);
        bool searchOverlapedRubberBand(const int x1, const int y1, const int x2, const int y2);

        // pushbutton
        void onOcrRoiPushButtonClicked();
        void onConfirmStencilRoiPushButtonClicked();
        void onGenerateStencilRoiPushButtonClicked();
        std::vector<int> findCornerPoints(const std::vector<cv::Rect>& rectInfo);
        void onPushItemPushButtonClicked();
        void changeRowCellsColor(QTableWidget* tableWidget, int row, const QColor& color);
        void changeRowWordsColor(QTableWidget* tableWidget, int row, const QColor& color);
        void onDeleteItemPushButtonClicked();

        // combobox
        void onProbeTypeChanged();

        // checkbox
        void onApplyStencilStateChanged();

        // testeditbox
        void onSystemNameTextEdited();
        void onProbeNameTextEdited();
        void updateTableCell(QTableWidget* tableWidget, int row, int column, const QString& newText);
        void insertTableCell(QTableWidget* tableWidget, int row, int column, const QString& newText);

        // tablewidget interaction
        void onTableWidgetListClicked(int row, int column);

        // debuger
        void printStencilData(bool debugMode);

    private:
        // gui
        Ui::MainWindow *ui;
        tesseract::TessBaseAPI* ocr_api;
        QGraphicsScene* scene;
        QGraphicsPixmapItem* pixmapItem;
        QGraphicsRectItem* ocrRectItem;
        std::vector<QGraphicsRectItem*> stencilRectItem;
        QTimer* updateTimer;
        // stream
        cv::VideoCapture videoStream;
        cv::Mat currentImgFrame;
        // tool state
        std::string redrawingState;
        std::string selectedTableIndex;
        // review/live data
        StructData liveData;
        std::unordered_map<std::string, StructData> reviewData;
        int prevMouseMovePosX, prevMouseMovePosY;
        bool isMouseMoveState;
};

#endif // MAINWINDOW_H
