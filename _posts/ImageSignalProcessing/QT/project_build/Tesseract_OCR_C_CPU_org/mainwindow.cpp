#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "utils.h"
#include <QPixmap>
#include <QMouseEvent>
#include <QDebug>
#include <QtMath>
#include <cmath>
#include <stdexcept>
#include <chrono>


// set global variables
StructData structdata;
int fixedMainWindowWidth = 1024;  // width of main window
int fixedMainWindowHeight = 576;  // height of main window
int dim1Size = 2;                 // x, y respectively
int dim2Size = 2;                 // two edges of rubber band
int dim3Size = 4;                 // four rubber bands
int leastRoiBoxSize = 10;
int displayIntervalMs = 100;      // displayed frame interval


MainWindow::MainWindow(const QStringList& arguments, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , ocr_api(nullptr)
{
    ui->setupUi(this);

    /// init Tesseract OCR
    initOcr();

    /// determine argument
    std::string arg1;
    if (arguments.length() == 1)  // there is no argument
    {
        arg1 = "stream";
    }
    else
    {
        arg1 = arguments[1].toStdString();
    }

    /// load first frame
    cv::Mat cvImage;
    std::vector<std::string> pngFiles;
    if (structdata.toolModeName == "stream")
    {
        // Open the camera
        videoStream.open(0);
        // Check whether the camera opened successfully, then its frame is ok
        if (!videoStream.isOpened())
        {
            qDebug() << "    Error: Unable to open video stream...\n\n";
            videoStream.release();
            QCoreApplication::quit();
        }
        videoStream >> currentImgFrame;
        if (currentImgFrame.empty())
        {
            qDebug() << "    Error: Sudden end of video stream...\n\n";
            videoStream.release();
            QCoreApplication::quit();
        }
        cvImage = currentImgFrame;
    }
    else if (structdata.toolModeName == "emulate")
    {
        pngFiles = Utils::getAllPNGFiles(arg1);
        if (!pngFiles.empty())
        {
            std::cout << "List of PNG files:" << std::endl;
            for (const auto& filePath : pngFiles)
            {
                std::cout << "    " << filePath << std::endl;
            }
            std::cout << "\n" << std::endl;
        }
        else
        {
            qDebug() << "    Error: No PNG files found in the specified directory. Please check the path of png files directory...\n\n";
            QCoreApplication::quit();
        }
        cvImage = cv::imread(pngFiles[0]);
    }
    // cvImage -> qImage
    cv::Mat cvOrgImage = cvImage;
    int originalWidth = cvOrgImage.cols;
    int originalHeight = cvOrgImage.rows;
    float aspectRatio = (float)originalWidth / (float)originalHeight;
    int targetHeight = static_cast<int>((float)fixedMainWindowWidth / aspectRatio);
    cv::resize(cvImage, cvImage, cv::Size(fixedMainWindowWidth, targetHeight));
    cv::cvtColor(cvImage, cvImage, cv::COLOR_BGR2RGB);
    QImage qtImage(cvImage.data, cvImage.cols, cvImage.rows, cvImage.step, QImage::Format_RGB888);

    /// init scene using the first image frame on window
    QGraphicsScene* scene = new QGraphicsScene(this);
    ui->mainGraphicsView->setScene(scene);
    pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
    scene->addItem(pixmapItem);
    ocrRectItem = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem0 = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem1 = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem2 = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem3 = new QGraphicsRectItem(0,0,0,0);
    ui->mainGraphicsView->setRenderHint(QPainter::Antialiasing); // anti-alias
    ui->mainGraphicsView->setRenderHint(QPainter::SmoothPixmapTransform);
    ui->mainGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff); // remove scrollbar
    ui->mainGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    /// Event callbacks
    // event filter, mouse events callback
    ui->mainGraphicsView->viewport()->installEventFilter(this);
    // push button event callback
    connect(ui->ocrRoiPushButton, &QPushButton::clicked, this, &MainWindow::onOcrRoiPushButtonClicked);
    connect(ui->confirmStencilRoiPushButton, &QPushButton::clicked, this, &MainWindow::onConfirmStencilRoiPushButtonClicked);
    connect(ui->generateStencilRoiPushButton, &QPushButton::clicked, this, &MainWindow::onGenerateStencilRoiPushButtonClicked);
    connect(ui->probeTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onProbeTypeChanged);
    connect(ui->applyStencilCheckBox, &QCheckBox::stateChanged, this, &MainWindow::onApplyStencilStateChanged);
    // init timer for image update
    updateTimer = new QTimer(this);
    updateTimer->start(displayIntervalMs);
    connect(updateTimer, &QTimer::timeout, this, &MainWindow::redrawWindow);

    /// init of structdata
    // common (will be reported)
    structdata.systemName = "";
    structdata.probeName = "";
    structdata.probeType = "Convex";  // because checkbox's default state is "Convex" (Convex, Linear)
    // common
    structdata.pngFileNames = pngFiles;
    structdata.currentDrawingIndex = 0;
    structdata.orgImg = cvOrgImage;
    structdata.scaledOrgImg = cvImage;
    structdata.widthFactor = (float)cvOrgImage.cols / (float)cvImage.cols;
    structdata.heightFactor = (float)cvOrgImage.rows / (float)cvImage.rows;
    // ocrdata
    structdata.ocrdata.isRoiDetermined = false;
    structdata.ocrdata.isRoiCandidated = false;
    structdata.ocrdata.isReadyToDraw = false;
    // stencildata
    structdata.stencildata.selectedIdx = 0;
    structdata.stencildata.updateTargetIdx = 0;
    structdata.stencildata.isReadyToDraw = false;
    structdata.stencildata.positionA.resize(dim1Size, std::vector<std::vector<int>>(dim2Size, std::vector<int>(dim3Size)));
    structdata.stencildata.isRoiCandidated.resize(dim3Size);
    structdata.stencildata.rectInfo.resize(dim3Size);
    structdata.stencildata.recoveredRectInfo.resize(dim3Size);
    for (int dim3 = 0; dim3 < dim3Size; ++dim3) {
        structdata.stencildata.isRoiCandidated[dim3] = false;
        structdata.stencildata.rectInfo[dim3].x = 0;
        structdata.stencildata.rectInfo[dim3].y = 0;
        structdata.stencildata.rectInfo[dim3].width = 0;
        structdata.stencildata.rectInfo[dim3].height = 0;
        structdata.stencildata.recoveredRectInfo[dim3].x = 0;
        structdata.stencildata.recoveredRectInfo[dim3].y = 0;
        structdata.stencildata.recoveredRectInfo[dim3].width = 0;
        structdata.stencildata.recoveredRectInfo[dim3].height = 0;
        for (int dim1 = 0; dim1 < dim1Size; ++dim1) {
            for (int dim2 = 0; dim2 < dim2Size; ++dim2) {
                structdata.stencildata.positionA[dim1][dim2][dim3] = -1;
            }
        }
    }
    structdata.stencildata.isFourRoiReady = false;
    structdata.stencildata.isRoiDetermined = false;
    structdata.stencildata.isMaskExist = false;
    structdata.stencildata.isStencilApplied = true;  // because checkbox's default state is "checked"
}
MainWindow::~MainWindow()
{
    delete ui;
    delete ocr_api;
    delete scene;
    delete pixmapItem;
    delete ocrRectItem;
    delete stencilRectItem0;
    delete stencilRectItem1;
    delete stencilRectItem2;
    delete stencilRectItem3;
    delete updateTimer;
    videoStream.release();
    currentImgFrame.release();
}


/// OCR
void MainWindow::initOcr()
{
    ocr_api = new tesseract::TessBaseAPI();
    if (ocr_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY) != 0)
    {
        qDebug() << "    Error: Could not initialize Tesseract...\n\n";
    }
    ocr_api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr_api->SetVariable("tessedit_char_whitelist", "0123456789");
}


std::string MainWindow::runOcr(const cv::Mat& img)
{
    ocr_api->SetImage(img.data, img.cols, img.rows, 3, img.step);
    // ocr_api->SetSourceResolution(250); // updatedImg resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
    std::string outText = ocr_api->GetUTF8Text();
    outText.erase(std::remove(outText.begin(), outText.end(), '\n'), outText.end());
    try
    {
        float outValueF = std::stof(outText);
        outValueF /= 10.0f;
        std::stringstream ss;
        ss << std::fixed << std::setprecision(1) << outValueF;
        outText = ss.str();
    }
    catch (const std::exception& e)
    {
        outText = "NAN";
    }
    return outText;
}


/// Mouse interactions on graphicsview
bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);

    if (mouseEvent->button() == Qt::LeftButton)
    {
        structdata.mouseButtonStatus = Qt::LeftButton;
    }
    else if (mouseEvent->button() == Qt::RightButton)
    {
        structdata.mouseButtonStatus = Qt::RightButton;
    }

    /// mouse events for OCR
    if (obj == ui->mainGraphicsView->viewport() && structdata.mouseButtonStatus == Qt::LeftButton)
    {
        if (event->type() == QEvent::MouseButtonPress)
        {
            ui->depthDisplaylabel->setText(QString::fromStdString(""));

            structdata.ocrdata.positionX1 = mouseEvent->pos().x();
            structdata.ocrdata.positionY1 = mouseEvent->pos().y();
            structdata.ocrdata.isRoiCandidated = false;
            structdata.ocrdata.isRoiDetermined = false;
            structdata.ocrdata.isReadyToDraw = false;
        }
        else if ((event->type() == QEvent::MouseButtonRelease) || (event->type() == QEvent::MouseMove))
        {
            structdata.ocrdata.positionX2 = mouseEvent->pos().x();
            structdata.ocrdata.positionY2 = mouseEvent->pos().y();
            structdata.ocrdata.isRoiCandidated = true;
            structdata.ocrdata.isRoiDetermined = false;
            structdata.ocrdata.isReadyToDraw = true;
        }
        int x, y, width, height;
        x = structdata.ocrdata.positionX1;
        y = structdata.ocrdata.positionY1;
        width = structdata.ocrdata.positionX2 - structdata.ocrdata.positionX1;
        height = structdata.ocrdata.positionY2 - structdata.ocrdata.positionY1;
        if (structdata.ocrdata.positionX1 > structdata.ocrdata.positionX2)
        {
            x = structdata.ocrdata.positionX2;
            width = structdata.ocrdata.positionX1 - structdata.ocrdata.positionX2;
        }
        if (structdata.ocrdata.positionY1 > structdata.ocrdata.positionY2)
        {
            y = structdata.ocrdata.positionY2;
            height = structdata.ocrdata.positionY1 - structdata.ocrdata.positionY2;
        }
        structdata.ocrdata.rectInfo.x = x;
        structdata.ocrdata.rectInfo.y = y;
        structdata.ocrdata.rectInfo.width = width;
        structdata.ocrdata.rectInfo.height = height;
    }

    /// mouse events for stencil
    else if (obj == ui->mainGraphicsView->viewport() && structdata.mouseButtonStatus == Qt::RightButton)
    {
        int candidatePositionX = mouseEvent->pos().x();
        int candidatePositionY = mouseEvent->pos().y();
        if (event->type() == QEvent::MouseButtonPress)
        {
            // selection logic of update target
            bool reselectCandidate = false;
            for (int dim3 = 0; dim3 < dim3Size; ++dim3) {
                if (structdata.stencildata.isRoiCandidated[dim3])
                {
                    if (((((structdata.stencildata.positionA[0][0][dim3] <= candidatePositionX) && (candidatePositionX <= structdata.stencildata.positionA[0][1][dim3]))
                        ||((structdata.stencildata.positionA[0][1][dim3] <= candidatePositionX) && (candidatePositionX <= structdata.stencildata.positionA[0][0][dim3])))
                       &&(((structdata.stencildata.positionA[1][0][dim3] <= candidatePositionY) && (candidatePositionY <= structdata.stencildata.positionA[1][1][dim3]))
                        ||((structdata.stencildata.positionA[1][1][dim3] <= candidatePositionY) && (candidatePositionY <= structdata.stencildata.positionA[1][0][dim3])))))
                    {
                        structdata.stencildata.updateTargetIdx = dim3;
                        reselectCandidate = true;
                        break;
                    }
                }
            }
            if (!reselectCandidate)
            {
                structdata.stencildata.updateTargetIdx = Utils::findFirstFalse(structdata.stencildata.isRoiCandidated);
            }
            if (structdata.stencildata.updateTargetIdx != -1)
            {
                // get point
                structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] = candidatePositionX;
                structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] = candidatePositionY;
                structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = false;
                structdata.stencildata.isFourRoiReady = false;
                structdata.stencildata.isRoiDetermined = false;
                structdata.stencildata.isReadyToDraw = Utils::orArray(structdata.stencildata.isRoiCandidated);
            }
        }
        else if (event->type() == QEvent::MouseMove)
        {
            if (structdata.stencildata.updateTargetIdx != -1)
            {
                if (!structdata.stencildata.isFourRoiReady)
                {
                    // get point
                    structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] = candidatePositionX;
                    structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] = candidatePositionY;
                    structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = true;
                    structdata.stencildata.isRoiDetermined = false;
                    structdata.stencildata.isReadyToDraw = true;
                }
            }
        }
        else if (event->type() == QEvent::MouseButtonRelease)
        {
            if (structdata.stencildata.updateTargetIdx != -1)
            {
                if ((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] == candidatePositionX) ||
                        (structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] == candidatePositionY))     // remove roi with a simple click
                {
                    structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = false;
                }
                else
                {
                    // evaluate the roi is meaningful or not
                    bool isMeaningfulRoi = true;
                    if ((std::abs(structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] - candidatePositionX) < leastRoiBoxSize) ||
                            (std::abs(structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] - candidatePositionY) < leastRoiBoxSize))    // meaningful roi size
                    {
                        isMeaningfulRoi = false;
                    }
                    else
                    {
                        // check whether candidate roi includes another roi or not
                        int numEdgePoints = 4;
                        std::vector<std::vector<int>> tmpPositionA(dim1Size, std::vector<int>(numEdgePoints, 0));
                        for (int dim3 = 0; dim3 < dim3Size; ++dim3)
                        {
                            tmpPositionA[0][0] = structdata.stencildata.positionA[0][0][dim3];
                            tmpPositionA[1][0] = structdata.stencildata.positionA[1][0][dim3];
                            tmpPositionA[0][1] = structdata.stencildata.positionA[0][1][dim3];
                            tmpPositionA[1][1] = structdata.stencildata.positionA[1][1][dim3];
                            tmpPositionA[0][2] = structdata.stencildata.positionA[0][0][dim3];
                            tmpPositionA[1][2] = structdata.stencildata.positionA[1][1][dim3];
                            tmpPositionA[0][3] = structdata.stencildata.positionA[0][1][dim3];
                            tmpPositionA[1][3] = structdata.stencildata.positionA[1][0][dim3];
                            if (!(dim3 == structdata.stencildata.updateTargetIdx) && (structdata.stencildata.isRoiCandidated[dim3]))
                            {
                                if (
                                        (((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[0][0]) && (tmpPositionA[0][0] <= candidatePositionX))
                                         ||((candidatePositionX <= tmpPositionA[0][0]) && (tmpPositionA[0][0] <= structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx])))
                                        &&
                                        (((structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[1][0]) && (tmpPositionA[1][0] <= candidatePositionY))
                                         ||((candidatePositionY <= tmpPositionA[1][0]) && (tmpPositionA[1][0] <= structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx])))
                                        ||
                                        (((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[0][1]) && (tmpPositionA[0][1] <= candidatePositionX))
                                         ||((candidatePositionX <= tmpPositionA[0][1]) && (tmpPositionA[0][1] <= structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx])))
                                        &&
                                        (((structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[1][1]) && (tmpPositionA[1][1] <= candidatePositionY))
                                         ||((candidatePositionY <= tmpPositionA[1][1]) && (tmpPositionA[1][1] <= structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx])))
                                        ||
                                        (((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[0][2]) && (tmpPositionA[0][2] <= candidatePositionX))
                                         ||((candidatePositionX <= tmpPositionA[0][2]) && (tmpPositionA[0][2] <= structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx])))
                                        &&
                                        (((structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[1][2]) && (tmpPositionA[1][2] <= candidatePositionY))
                                         ||((candidatePositionY <= tmpPositionA[1][2]) && (tmpPositionA[1][2] <= structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx])))
                                        ||
                                        (((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[0][3]) && (tmpPositionA[0][3] <= candidatePositionX))
                                         ||((candidatePositionX <= tmpPositionA[0][3]) && (tmpPositionA[0][3] <= structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx])))
                                        &&
                                        (((structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] <= tmpPositionA[1][3]) && (tmpPositionA[1][3] <= candidatePositionY))
                                         ||((candidatePositionY <= tmpPositionA[1][3]) && (tmpPositionA[1][3] <= structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx])))
                                        )
                                {
                                    isMeaningfulRoi = false;
                                    break;
                                }
                            }
                        }
                    }
                    // get point
                    if (isMeaningfulRoi)
                    {
                        structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] = candidatePositionX;
                        structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] = candidatePositionY;
                        structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = true;
                    }
                    else
                    {
                        structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = false;
                    }
                }
                structdata.stencildata.isFourRoiReady = Utils::andArray(structdata.stencildata.isRoiCandidated);
                structdata.stencildata.isRoiDetermined = false;
                structdata.stencildata.isReadyToDraw = true;
            }
        }
        int x, y, width, height;
        x = structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx];
        y = structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx];
        width = structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx];
        height = structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx];
        if (structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] > structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx])
        {
            x = structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx];
            width = structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx];
        }
        if (structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] > structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx])
        {
            y = structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx];
            height = structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx];
        }
        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].x = x;
        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].y = y;
        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].width = width;
        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].height = height;
        structdata.stencildata.isMaskExist = false;
    }
    if (event->type() == QEvent::MouseButtonRelease)
    {
        structdata.mouseButtonStatus = Qt::NoButton;
    }
    return QMainWindow::eventFilter(obj, event);
}


/// pushbuttons
void MainWindow::onOcrRoiPushButtonClicked()
{
    if (structdata.ocrdata.isRoiCandidated)
    {
        structdata.ocrdata.recoveredRectInfo.x = (int)((float)structdata.ocrdata.rectInfo.x * structdata.widthFactor + 0.5f);
        structdata.ocrdata.recoveredRectInfo.y = (int)((float)structdata.ocrdata.rectInfo.y * structdata.heightFactor + 0.5f);
        structdata.ocrdata.recoveredRectInfo.width = (int)((float)structdata.ocrdata.rectInfo.width * structdata.widthFactor + 0.5f);
        structdata.ocrdata.recoveredRectInfo.height = (int)((float)structdata.ocrdata.rectInfo.height * structdata.heightFactor + 0.5f);
        structdata.ocrdata.isRoiDetermined = true;
    }
}


void MainWindow::onConfirmStencilRoiPushButtonClicked()
{
    if (Utils::andArray(structdata.stencildata.isRoiCandidated))
    {
        for (int dim3 = 0; dim3 < dim3Size; ++dim3)
        {
            structdata.stencildata.recoveredRectInfo[dim3].x = (int)((float)structdata.stencildata.rectInfo[dim3].x * structdata.widthFactor + 0.5f);
            structdata.stencildata.recoveredRectInfo[dim3].y = (int)((float)structdata.stencildata.rectInfo[dim3].y * structdata.heightFactor + 0.5f);
            structdata.stencildata.recoveredRectInfo[dim3].width = (int)((float)structdata.stencildata.rectInfo[dim3].width * structdata.widthFactor + 0.5f);
            structdata.stencildata.recoveredRectInfo[dim3].height = (int)((float)structdata.stencildata.rectInfo[dim3].height * structdata.heightFactor + 0.5f);
        }
        structdata.stencildata.isRoiDetermined = true;
    }
}


void MainWindow::onGenerateStencilRoiPushButtonClicked()
{

    if (structdata.stencildata.isRoiDetermined)
    {
        /// get four corner points from four subROIs
        // calc center of mass
        float centerOfMassX = 0.0f;
        float centerOfMassY = 0.0f;
        for (int dim3 = 0; dim3 < dim3Size; ++dim3)
        {
            centerOfMassX += (float)structdata.stencildata.rectInfo[dim3].x;
            centerOfMassY += (float)structdata.stencildata.rectInfo[dim3].y;
        }
        centerOfMassX /= (float)dim3Size;
        centerOfMassY /= (float)dim3Size;
        // classify four points
        int topLeftIdx = -1;
        int topRightIdx = -1;
        int bottomLeftIdx = -1;
        int bottomRightIdx = -1;
        for (int dim3 = 0; dim3 < dim3Size; ++dim3)
        {
            if (((float)structdata.stencildata.rectInfo[dim3].x < centerOfMassX) &&
                ((float)structdata.stencildata.rectInfo[dim3].y < centerOfMassY))
            {
                topLeftIdx = dim3;
            }
            else if (((float)structdata.stencildata.rectInfo[dim3].x > centerOfMassX) &&
                     ((float)structdata.stencildata.rectInfo[dim3].y < centerOfMassY))
            {
                topRightIdx = dim3;
            }
            else if (((float)structdata.stencildata.rectInfo[dim3].x < centerOfMassX) &&
                     ((float)structdata.stencildata.rectInfo[dim3].y > centerOfMassY))
            {
                bottomLeftIdx = dim3;
            }
            else if (((float)structdata.stencildata.rectInfo[dim3].x > centerOfMassX) &&
                     ((float)structdata.stencildata.rectInfo[dim3].y > centerOfMassY))
            {
                bottomRightIdx = dim3;
            }
        }
        // subROis
        cv::Mat topLeftImg = structdata.orgImg(structdata.stencildata.recoveredRectInfo[topLeftIdx]);
        cv::Mat topRightImg = structdata.orgImg(structdata.stencildata.recoveredRectInfo[topRightIdx]);
        cv::Mat bottomLeftImg = structdata.orgImg(structdata.stencildata.recoveredRectInfo[bottomLeftIdx]);
        cv::Mat bottomRightImg = structdata.orgImg(structdata.stencildata.recoveredRectInfo[bottomRightIdx]);
        // profiles
        std::pair<std::vector<int>, std::vector<int>> profileData = Utils::calculateProfileData(topLeftImg);
        std::vector<int> topLeftImgWidthProfileData = profileData.first;
        std::vector<int> topLeftImgHeightProfileData = profileData.second;
        profileData = Utils::calculateProfileData(topRightImg);
        std::vector<int> topRightImgWidthProfileData = profileData.first;
        std::vector<int> topRightImgHeightProfileData = profileData.second;
        profileData = Utils::calculateProfileData(bottomLeftImg);
        std::vector<int> bottomLeftImgWidthProfileData = profileData.first;
        std::vector<int> bottomLeftImgHeightProfileData = profileData.second;
        profileData = Utils::calculateProfileData(bottomRightImg);
        std::vector<int> bottomRightImgWidthProfileData = profileData.first;
        std::vector<int> bottomRightImgHeightProfileData = profileData.second;
        int topLeftImgX, topLeftImgY, topRightImgX, topRightImgY, bottomLeftImgX, bottomLeftImgY, bottomRightImgX, bottomRightImgY;
        if (structdata.probeType == "Convex")
        {
            // find four corner points
            topLeftImgX = Utils::findFirstMaxIndex(topLeftImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[topLeftIdx].x;
            topLeftImgY = Utils::findFirstNonZeroIndex(topLeftImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[topLeftIdx].y;
            topRightImgX = Utils::findLastMaxIndex(topRightImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[topRightIdx].x;
            topRightImgY = Utils::findFirstNonZeroIndex(topRightImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[topRightIdx].y;
            bottomLeftImgX = Utils::findFirstNonZeroIndex(bottomLeftImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[bottomLeftIdx].x;
            bottomLeftImgY = Utils::findLastMaxIndex(bottomLeftImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[bottomLeftIdx].y;
            bottomRightImgX = Utils::findLastNonZeroIndex(bottomRightImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[bottomRightIdx].x;
            bottomRightImgY = Utils::findLastMaxIndex(bottomRightImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[bottomRightIdx].y;
        }
        else if (structdata.probeType == "Linear")
        {
            // find four corner points
            topLeftImgX = Utils::findFirstNonZeroIndex(topLeftImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[topLeftIdx].x;
            topLeftImgY = Utils::findFirstNonZeroIndex(topLeftImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[topLeftIdx].y;
            topRightImgX = Utils::findLastNonZeroIndex(topRightImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[topRightIdx].x;
            topRightImgY = Utils::findFirstNonZeroIndex(topRightImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[topRightIdx].y;
            bottomLeftImgX = Utils::findFirstNonZeroIndex(bottomLeftImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[bottomLeftIdx].x;
            bottomLeftImgY = Utils::findLastNonZeroIndex(bottomLeftImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[bottomLeftIdx].y;
            bottomRightImgX = Utils::findLastNonZeroIndex(bottomRightImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[bottomRightIdx].x;
            bottomRightImgY = Utils::findLastNonZeroIndex(bottomRightImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[bottomRightIdx].y;
        }
        /// making stencil
        QPoint topLeftPoint(topLeftImgX, topLeftImgY);
        QPoint bottomLeftPoint(bottomLeftImgX, bottomLeftImgY);
        QPoint topRightPoint(topRightImgX, topRightImgY);
        QPoint bottomRightPoint(bottomRightImgX, bottomRightImgY);
        QPainterPath stencilPath;
        if (structdata.probeType == "Convex")
        {
            // calc apex pont
            QPointF apexPoint;
            apexPoint = Utils::intersectionPoint(topLeftPoint, bottomLeftPoint, topRightPoint, bottomRightPoint);
            qDebug() << apexPoint.x() << "x " << apexPoint.y() << " y";
            // draw FOV
            QPointF topLeftPointF = QPointF(topLeftPoint);
            QPointF bottomLeftPointF = QPointF(bottomLeftPoint);
            QPointF topRightPointF = QPointF(topRightPoint);
            QPointF bottomRightPointF = QPointF(bottomRightPoint);
            qreal innerRadius = qCeil(QLineF(apexPoint, topLeftPointF).length());
            qreal outerRadius = qFloor(QLineF(apexPoint, bottomLeftPointF).length());
            double leftSlopeAngleRad = std::atan2(bottomLeftPointF.y() - apexPoint.y(), bottomLeftPointF.x() - apexPoint.x());
            double rightSlopeAngleRad = std::atan2(bottomRightPointF.y() - apexPoint.y(), bottomRightPointF.x() - apexPoint.x());
            double leftSlopeAngleDeg = leftSlopeAngleRad * 180 / M_PI;
            double rightSlopeAngleDeg = rightSlopeAngleRad * 180 / M_PI;
            double spanAngleDeg = leftSlopeAngleDeg - rightSlopeAngleDeg;
            stencilPath.moveTo(topLeftPointF);
            stencilPath.lineTo(bottomLeftPointF);
            stencilPath.arcTo(apexPoint.x()-outerRadius, apexPoint.y()-outerRadius, 2 * outerRadius, 2 * outerRadius, -leftSlopeAngleDeg, spanAngleDeg);
            stencilPath.lineTo(topRightPointF);
            stencilPath.arcTo(apexPoint.x()-innerRadius, apexPoint.y()-innerRadius, 2 * innerRadius, 2 * innerRadius, -rightSlopeAngleDeg, -spanAngleDeg);
            stencilPath.closeSubpath();
        }
        else if (structdata.probeType == "Linear")
        {
            // draw FOV
            QPointF topLeftPointF = QPointF(topLeftPoint);
            QPointF bottomLeftPointF = QPointF(bottomLeftPoint);
            QPointF topRightPointF = QPointF(topRightPoint);
            QPointF bottomRightPointF = QPointF(bottomRightPoint);
            stencilPath.moveTo(topLeftPointF);
            stencilPath.lineTo(bottomLeftPointF);
            stencilPath.lineTo(bottomRightPointF);
            stencilPath.lineTo(topRightPointF);
            stencilPath.lineTo(topLeftPointF);
            stencilPath.closeSubpath();
        }
        // get masks
        QImage qtOrgImgMask(structdata.orgImg.cols, structdata.orgImg.rows, QImage::Format_ARGB32);
        qtOrgImgMask.fill(Qt::black);
        QPainter maskPainter(&qtOrgImgMask);
        QBrush brush(Qt::white);
        maskPainter.setBrush(brush);
        maskPainter.fillPath(stencilPath, brush);
        cv::Mat cvScaledImgMask;
        cv::Mat cvOrgImgMask = cv::Mat(qtOrgImgMask.height(), qtOrgImgMask.width(), CV_8UC4, const_cast<uchar*>(qtOrgImgMask.bits()), qtOrgImgMask.bytesPerLine());
        cv::resize(cvOrgImgMask, cvScaledImgMask, cv::Size(structdata.scaledOrgImg.cols, structdata.scaledOrgImg.rows));
        cv::threshold(cvOrgImgMask, cvOrgImgMask, 127, 1, cv::THRESH_BINARY);
        cv::threshold(cvScaledImgMask, cvScaledImgMask, 127, 1, cv::THRESH_BINARY);
        cv::cvtColor(cvOrgImgMask, cvOrgImgMask, cv::COLOR_BGR2RGB);
        cv::cvtColor(cvScaledImgMask, cvScaledImgMask, cv::COLOR_BGR2RGB);
        structdata.orgImgMask = cvOrgImgMask;
        structdata.scaledOrgImgMask = cvScaledImgMask;
        structdata.stencildata.isMaskExist = true;
    }
}


/// comboboxs
void MainWindow::onProbeTypeChanged()
{
    structdata.probeType = ui->probeTypeComboBox->currentText().toStdString();
    onGenerateStencilRoiPushButtonClicked();
}


/// checkboxs
void MainWindow::onApplyStencilStateChanged()
{
    if (ui->applyStencilCheckBox->isChecked())
    {
        structdata.stencildata.isStencilApplied = true;
    }
    else
    {
        structdata.stencildata.isStencilApplied = false;
    }

}


/// recalc of drawing on mainwindow
void MainWindow::redrawWindow()
{
//    auto start_time = std::chrono::high_resolution_clock::now();

    // check items for debug
    QList<QGraphicsItem*> allItems = ui->mainGraphicsView->scene()->items();
    qDebug() << "all graphic items \n" << allItems;

    /// draw US image
    // update drawing index
    int nextDrawingIndex = (structdata.currentDrawingIndex + 1) % ((int)structdata.pngFileNames.size());
    // load new image, then draw
    cv::Mat cvImage;
    if (structdata.toolModeName == "stream")
    {
        videoStream >> currentImgFrame;
        if (currentImgFrame.empty())
        {
            qDebug() << "    Error: Sudden end of video stream...\n\n";
            videoStream.release();
            QCoreApplication::quit();
        }
        cvImage = currentImgFrame;
    }
    else if (structdata.toolModeName == "emulate")
    {
        cvImage = cv::imread(structdata.pngFileNames[nextDrawingIndex]);
    }
    cv::Mat cvOrgImage = cvImage;
    cv::resize(cvImage, cvImage, cv::Size(structdata.scaledOrgImg.cols, structdata.scaledOrgImg.rows));
    cv::cvtColor(cvImage, cvImage, cv::COLOR_BGR2RGB);
    if (pixmapItem != nullptr)
    {
        delete pixmapItem;
        pixmapItem = nullptr;
    }
    if ((structdata.stencildata.isMaskExist) && (structdata.stencildata.isStencilApplied))
    {
        cv::Mat maskedCvImage;
        cv::multiply(cvImage, structdata.scaledOrgImgMask, maskedCvImage);
        QImage qtImage(maskedCvImage.data, maskedCvImage.cols, maskedCvImage.rows, maskedCvImage.step, QImage::Format_RGB888);
        pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
    }
    else
    {
        QImage qtImage(cvImage.data, cvImage.cols, cvImage.rows, cvImage.step, QImage::Format_RGB888);
        pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
//        pixmapItem = QPixmap::fromImage(QImage(reinterpret_cast<uchar const*>(color_image -> imagedata), color_image -> width, color_image -> height. QImage::Format_ARGB32));
    }
    ui->mainGraphicsView->scene()->addItem(pixmapItem);
    structdata.orgImg = cvOrgImage;
    structdata.scaledOrgImg = cvImage;
    structdata.currentDrawingIndex = nextDrawingIndex;

    /// draw rubber bands
    if (!(structdata.stencildata.isMaskExist && structdata.stencildata.isStencilApplied))
    {
        // draw rect rubber band for ocr
        if (structdata.ocrdata.isReadyToDraw)
        {
            if (ocrRectItem != nullptr)
            {
                delete ocrRectItem;
                ocrRectItem = nullptr;
            }
            ocrRectItem = new QGraphicsRectItem(structdata.ocrdata.positionX1,
                                             structdata.ocrdata.positionY1,
                                             structdata.ocrdata.positionX2-structdata.ocrdata.positionX1,
                                             structdata.ocrdata.positionY2-structdata.ocrdata.positionY1);
            if (!structdata.ocrdata.isRoiDetermined)
            {
                ocrRectItem->setPen(QPen(Qt::red, 1));
            }
            else
            {
                ocrRectItem->setPen(QPen(Qt::green, 2));
            }
            ui->mainGraphicsView->scene()->addItem(ocrRectItem);
        }
        // draw rect rubber band for stencil
        if (structdata.stencildata.isReadyToDraw)
        {
            int dim3 = 0;
            if (structdata.stencildata.isRoiCandidated[dim3])
            {
                if (stencilRectItem0 != nullptr)
                {
                    delete stencilRectItem0;
                    stencilRectItem0 = nullptr;
                }
                stencilRectItem0 = new QGraphicsRectItem(structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][0][dim3],
                                                         structdata.stencildata.positionA[0][1][dim3]-structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][1][dim3]-structdata.stencildata.positionA[1][0][dim3]);
                if (!structdata.stencildata.isRoiDetermined)
                {
                    stencilRectItem0->setPen(QPen(Qt::yellow, 1));
                }
                else
                {
                    stencilRectItem0->setPen(QPen(Qt::blue, 2));
                }
                ui->mainGraphicsView->scene()->addItem(stencilRectItem0);
            }

            dim3 = 1;
            if (structdata.stencildata.isRoiCandidated[dim3])
            {
                if (stencilRectItem1 != nullptr)
                {
                    delete stencilRectItem1;
                    stencilRectItem1 = nullptr;
                }
                stencilRectItem1 = new QGraphicsRectItem(structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][0][dim3],
                                                         structdata.stencildata.positionA[0][1][dim3]-structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][1][dim3]-structdata.stencildata.positionA[1][0][dim3]);
                if (!structdata.stencildata.isRoiDetermined)
                {
                    stencilRectItem1->setPen(QPen(Qt::yellow, 1));
                }
                else
                {
                    stencilRectItem1->setPen(QPen(Qt::blue, 2));
                }
                ui->mainGraphicsView->scene()->addItem(stencilRectItem1);
            }

            dim3 = 2;
            if (structdata.stencildata.isRoiCandidated[dim3])
            {
                if (stencilRectItem2 != nullptr)
                {
                    delete stencilRectItem2;
                    stencilRectItem2 = nullptr;
                }
                stencilRectItem2 = new QGraphicsRectItem(structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][0][dim3],
                                                         structdata.stencildata.positionA[0][1][dim3]-structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][1][dim3]-structdata.stencildata.positionA[1][0][dim3]);
                if (!structdata.stencildata.isRoiDetermined)
                {
                    stencilRectItem2->setPen(QPen(Qt::yellow, 1));
                }
                else
                {
                    stencilRectItem2->setPen(QPen(Qt::blue, 2));
                }
                ui->mainGraphicsView->scene()->addItem(stencilRectItem2);
            }

            dim3 = 3;
            if (structdata.stencildata.isRoiCandidated[dim3])
            {
                if (stencilRectItem3 != nullptr)
                {
                    delete stencilRectItem3;
                    stencilRectItem3 = nullptr;
                }
                stencilRectItem3 = new QGraphicsRectItem(structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][0][dim3],
                                                         structdata.stencildata.positionA[0][1][dim3]-structdata.stencildata.positionA[0][0][dim3],
                                                         structdata.stencildata.positionA[1][1][dim3]-structdata.stencildata.positionA[1][0][dim3]);
                if (!structdata.stencildata.isRoiDetermined)
                {
                    stencilRectItem3->setPen(QPen(Qt::yellow, 1));
                }
                else
                {
                    stencilRectItem3->setPen(QPen(Qt::blue, 2));
                }
                ui->mainGraphicsView->scene()->addItem(stencilRectItem3);
            }
        }
    }

    /// OCR inference
    if (structdata.ocrdata.isRoiDetermined)
    {
        // get subROI
        cv::Mat subRoi_img = structdata.orgImg(structdata.ocrdata.recoveredRectInfo);
        // conduct Tesseract OCR
        auto start_time = std::chrono::high_resolution_clock::now();
        std::string outText = runOcr(subRoi_img);
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        std::cout << "OCR elapsed time: " << duration << " ms" << std::endl;
        // write the result at textbox
        ui->depthDisplaylabel->setText(QString::fromStdString(outText));
        ui->depthDisplaylabel->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);
    }
//    auto end_time = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
//    std::cout << "Redraw elapsed time: " << duration << " ms" << std::endl;
}



