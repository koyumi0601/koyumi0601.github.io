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
#include <QKeyEvent>
#include <QMessageBox>
#include <QColor>


// set global variables
StructData structdata;
// display
int fixedMainWindowWidth = 1024;  // width of main window
int fixedMainWindowHeight = 576;  // height of main window
int displayIntervalMsec = 100;    // displayed frame interval
// ocr
int steadyStateTimeMSec = 1500;
int numSteadyStateFrame = steadyStateTimeMSec / displayIntervalMsec;
// stencil
int dim1Size = 2;                 // x, y respectively
int dim2Size = 2;                 // two edges of rubber band
int dim3Size = 4;                 // four rubber bands
int leastRoiBoxSize = 10;         // minumun size of rubber band
int stencilRoiRectSize = 15;      // should be odd
int stencilRoiRectHalfSize = stencilRoiRectSize / 2;
bool debugMode = true;


MainWindow::MainWindow(const QStringList& arguments, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , ocr_api(nullptr)
{
    // init ui
    ui->setupUi(this);

    // get image
    std::tuple<std::string, std::vector<std::string>, cv::Mat> out = getImageAndOthers(arguments);
    std::string toolModeName = std::get<0>(out);
    std::vector<std::string> pngFiles = std::get<1>(out);
    cv::Mat cvOrgImage = std::get<2>(out);

    // resize image
    cv::Mat cvScaledImage = resizeCvImage(cvOrgImage);

    // init of privates
    ocrRectItem = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem.resize(dim3Size);
    stencilRectItem[0] = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem[1] = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem[2] = new QGraphicsRectItem(0,0,0,0);
    stencilRectItem[3] = new QGraphicsRectItem(0,0,0,0);

    initStructData(toolModeName, pngFiles, cvOrgImage, cvScaledImage);
    initMainGraphicsViewAndScene(cvScaledImage);
    initOcr();
    connectCallbacks();
}
MainWindow::~MainWindow()
{
    delete ui;
    delete ocr_api;
    delete scene;
    delete pixmapItem;
    delete ocrRectItem;
    for (QGraphicsRectItem* tmpStencilRectItem : stencilRectItem)
    {
        delete tmpStencilRectItem;
    }
    delete updateTimer;
    videoStream.release();
    currentImgFrame.release();
}


std::tuple<std::string, std::vector<std::string>, cv::Mat> MainWindow::getImageAndOthers(QStringList args)
{
    /// determine argument
    std::string arg;
    std::string mode;
    if (args.length() == 1)  // there is no argument
    {
        arg = "stream";
        mode = "stream";
    }
    else
    {
        arg = args[1].toStdString();
        mode = "emulate";
    }

    /// load first frame
    cv::Mat cvOrgImage;
    std::vector<std::string> pngFiles;
    if (mode == "stream")
    {
        cvOrgImage = openStreamAndGetFrame();
    }
    else if (mode == "emulate")
    {
       pngFiles = grapEmulationPngList(arg);
       cvOrgImage = cv::imread(pngFiles[0]);
    }
    return std::make_tuple(mode, pngFiles, cvOrgImage);
}


cv::Mat MainWindow::openStreamAndGetFrame()
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
    return currentImgFrame;
}


std::vector<std::string> MainWindow::grapEmulationPngList(const std::string& directoryPath)
{
    std::vector<std::string> pngFiles = Utils::getAllPNGFiles(directoryPath);
    if (!pngFiles.empty()) {
        std::cout << "List of PNG files:" << std::endl;
        for (const auto& filePath : pngFiles) {
            std::cout << "    " << filePath << std::endl;
        }
        std::cout << "\n" << std::endl;
    } else {
        std::cout << "No PNG files found in the specified directory. Please check the path of png files directory ..." << "\n\n" << std::endl;
        QCoreApplication::quit();
    }
    return pngFiles;
}


cv::Mat MainWindow::resizeCvImage(cv::Mat img)
{
    // Resize image to fit the window with maintaining aspect ratio
    int originalWidth = img.cols;
    int originalHeight = img.rows;
    float aspectRatio = (float)originalWidth / (float)originalHeight;
    int targetHeight = static_cast<int>((float)fixedMainWindowWidth / aspectRatio);
    cv::resize(img, img, cv::Size(fixedMainWindowWidth, targetHeight));
    // Convert BGR image to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    return img;
}


void MainWindow::initMainGraphicsViewAndScene(cv::Mat img)
{
    // Setup graphics view here
    ui->mainGraphicsView->setRenderHint(QPainter::Antialiasing);
    ui->mainGraphicsView->setRenderHint(QPainter::SmoothPixmapTransform);
    ui->mainGraphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->mainGraphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->mainGraphicsView->viewport()->installEventFilter(this);
    // Convert cv::Mat to QImage
    QImage qtImage(img.data, img.cols, img.rows, img.step, QImage::Format_RGB888);
    // Set up QGraphicsScene
    QGraphicsScene* scene = new QGraphicsScene(ui->mainGraphicsView);
    ui->mainGraphicsView->setScene(scene);
    // Add image to the scene
    pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
    scene->addItem(pixmapItem);
    // texteditbox
    ui->systemNameTextEditBox->setWordWrapMode(QTextOption::NoWrap);
    ui->probeNameTextEditBox->setWordWrapMode(QTextOption::NoWrap);
    // tablewidget
    int fontSize = 8;
    int tableWidgetWidth = ui->monitoringTableWidget->width();
    int tableWidgetHeight = ui->monitoringTableWidget->height();
    int verticalScrollBarWidth = ui->monitoringTableWidget->style()->pixelMetric(QStyle::PM_ScrollBarExtent);
    ui->monitoringTableWidget->horizontalHeader()->setDefaultSectionSize((tableWidgetWidth - verticalScrollBarWidth) / 3);  // because tableWidget's width is 191, and it has 3 columns.
    ui->monitoringTableWidget->verticalHeader()->setDefaultSectionSize(fontSize);
//    ui->monitoringTableWidget->setRowCount((tableWidgetHeight - 2)/(ui->monitoringTableWidget->rowHeight(0))); // -2 is graphics geek number
    ui->monitoringTableWidget->verticalHeader()->setVisible(false);
    ui->monitoringTableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
    QFont font = ui->monitoringTableWidget->font();
    font.setPointSize(fontSize);
    ui->monitoringTableWidget->setFont(font);
    ui->monitoringTableWidget->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOn);
    ui->monitoringTableWidget->setSelectionMode(QAbstractItemView::NoSelection);
}


void MainWindow::connectCallbacks()
{
    // pushbutton
    connect(ui->ocrRoiPushButton, &QPushButton::clicked, this, &MainWindow::onOcrRoiPushButtonClicked);
    connect(ui->confirmStencilRoiPushButton, &QPushButton::clicked, this, &MainWindow::onConfirmStencilRoiPushButtonClicked);
    connect(ui->generateStencilRoiPushButton, &QPushButton::clicked, this, &MainWindow::onGenerateStencilRoiPushButtonClicked);
    connect(ui->pushItemPushButton, &QPushButton::clicked, this, &MainWindow::onPushItemPushButtonClicked);
    connect(ui->removeItemPushButton_2, &QPushButton::clicked, this, &MainWindow::onRemoveItemPushButton_2Clicked);
    // combobox
    connect(ui->probeTypeComboBox, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &MainWindow::onProbeTypeChanged);
    // checkbox
    connect(ui->applyStencilCheckBox, &QCheckBox::stateChanged, this, &MainWindow::onApplyStencilStateChanged);
    // texteditbox
    connect(ui->systemNameTextEditBox, &QTextEdit::textChanged, this, &MainWindow::onSystemNameTextEdited);
    connect(ui->probeNameTextEditBox, &QTextEdit::textChanged, this, &MainWindow::onProbeNameTextEdited);
    // stream
    updateTimer = new QTimer(this);
    updateTimer->start(displayIntervalMsec);
    connect(updateTimer, &QTimer::timeout, this, &MainWindow::redrawWindow);
    // tableWidget
    connect(ui->monitoringTableWidget, &QTableWidget::cellClicked, this, [=](int row, int column)
    {
        QTableWidgetItem *item = ui->monitoringTableWidget->item(row, column);
        if (item)
        {
            // selection clear
            QTableWidgetItem* tmpItem;
            for (int tmpRow = 0; tmpRow <  ui->monitoringTableWidget->rowCount(); ++tmpRow)
            {
                for (int tmpCol = 0; tmpCol < ui->monitoringTableWidget->columnCount(); ++tmpCol)
                {
                    if (tmpRow != row)
                    {
                        tmpItem = ui->monitoringTableWidget->item(tmpRow, tmpCol);
                        if (tmpItem) // && (!tmpItem->text().isEmpty())
                        {
                            tmpItem->setSelected(false);
                        }
                    }
                }
            }
            // selection update

            if (item -> isSelected())
            {
                for (int tmpCol = 0; tmpCol < ui->monitoringTableWidget->columnCount(); ++tmpCol)
                {
                    tmpItem = ui->monitoringTableWidget->item(row, tmpCol);
                    tmpItem->setSelected(false);
                }
            }
            else
            {
                for (int tmpCol = 0; tmpCol < ui->monitoringTableWidget->columnCount(); ++tmpCol)
                {
                    tmpItem = ui->monitoringTableWidget->item(row, tmpCol);
                    tmpItem->setSelected(true);
                }
            }
        }
    });
}


void MainWindow::initStructData(std::string toolModeName, std::vector<std::string> files, cv::Mat cvOrgImg, cv::Mat cvScaledImg)
{
    // common (will be reported)
    structdata.systemName = "";
    structdata.probeName = "";
    structdata.probeType = "Convex";  // because checkbox's default state is "Convex" (Convex, Linear)
    // common
    structdata.toolModeName = toolModeName;
    structdata.pngFileNames = files;
    structdata.currentDrawingIndex = 0;
    structdata.orgImg = cvOrgImg;
    structdata.scaledOrgImg = cvScaledImg;
    structdata.widthFactor = (float)cvOrgImg.cols / (float)cvScaledImg.cols;
    structdata.heightFactor = (float)cvOrgImg.rows / (float)cvScaledImg.rows;
    // ocrdata
    structdata.ocrdata.isRoiDetermined = false;
    structdata.ocrdata.isRoiCandidated = false;
    structdata.ocrdata.isReadyToDraw = false;
    structdata.ocrdata.numSteadyFrames = 0;
    structdata.ocrdata.depth = "";
    // stencildata
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


/// OCR
void MainWindow::initOcr()
{
    ocr_api = new tesseract::TessBaseAPI();
    if (ocr_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY) != 0)
    {
        std::cerr << "Could not initialize Tesseract." << std::endl;
    }
    ocr_api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr_api->SetVariable("tessedit_char_whitelist", "0123456789");
}


std::string MainWindow::runOcr(const cv::Mat& img)
{
    ocr_api->SetImage(img.data, img.cols, img.rows, 3, img.step);
    ocr_api->SetSourceResolution(250); // updatedImg resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
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


bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    /// mouse event
    if (obj == ui->mainGraphicsView->viewport())
    {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        // check mousebutton left or right
        if (mouseEvent->button() == Qt::LeftButton)
        {
            structdata.mouseButtonStatus = Qt::LeftButton;
        }
        else if (mouseEvent->button() == Qt::RightButton)
        {
            structdata.mouseButtonStatus = Qt::RightButton;
        }

        // actual actions
        if (structdata.mouseButtonStatus == Qt::LeftButton) // Ocr
        {
            leftMouseButtonEvent(mouseEvent);
        }
        else if (structdata.mouseButtonStatus == Qt::RightButton) // Stencil
        {
            rightMouseButtonEvent(mouseEvent);
        }

        // neutralize mousebutton
        if (mouseEvent->type() == QEvent::MouseButtonRelease)
        {
            structdata.mouseButtonStatus = Qt::NoButton;
        }
    }
    return QMainWindow::eventFilter(obj, event);
}


void MainWindow::leftMouseButtonEvent(QMouseEvent* mouseEvent)
{
    if (mouseEvent->type() == QEvent::MouseButtonPress)
    {
        startSelectOcrRubberBand(mouseEvent->pos());
    }
    else if (mouseEvent->type() == QEvent::MouseMove)
    {
        endSelectOcrRubberBand(mouseEvent->pos());
        updateOcrRubberBandInfo();
    }
    else if (mouseEvent->type() == QEvent::MouseButtonRelease)
    {
        endSelectOcrRubberBand(mouseEvent->pos());
        updateOcrRubberBandInfo();
    }
}


void MainWindow::startSelectOcrRubberBand(const QPoint& startPos)
{
    // OCR, initial start position and set status
    structdata.ocrdata.positionX1 = startPos.x();
    structdata.ocrdata.positionY1 = startPos.y();
    structdata.ocrdata.isRoiCandidated = false;
    structdata.ocrdata.isRoiDetermined = false;
    structdata.ocrdata.isReadyToDraw = false;
    ui->depthDisplaylabel->setText(QString::fromStdString(""));
}


void MainWindow::endSelectOcrRubberBand(const QPoint& endPos)
{
    // OCR, end position and set status
    structdata.ocrdata.positionX2 = endPos.x();
    structdata.ocrdata.positionY2 = endPos.y();
    structdata.ocrdata.isRoiCandidated = true;
    structdata.ocrdata.isRoiDetermined = false;
    structdata.ocrdata.isReadyToDraw = true;
}


void MainWindow::updateOcrRubberBandInfo()
{
    // calcuate x, y, width, height from start position and end position
    int x = std::min(structdata.ocrdata.positionX1, structdata.ocrdata.positionX2);
    int y = std::min(structdata.ocrdata.positionY1, structdata.ocrdata.positionY2);
    int width = std::abs(structdata.ocrdata.positionX2 - structdata.ocrdata.positionX1) + 1;
    int height = std::abs(structdata.ocrdata.positionY2 - structdata.ocrdata.positionY1) + 1;
    // save value to structdata.ocrdata.rectInfo
    structdata.ocrdata.rectInfo.x = x;
    structdata.ocrdata.rectInfo.y = y;
    structdata.ocrdata.rectInfo.width = width;
    structdata.ocrdata.rectInfo.height = height;
}


void MainWindow::rightMouseButtonEvent(QMouseEvent* mouseEvent)
{
    if (mouseEvent->type() == QEvent::MouseButtonPress)
    {
        startSelectStencilRubberBand(mouseEvent->pos());
    }
    else if (mouseEvent->type() == QEvent::MouseMove)
    {
        updateSelectStencilRubberBand(mouseEvent->pos());
    }
    else if (mouseEvent->type() == QEvent::MouseButtonRelease)
    {
        endSelectStencilRubberBand(mouseEvent->pos());
    }
    int left_x = std::min(structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx], structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx]);
    int right_x = std::max(structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx], structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx]);
    int top_y = std::min(structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx], structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx]);
    int bottom_y = std::max(structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx], structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx]);
    structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].x = left_x;
    structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].y = top_y;
    structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].width = (right_x - left_x) + 1;
    structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].height = (bottom_y - top_y) + 1;
    structdata.stencildata.isMaskExist = false;

    printStencilData(debugMode);
}


void MainWindow::startSelectStencilRubberBand(const QPoint& pos)
{
    // selection logic of update target
    int candidatePositionX = pos.x();
    int candidatePositionY = pos.y();

    structdata.stencildata.updateTargetIdx = searchAlreadyExistStencilRubberBandIndex(pos);
    if (structdata.stencildata.updateTargetIdx == -1)
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


int MainWindow::searchAlreadyExistStencilRubberBandIndex(const QPoint& point)
{
    for (int dim3 = 0; dim3 < dim3Size; ++dim3)
    {
        if (structdata.stencildata.isRoiCandidated[dim3])
        {
            int left_x = std::min(structdata.stencildata.positionA[0][0][dim3], structdata.stencildata.positionA[0][1][dim3]);
            int right_x = std::max(structdata.stencildata.positionA[0][0][dim3], structdata.stencildata.positionA[0][1][dim3]);
            int top_y = std::min(structdata.stencildata.positionA[1][0][dim3], structdata.stencildata.positionA[1][1][dim3]);
            int bottom_y = std::max(structdata.stencildata.positionA[1][0][dim3], structdata.stencildata.positionA[1][1][dim3]);
            if ((point.x() >= left_x) && (point.x() <= right_x) && (point.y() >= top_y) && (point.y() <= bottom_y))
            {
                return dim3;
            }
        }
    }
    return -1;
}


void MainWindow::updateSelectStencilRubberBand(const QPoint& pos)
{
    int candidatePositionX = pos.x();
    int candidatePositionY = pos.y();
    if (structdata.stencildata.updateTargetIdx != -1)
    //    if ((structdata.stencildata.updateTargetIdx != -1) && (!structdata.stencildata.isFourRoiReady))
    {
        structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] = candidatePositionX;
        structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] = candidatePositionY;
        structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = true;
        structdata.stencildata.isFourRoiReady = false;
        structdata.stencildata.isRoiDetermined = false;
        structdata.stencildata.isReadyToDraw = true;
    }
}


void MainWindow::endSelectStencilRubberBand(const QPoint& pos)
{
    int candidatePositionX = pos.x();
    int candidatePositionY = pos.y();
    if (structdata.stencildata.updateTargetIdx != -1)
    {
        bool isEndedAtIdenticalLocation = ((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] == candidatePositionX) ||
                                           (structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] == candidatePositionY));
        bool isTooSmallRoiSize = ((std::abs(structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] - candidatePositionX) < leastRoiBoxSize) ||
                                  (std::abs(structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] - candidatePositionY) < leastRoiBoxSize));
        bool isOverwrappedRoi = searchOverlapedRubberBand(structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx], structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx], candidatePositionX, candidatePositionY);
        if (isEndedAtIdenticalLocation || isTooSmallRoiSize || isOverwrappedRoi)
        {
            structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = false;
        }
        else
        {
            structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] = candidatePositionX;
            structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] = candidatePositionY;
            structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = true;
        }
        structdata.stencildata.isFourRoiReady = Utils::andArray(structdata.stencildata.isRoiCandidated);
        structdata.stencildata.isRoiDetermined = false;
        structdata.stencildata.isReadyToDraw = true;
    }
}


bool MainWindow::searchOverlapedRubberBand(const int x1, const int y1, const int x2, const int y2)
{
    int rect2_left_x = std::min(x1, x2);
    int rect2_right_x = std::max(x1, x2);
    int rect2_top_y = std::min(y1, y2);
    int rect2_bottom_y = std::max(y1, y2);
    for (int dim3 = 0; dim3 < dim3Size; ++dim3)
    {
        if (!(dim3 == structdata.stencildata.updateTargetIdx) && (structdata.stencildata.isRoiCandidated[dim3]))
        {
            int rect1_left_x = std::min(structdata.stencildata.positionA[0][0][dim3], structdata.stencildata.positionA[0][1][dim3]);
            int rect1_right_x = std::max(structdata.stencildata.positionA[0][0][dim3], structdata.stencildata.positionA[0][1][dim3]);
            int rect1_top_y = std::min(structdata.stencildata.positionA[1][0][dim3], structdata.stencildata.positionA[1][1][dim3]);
            int rect1_bottom_y = std::max(structdata.stencildata.positionA[1][0][dim3], structdata.stencildata.positionA[1][1][dim3]);

            bool xOverlap = rect1_left_x < rect2_right_x && rect1_right_x > rect2_left_x;
            bool yOverlap = rect1_top_y < rect2_bottom_y && rect1_bottom_y > rect2_top_y;
            if (xOverlap && yOverlap)
            {
                return true;
            }
        }
    }
    return false;
}


/// pushbutton
void MainWindow::onOcrRoiPushButtonClicked()
{
    if (structdata.ocrdata.isRoiCandidated)
    {
        structdata.ocrdata.recoveredRectInfo.x = (int)((float)structdata.ocrdata.rectInfo.x * structdata.widthFactor + 0.5f);
        structdata.ocrdata.recoveredRectInfo.y = (int)((float)structdata.ocrdata.rectInfo.y * structdata.heightFactor + 0.5f);
        structdata.ocrdata.recoveredRectInfo.width = (int)((float)structdata.ocrdata.rectInfo.width * structdata.widthFactor + 0.5f);
        structdata.ocrdata.recoveredRectInfo.height = (int)((float)structdata.ocrdata.rectInfo.height * structdata.heightFactor + 0.5f);
        structdata.ocrdata.isRoiDetermined = true;
        structdata.ocrdata.numSteadyFrames = numSteadyStateFrame;
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
        std::tuple<std::vector<int>, std::vector<int>, int> profileData = Utils::calculateProfileData(topLeftImg, -1);
        std::vector<int> topLeftImgWidthProfileData = std::get<0>(profileData);
        std::vector<int> topLeftImgHeightProfileData = std::get<1>(profileData);
        int minBrightness = std::get<2>(profileData);
        profileData = Utils::calculateProfileData(topRightImg, -1);
        std::vector<int> topRightImgWidthProfileData = std::get<0>(profileData);
        std::vector<int> topRightImgHeightProfileData = std::get<1>(profileData);
        profileData = Utils::calculateProfileData(bottomLeftImg, -1);
        std::vector<int> bottomLeftImgWidthProfileData = std::get<0>(profileData);
        std::vector<int> bottomLeftImgHeightProfileData = std::get<1>(profileData);
        profileData = Utils::calculateProfileData(bottomRightImg, -1);
        std::vector<int> bottomRightImgWidthProfileData = std::get<0>(profileData);
        std::vector<int> bottomRightImgHeightProfileData = std::get<1>(profileData);
        int topLeftImgX=0, topLeftImgY=0, topRightImgX=0, topRightImgY=0, bottomLeftImgX=0, bottomLeftImgY=0, bottomRightImgX=0, bottomRightImgY=0;
        if (structdata.probeType == "Convex")
        {
            // find four corner points
            topLeftImgX = Utils::findFirstMaxIndex(topLeftImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[topLeftIdx].x;
            topLeftImgY = Utils::findFirstNonZeroIndex(topLeftImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[topLeftIdx].y;
            topRightImgX = Utils::findLastMaxIndex(topRightImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[topRightIdx].x;
            topRightImgY = Utils::findFirstNonZeroIndex(topRightImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[topRightIdx].y;
            bottomLeftImgX = Utils::findFirstNonZeroIndex(bottomLeftImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[bottomLeftIdx].x;
            bottomLeftImgY = Utils::findFirstMaxIndex(bottomLeftImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[bottomLeftIdx].y;
            bottomRightImgX = Utils::findLastNonZeroIndex(bottomRightImgWidthProfileData) + structdata.stencildata.recoveredRectInfo[bottomRightIdx].x;
            bottomRightImgY = Utils::findFirstMaxIndex(bottomRightImgHeightProfileData) + structdata.stencildata.recoveredRectInfo[bottomRightIdx].y;
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
        QPoint bottomRightPoint((topLeftImgX + topRightImgX) * 0.5 + ((topLeftImgX + topRightImgX) * 0.5 - bottomLeftImgX), bottomLeftImgY);
        QPainterPath stencilPath;
        QPointF topLeftPointF, bottomLeftPointF, topRightPointF, bottomRightPointF;
        cv::Rect boundRect;
        topLeftPointF = QPointF(topLeftPoint);
        bottomLeftPointF = QPointF(bottomLeftPoint);
        topRightPointF = QPointF(topRightPoint);
        bottomRightPointF = QPointF(bottomRightPoint);

        //structdata.scaledMaskTopLeftPointInt.setX(static_cast<int>(topLeftPointF.x() / structdata.widthFactor));
        //structdata.scaledMaskTopLeftPointInt.setY(static_cast<int>(topLeftPointF.y() / structdata.widthFactor));
        //structdata.scaledMaskBottomLeftPointInt.setX(static_cast<int>(bottomLeftPointF.x() / structdata.widthFactor));
        //structdata.scaledMaskBottomLeftPointInt.setY(static_cast<int>(bottomLeftPointF.y() / structdata.widthFactor));
        //structdata.scaledMaskTopRightPointInt.setX(static_cast<int>(topRightPointF.x() / structdata.widthFactor));
        //structdata.scaledMaskTopRightPointInt.setY(static_cast<int>(topRightPointF.y() / structdata.widthFactor));
        //structdata.scaledMaskBottomRightPointInt.setX(static_cast<int>(bottomRightPointF.x() / structdata.widthFactor));
        //structdata.scaledMaskBottomRightPointInt.setY(static_cast<int>(bottomRightPointF.y() / structdata.widthFactor));

        if (structdata.probeType == "Convex")
        {
            // calc apex point
            QPointF apexPoint;
            apexPoint = Utils::intersectionPointof2Lines(topLeftPoint, bottomLeftPoint, topRightPoint, bottomRightPoint);
            // outerRadius and innerRadius
            cv::Mat centralImg = structdata.orgImg(cv::Rect(topLeftImgX, topLeftImgY, (topRightImgX - topLeftImgX) + 1, ((structdata.orgImg.rows - 1) - topLeftImgY) + 1));
            profileData = Utils::calculateProfileData(centralImg, minBrightness);
            std::vector<int> centralImgHeightProfileData = std::get<1>(profileData);
            qreal centralImgOuterY = qreal(Utils::findFirstZeroIndex(centralImgHeightProfileData) + topLeftImgY - 1);
            qreal outerRadius = centralImgOuterY - apexPoint.y() + 1;
            qreal innerRadius = QLineF(apexPoint, topLeftPointF).length();
            // get boundBox info
            boundRect.x = bottomLeftPoint.x();
            boundRect.y = topLeftPoint.y();
            boundRect.width = bottomRightPoint.x() - bottomLeftPoint.x() + 1;
            boundRect.height = centralImgOuterY - topLeftPoint.y() + 1;
            // fov span angle
            double leftSlopeAngleRad = std::atan2(bottomLeftPointF.y() - apexPoint.y(), bottomLeftPointF.x() - apexPoint.x());
            double rightSlopeAngleRad = std::atan2(bottomRightPointF.y() - apexPoint.y(), bottomRightPointF.x() - apexPoint.x());
            double leftSlopeAngleDeg = leftSlopeAngleRad * 180.0 / M_PI;
            double rightSlopeAngleDeg = rightSlopeAngleRad * 180.0 / M_PI;
            double spanAngleDeg = leftSlopeAngleDeg - rightSlopeAngleDeg;
            // recalc bottomLeftPointF bottomRightPointF
            bottomLeftPointF.setX(apexPoint.x() - std::sin(leftSlopeAngleRad - (M_PI * 0.5)) * outerRadius);
            bottomLeftPointF.setY(apexPoint.y() + std::cos(leftSlopeAngleRad - (M_PI * 0.5)) * outerRadius);
            bottomRightPointF.setX(apexPoint.x() + (apexPoint.x() - bottomLeftPointF.x()));
            bottomRightPointF.setY(bottomLeftPointF.y());
            // draw fov
            stencilPath.moveTo(topLeftPointF);
            stencilPath.lineTo(bottomLeftPointF);
            stencilPath.arcTo(apexPoint.x()-outerRadius, apexPoint.y()-outerRadius, 2 * outerRadius, 2 * outerRadius, -leftSlopeAngleDeg, spanAngleDeg);
            stencilPath.lineTo(bottomRightPointF);
            stencilPath.lineTo(topRightPointF);
            stencilPath.arcTo(apexPoint.x()-innerRadius, apexPoint.y()-innerRadius, 2 * innerRadius, 2 * innerRadius, -rightSlopeAngleDeg, -spanAngleDeg);
            stencilPath.closeSubpath();
        }
        else if (structdata.probeType == "Linear")
        {
            // get boundBox info
            boundRect.x = topLeftPoint.x();
            boundRect.y = topLeftPoint.y();
            boundRect.width = topRightPoint.x() - topLeftPoint.x() + 1;
            boundRect.height = bottomLeftPoint.y() - topLeftPoint.y() + 1;
            // draw fov
            stencilPath.moveTo(topLeftPointF);
            stencilPath.lineTo(bottomLeftPointF);
            stencilPath.lineTo(bottomRightPointF);
            stencilPath.lineTo(topRightPointF);
            stencilPath.lineTo(topLeftPointF);
            stencilPath.closeSubpath();
        }
        // draw bound
        QPainterPath boundPath;
        QRect qBoundRect(boundRect.x, boundRect.y, boundRect.width, boundRect.height);
        boundPath.addRect(qBoundRect);

        // get masks
        QImage qtOrgImgMask(structdata.orgImg.cols, structdata.orgImg.rows, QImage::Format_ARGB32);
        QImage qtOrgImgBoundMask(structdata.orgImg.cols, structdata.orgImg.rows, QImage::Format_ARGB32);
        qtOrgImgMask.fill(Qt::black);
        qtOrgImgBoundMask.fill(Qt::black);
        QPainter maskPainter(&qtOrgImgMask);
        QPainter boundMaskPainter(&qtOrgImgBoundMask);
        QBrush brush(Qt::white);
        maskPainter.setBrush(brush);
        boundMaskPainter.setBrush(brush);
        maskPainter.fillPath(stencilPath, brush);
        boundMaskPainter.fillPath(boundPath, brush);
        cv::Mat cvScaledImgMask;
        cv::Mat cvScaledImgBoundMask;
        cv::Mat cvOrgImgMask = cv::Mat(qtOrgImgMask.height(), qtOrgImgMask.width(), CV_8UC4, const_cast<uchar*>(qtOrgImgMask.bits()), qtOrgImgMask.bytesPerLine());
        cv::Mat cvOrgImgBoundMask = cv::Mat(qtOrgImgBoundMask.height(), qtOrgImgBoundMask.width(), CV_8UC4, const_cast<uchar*>(qtOrgImgBoundMask.bits()), qtOrgImgBoundMask.bytesPerLine());
        cv::resize(cvOrgImgMask, cvScaledImgMask, cv::Size(structdata.scaledOrgImg.cols, structdata.scaledOrgImg.rows));
        cv::resize(cvOrgImgBoundMask, cvScaledImgBoundMask, cv::Size(structdata.scaledOrgImg.cols, structdata.scaledOrgImg.rows));
        cv::threshold(cvOrgImgMask, cvOrgImgMask, 127, 1, cv::THRESH_BINARY);
        cv::threshold(cvOrgImgBoundMask, cvOrgImgBoundMask, 127, 1, cv::THRESH_BINARY);
        cv::threshold(cvScaledImgMask, cvScaledImgMask, 127, 1, cv::THRESH_BINARY);
        cv::threshold(cvScaledImgBoundMask, cvScaledImgBoundMask, 127, 1, cv::THRESH_BINARY);
        cv::cvtColor(cvOrgImgMask, cvOrgImgMask, cv::COLOR_BGR2RGB);
        cv::cvtColor(cvOrgImgBoundMask, cvOrgImgBoundMask, cv::COLOR_BGR2RGB);
        cv::cvtColor(cvScaledImgMask, cvScaledImgMask, cv::COLOR_BGR2RGB);
        cv::cvtColor(cvScaledImgBoundMask, cvScaledImgBoundMask, cv::COLOR_BGR2RGB);
        cv::multiply(cvOrgImgMask, cvOrgImgBoundMask, cvOrgImgMask);
        cv::multiply(cvScaledImgMask, cvScaledImgBoundMask, cvScaledImgMask);
        structdata.orgImgMask = cvOrgImgMask;
        structdata.scaledOrgImgMask = cvScaledImgMask;
        structdata.stencildata.isMaskExist = true;

        // get data for report
        structdata.stencildata.topLeftX = static_cast<float>(topLeftPointF.x());
        structdata.stencildata.topLeftY = static_cast<float>(topLeftPointF.y());
        structdata.stencildata.topRightX = static_cast<float>(topRightPointF.x());
        structdata.stencildata.topRightY = static_cast<float>(topRightPointF.y());
        structdata.stencildata.bottomLeftX = static_cast<float>(bottomLeftPointF.x());
        structdata.stencildata.bottomLeftY = static_cast<float>(bottomLeftPointF.y());
        structdata.stencildata.bottomRightX = static_cast<float>(bottomRightPointF.x());
        structdata.stencildata.bottomRightY = static_cast<float>(bottomRightPointF.y());
        structdata.stencildata.boundRect = boundRect;
    }
}


void MainWindow::onPushItemPushButtonClicked()
{
    if ((!structdata.systemName.empty()) && (!structdata.probeName.empty()) && (!structdata.ocrdata.depth.empty()))
    {
        QString qSystemName = QString::fromStdString(structdata.systemName);
        QString qProbeName = QString::fromStdString(structdata.probeName);
        QString qDepth = QString::fromStdString(structdata.ocrdata.depth);
        QTableWidgetItem* itemSystemName;
        QTableWidgetItem* itemProbeName;
        QTableWidgetItem* itemDepth;
        QMessageBox::StandardButton reply = QMessageBox::Yes;
        std::string tableUpdateState = "insert";
        int updateRow = 0;
        for (int row = 0; row < ui->monitoringTableWidget->rowCount(); row++)
        {
            itemSystemName = ui->monitoringTableWidget->item(row, 0);
            itemProbeName = ui->monitoringTableWidget->item(row, 1);
            itemDepth = ui->monitoringTableWidget->item(row, 2);
            if (((qSystemName == itemSystemName->text()) && (qProbeName == itemProbeName->text()) && (qDepth == itemDepth->text())))
            {
                reply = QMessageBox::question(nullptr, "Question", "Do you want to overwrite data?", QMessageBox::Yes|QMessageBox::No);
                updateRow = row;
                tableUpdateState = "update";
                break;
            }
            else if ((qSystemName.compare(itemSystemName->text()) < 0) ||
               ((qSystemName == itemSystemName->text()) && (qProbeName.compare(itemProbeName->text()) < 0) ||
               ((qSystemName == itemSystemName->text()) && (qProbeName == itemProbeName->text()) && (qDepth.toFloat() < itemDepth->text().toFloat()))))
            {
                reply = QMessageBox::Yes;
                updateRow = row;
                tableUpdateState = "insert";
                break;
            }
            else // error fix
            {
                updateRow = ui->monitoringTableWidget->rowCount();
                tableUpdateState = "insert";
            }
        }
        if (reply == QMessageBox::Yes)
        {
            if (tableUpdateState == "update")
            {
                updateTableCell(ui->monitoringTableWidget, updateRow, 0, qSystemName);
                updateTableCell(ui->monitoringTableWidget, updateRow, 1, qProbeName);
                updateTableCell(ui->monitoringTableWidget, updateRow, 2, qDepth);
            }
            else if (tableUpdateState == "insert")
            {
                insertTableCell(ui->monitoringTableWidget, updateRow, 0, qSystemName);
                updateTableCell(ui->monitoringTableWidget, updateRow, 1, qProbeName);
                updateTableCell(ui->monitoringTableWidget, updateRow, 2, qDepth);
            }
            changeRowCellsColor(ui->monitoringTableWidget, updateRow, QColor(Qt::green));
        }
        else
        {
            if (tableUpdateState == "update")
            {
               changeRowCellsColor(ui->monitoringTableWidget, updateRow, QColor(Qt::red));
            }
        }
    }
}


void MainWindow::updateTableCell(QTableWidget* tableWidget, int row, int column, const QString& newText)
{
    QTableWidgetItem* item = tableWidget->item(row, column);
    if (item)
    {
        item->setText(newText);
    }
    else
    {
        item = new QTableWidgetItem(newText);
        tableWidget->setItem(row, column, item);
    }
}


void MainWindow::insertTableCell(QTableWidget* tableWidget, int row, int column, const QString& newText)
{
    QTableWidgetItem* newItemText = new QTableWidgetItem(newText);
    tableWidget->insertRow(row);
    tableWidget->setItem(row, column, newItemText);
}


void MainWindow::changeRowCellsColor(QTableWidget* tableWidget, int row, const QColor& color)
{
    int colorPersistTime = 500;
    int columnCount = tableWidget->columnCount();
    for (int column = 0; column < columnCount; ++column)
    {
        QTableWidgetItem* item = tableWidget->item(row, column);
        if (item)
        {
            item->setBackground(color);
        }
    }
    QTimer::singleShot(colorPersistTime, [=]()
    {
        for (int column = 0; column < columnCount; ++column)
        {
            QTableWidgetItem* item = tableWidget->item(row, column);
            if (item)
            {
                item->setBackground(QColor(Qt::white));
            }
        }
    });
}



void MainWindow::changeRowWordsColor(QTableWidget* tableWidget, int row, const QColor& color)
{
    int colorPersistTime = 500;
    int columnCount = tableWidget->columnCount();
    QBrush brush(color);
    for (int column = 0; column < columnCount; ++column)
    {
        QTableWidgetItem* item = tableWidget->item(row, column);
        if (item)
        {
            item->setForeground(brush);
        }
    }
    brush.setColor(Qt::black);
    QTimer::singleShot(colorPersistTime, [=]()
    {
        for (int column = 0; column < columnCount; ++column)
        {
            QTableWidgetItem* item = tableWidget->item(row, column);
            if (item)
            {
                item->setForeground(brush);
            }
        }
    });
}


void MainWindow::onRemoveItemPushButton_2Clicked()
{
    QList<QTableWidgetItem*> selectedItems = ui->monitoringTableWidget->selectedItems();
    if (!selectedItems.isEmpty())
    {
        int selectedRow = selectedItems.first()->row();
        ui->monitoringTableWidget->removeRow(selectedRow);
    }
}


/// combobox
void MainWindow::onProbeTypeChanged()
{
    structdata.probeType = ui->probeTypeComboBox->currentText().toStdString();
    onGenerateStencilRoiPushButtonClicked();
}


/// checkbox
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


/// texteditbox
void MainWindow::onSystemNameTextEdited()
{
    QString text = ui->systemNameTextEditBox->toPlainText();
    text.replace("\n", "").replace("\r", "").replace("\t", "").remove(QRegularExpression("\\s+"));
    disconnect(ui->systemNameTextEditBox, &QTextEdit::textChanged, this, &MainWindow::onSystemNameTextEdited);
    ui->systemNameTextEditBox->setPlainText(text);
    connect(ui->systemNameTextEditBox, &QTextEdit::textChanged, this, &MainWindow::onSystemNameTextEdited);
    QTextCursor cursor(ui->systemNameTextEditBox->textCursor());
    cursor.movePosition(QTextCursor::End);
    ui->systemNameTextEditBox->setTextCursor(cursor);
    structdata.systemName = text.toStdString();
}


void MainWindow::onProbeNameTextEdited()
{
    QString text = ui->probeNameTextEditBox->toPlainText();
    text.replace("\n", "").replace("\r", "").replace("\t", "").remove(QRegularExpression("\\s+"));
    disconnect(ui->probeNameTextEditBox, &QTextEdit::textChanged, this, &MainWindow::onProbeNameTextEdited);
    ui->probeNameTextEditBox->setPlainText(text);
    connect(ui->probeNameTextEditBox, &QTextEdit::textChanged, this, &MainWindow::onProbeNameTextEdited);
    QTextCursor cursor(ui->probeNameTextEditBox->textCursor());
    cursor.movePosition(QTextCursor::End);
    ui->probeNameTextEditBox->setTextCursor(cursor);
    structdata.probeName = text.toStdString();
}


/// recalc of drawing on mainwindow
void MainWindow::redrawWindow()
{
    cv::Mat cvImage = reloadCvImage();

    // draw image on scene
    if ((structdata.stencildata.isMaskExist) && (structdata.stencildata.isStencilApplied))
    {
        cv::Mat maskedCvImage;
        cv::multiply(cvImage, structdata.scaledOrgImgMask, maskedCvImage);
        redrawScene(maskedCvImage);
    }
    else
    {
        redrawScene(cvImage);
    }

    // draw rubber bands
    if (!(structdata.stencildata.isMaskExist && structdata.stencildata.isStencilApplied))
    {
        if (structdata.ocrdata.isReadyToDraw)
        {
            drawOcrRectRubberBand();


            // OCR inference
            ui->depthDisplaylabel->setText(QString::fromStdString(""));
            structdata.ocrdata.depth = "";
            if (structdata.ocrdata.isRoiDetermined)
            {
                // get subROI
                cv::Mat subRoi_img = structdata.orgImg(structdata.ocrdata.recoveredRectInfo);
                cv::Mat diff;
                // count for whether of steady state or not
                if (structdata.ocrdata.prevSubRoiImg.empty())
                {
                    ++structdata.ocrdata.numSteadyFrames;
                }
                else
                {
                    if (!((subRoi_img.rows == structdata.ocrdata.prevSubRoiImg.rows) && (subRoi_img.cols == structdata.ocrdata.prevSubRoiImg.cols)))
                    {
                        structdata.ocrdata.numSteadyFrames = numSteadyStateFrame;
                    }
                    else
                    {
                        diff = subRoi_img != structdata.ocrdata.prevSubRoiImg;
                        if (cv::sum(diff)==cv::Scalar(0,0,0,0))
                        {
                            ++structdata.ocrdata.numSteadyFrames;
                        }
                        else
                        {
                            structdata.ocrdata.numSteadyFrames = 0;
                        }
                    }
                }
                if (structdata.ocrdata.numSteadyFrames >= numSteadyStateFrame)
                {
                    // conduct Tesseract OCR
                    std::string outText = runOcr(subRoi_img);
                    // write the result at textbox
                    ui->depthDisplaylabel->setText(QString::fromStdString(outText));
                    ui->depthDisplaylabel->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);
                    structdata.ocrdata.depth = ui->depthDisplaylabel->text().toStdString();
                }
                // prevSubRoiImg
                structdata.ocrdata.prevSubRoiImg = subRoi_img.clone();
            }


        }
        if (structdata.stencildata.isReadyToDraw)
        {
            drawStencilRectRubberBands();
            //if (structdata.stencildata.isMaskExist)
            //{
            //    QList<QPoint> points;
            //    points << structdata.scaledMaskTopLeftPointInt << structdata.scaledMaskBottomLeftPointInt << structdata.scaledMaskTopRightPointInt << structdata.scaledMaskBottomRightPointInt;
            //    drawCrossLines(points);
            //}
        }
    }
}


cv::Mat MainWindow::reloadCvImage() {
    int nextDrawingIndex = (structdata.currentDrawingIndex + 1) % ((int)structdata.pngFileNames.size());
    cv::Mat cvOrgImage, cvScaledImage;
    if (structdata.toolModeName == "stream")
    {
        videoStream >> currentImgFrame;
        if (currentImgFrame.empty())
        {
            qDebug() << "    Error: Sudden end of video stream...\n\n";
            videoStream.release();
            QCoreApplication::quit();
        }
        cvOrgImage = currentImgFrame;
    }
    else if (structdata.toolModeName == "emulate")
    {
        cvOrgImage = cv::imread(structdata.pngFileNames[nextDrawingIndex]);
    }
    cv::resize(cvOrgImage, cvScaledImage, cv::Size(structdata.scaledOrgImg.cols, structdata.scaledOrgImg.rows));
    cv::cvtColor(cvScaledImage, cvScaledImage, cv::COLOR_BGR2RGB);
    structdata.orgImg = cvOrgImage;
    structdata.scaledOrgImg = cvScaledImage;
    structdata.currentDrawingIndex = nextDrawingIndex;
    return cvScaledImage;
}


void MainWindow::redrawScene(const cv::Mat& image)
{
    if (pixmapItem != nullptr)
    {
        delete pixmapItem;
        pixmapItem = nullptr;
    }
    QImage qtImage(image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
    ui->mainGraphicsView->scene()->addItem(pixmapItem);
}


void MainWindow::drawOcrRectRubberBand()
{
    if (ocrRectItem != nullptr)
    {
        delete ocrRectItem;
        ocrRectItem = nullptr;
    }
    QPen pen(structdata.ocrdata.isRoiDetermined ? Qt::green : Qt::red);
    pen.setWidthF(structdata.ocrdata.isRoiDetermined ? 2.0 : 1.0);
    QRect qRect = QRect(structdata.ocrdata.rectInfo.x, structdata.ocrdata.rectInfo.y, structdata.ocrdata.rectInfo.width, structdata.ocrdata.rectInfo.height);
    ocrRectItem = new QGraphicsRectItem(qRect);
    ocrRectItem->setPen(pen);
    ui->mainGraphicsView->scene()->addItem(ocrRectItem);
}


void MainWindow::drawStencilRectRubberBands()
{
    QPen pen(structdata.stencildata.isRoiDetermined ? Qt::blue : Qt::yellow);
    pen.setWidthF(structdata.stencildata.isRoiDetermined ? 2.0 : 1.0);
    for (int dim3 = 0; dim3 < dim3Size; ++dim3)
    {
        if (structdata.stencildata.isRoiCandidated[dim3])
        {
            if (stencilRectItem[dim3] != nullptr)
            {
                delete stencilRectItem[dim3];
                stencilRectItem[dim3] = nullptr;
            }
            cv::Rect tmpRect = structdata.stencildata.rectInfo[dim3];
            QRectF qRect(tmpRect.x, tmpRect.y, tmpRect.width, tmpRect.height);
            stencilRectItem[dim3] = new QGraphicsRectItem(qRect);
            stencilRectItem[dim3]->setPen(pen);
            ui->mainGraphicsView->scene()->addItem(stencilRectItem[dim3]);
        }
    }
}


//void MainWindow::drawCrossLines(const QList<QPoint>& points)
//{
//    static QList<QGraphicsLineItem*> previousLines;
//    for (QGraphicsLineItem* line : previousLines)
//    {
//        ui->mainGraphicsView->scene()->removeItem(line);
//        delete line;
//    }
//    QList<QGraphicsLineItem*> currentLines;
//    QPen pen(Qt::green);
//    pen.setWidthF(1.5);
//    int crossLineLength = 8;
//    foreach (const QPoint& point, points)
//    {
//        QGraphicsLineItem* horizontalLine1 = new QGraphicsLineItem(
//            point.x() - crossLineLength / 2,
//            point.y() - crossLineLength / 2,
//            point.x() + crossLineLength / 2,
//            point.y() + crossLineLength / 2
//        );
//        horizontalLine1->setPen(pen);
//        ui->mainGraphicsView->scene()->addItem(horizontalLine1);
//        currentLines.append(horizontalLine1);
//        QGraphicsLineItem* horizontalLine2 = new QGraphicsLineItem(
//            point.x() - crossLineLength / 2,
//            point.y() + crossLineLength / 2,
//            point.x() + crossLineLength / 2,
//            point.y() - crossLineLength / 2
//        );
//        horizontalLine2->setPen(pen);
//        ui->mainGraphicsView->scene()->addItem(horizontalLine2);
//        currentLines.append(horizontalLine2);
//    }
//    previousLines = currentLines;
//}



// debuger
void MainWindow::printStencilData(bool debugMode)
{
    if (debugMode)
    {
        qDebug() << "\n";
        qDebug() << "updateTargetIdx: " << structdata.stencildata.updateTargetIdx;
//        qDebug() << "positionA:";
//        for (const auto& dim1 : structdata.stencildata.positionA)
//        {
//            for (const auto& dim2 : dim1)
//            {
//                for (int dim3 : dim2)
//                {
//                    qDebug() << dim3;
//                }
//            }
//        }
        qDebug() << "isRoiCandidated:" << structdata.stencildata.isRoiCandidated[0] << ", " << structdata.stencildata.isRoiCandidated[1] << ", " << structdata.stencildata.isRoiCandidated[2] << ", " << structdata.stencildata.isRoiCandidated[3];
        qDebug() << "isFourRoiReady: " << structdata.stencildata.isFourRoiReady;
        qDebug() << "isRoiDetermined: " << structdata.stencildata.isRoiDetermined;
        qDebug() << "isReadyToDraw: " << structdata.stencildata.isReadyToDraw;
//        qDebug() << "isMaskExist: " << structdata.stencildata.isMaskExist;
//        qDebug() << "isStencilApplied: " << structdata.stencildata.isStencilApplied;
//        qDebug() << "rectInfo:";
//        for (const cv::Rect& rect : structdata.stencildata.rectInfo)
//        {
//            qDebug() << "x: " << rect.x << ", y: " << rect.y << ", width: " << rect.width << ", height: " << rect.height;
//        }
//        qDebug() << "recoveredRectInfo:";
//        for (const cv::Rect& rect : structdata.stencildata.recoveredRectInfo)
//        {
//            qDebug() << "x: " << rect.x << ", y: " << rect.y << ", width: " << rect.width << ", height: " << rect.height;
//        }
    }
}
