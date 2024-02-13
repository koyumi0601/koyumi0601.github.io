#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include "utils.h"
#include <QPixmap>
#include <QMouseEvent>
#include <QDebug>
#include <QtMath>
#include <cmath>
#include <stdexcept>


// set global variables
StructData structdata;
int fixedMainWindowWidth = 1024;  // width of main window
int fixedMainWindowHeight = 576;  // height of main window
int dim1Size = 2;                 // x, y respectively
int dim2Size = 2;                 // two edges of rubber band
int dim3Size = 4;                 // four rubber bands
float distanceThreshold = (float)fixedMainWindowHeight * 0.05;
int leastRoiBoxSize = 10;


MainWindow::MainWindow(const QStringList& arguments, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , ocr_api(nullptr)
{
    ui->setupUi(this);

    /// get all pngs at resource directory
    std::vector<std::string> pngFiles = Utils::getAllPNGFiles(arguments[1].toStdString());
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
        std::cout << "No PNG files found in the specified directory. Please check the path of png files directory ..." << "\n\n" << std::endl;
        QCoreApplication::quit();
    }

    /// init Tesseract OCR
    initOCR();

    /// load image through cv
    cv::Mat cvImage = cv::imread(pngFiles[0]);
    cv::Mat cvOrgImage = cvImage;
    // resize image to fit the window with keeping aspect ratio of org image
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

    // init timer for image update, mimiced image stream ...
    updateTimer = new QTimer(this);
    connect(updateTimer, &QTimer::timeout, this, &MainWindow::redrawWindow);
    updateTimer->start(33);

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
    delete pixmapItem;
    delete rectItem;
    delete updateTimer;
}


/// OCR
void MainWindow::initOCR()
{
    ocr_api = new tesseract::TessBaseAPI();
    if (ocr_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY) != 0)
    {
        std::cerr << "Could not initialize Tesseract." << std::endl;
    }
    ocr_api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
    ocr_api->SetVariable("tessedit_char_whitelist", "0123456789"); // ocr_api->SetSourceResolution(250); // updatedImg resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
}


std::string MainWindow::runOCR(const cv::Mat& inputImage)
{

    ocr_api->SetImage(inputImage.data, inputImage.cols, inputImage.rows, 3, inputImage.step); // ocr_api->SetSourceResolution(250); // updatedImg resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
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


//    // 우클릭인 경우
//    /// mouse events for stencil
//    else if (obj == ui->mainGraphicsView->viewport() && structdata.mouseButtonStatus == Qt::RightButton)
//    {
//        int candidatePositionX = mouseEvent->pos().x();
//        int candidatePositionY = mouseEvent->pos().y();
//        if (event->type() == QEvent::MouseButtonPress)
//        {
//            // selection logic of update target
//            bool reselectCandidate = false;
//            for (int dim3 = 0; dim3 < dim3Size; ++dim3) {
//                if (structdata.stencildata.isRoiCandidated[dim3])
//                {
//                    if (((((structdata.stencildata.positionA[0][0][dim3] <= candidatePositionX) && (candidatePositionX <= structdata.stencildata.positionA[0][1][dim3]))
//                        ||((structdata.stencildata.positionA[0][1][dim3] <= candidatePositionX) && (candidatePositionX <= structdata.stencildata.positionA[0][0][dim3])))
//                       &&(((structdata.stencildata.positionA[1][0][dim3] <= candidatePositionY) && (candidatePositionY <= structdata.stencildata.positionA[1][1][dim3]))
//                        ||((structdata.stencildata.positionA[1][1][dim3] <= candidatePositionY) && (candidatePositionY <= structdata.stencildata.positionA[1][0][dim3])))))
//                    {
//                        structdata.stencildata.updateTargetIdx = dim3;
//                        reselectCandidate = true;
//                        break;
//                    }
//                }
//            }
//            if (!reselectCandidate)
//            {
//                structdata.stencildata.updateTargetIdx = Utils::findFirstFalse(structdata.stencildata.isRoiCandidated);
//            }
//            if (structdata.stencildata.updateTargetIdx == -1)
//            {
//                structdata.stencildata.updateTargetIdx = dim3Size - 1;
//            }
//            // get point
//            structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] = candidatePositionX;
//            structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] = candidatePositionY;
//            structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = false;
//            structdata.stencildata.isFourRoiReady = false;
//            structdata.stencildata.isRoiDetermined = false;
//            structdata.stencildata.isReadyToDraw = Utils::orArray(structdata.stencildata.isRoiCandidated);
//        }
//        else if (event->type() == QEvent::MouseMove)
//        {
//            if (!structdata.stencildata.isFourRoiReady)
//            {
//                // get point
//                structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] = candidatePositionX;
//                structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] = candidatePositionY;
//                structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = true;
//                structdata.stencildata.isRoiDetermined = false;
//                structdata.stencildata.isReadyToDraw = true;
//            }
//        }
//        else if (event->type() == QEvent::MouseButtonRelease)
//        {
//            if ((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] == candidatePositionX) ||
//                (structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] == candidatePositionY))     // remove roi with a simple click
//            {
//                structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = false;
//            }
//            else
//            {
//                // evaluate the roi is meaningful or not
//                bool isMeaningfulRoi = true;
//                if ((std::abs(structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] - candidatePositionX) < leastRoiBoxSize) ||
//                    (std::abs(structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] - candidatePositionY) < leastRoiBoxSize))    // meaningful roi size
//                {
//                    isMeaningfulRoi = false;
//                }
//                else
//                {
//                    // check whether candidate roi includes another roi or not
//                    for (int dim2 = 0; dim2 < dim2Size; ++dim2) {
//                        for (int dim3 = 0; dim3 < dim3Size; ++dim3) {
//                            if (!(dim3 == structdata.stencildata.updateTargetIdx) && (structdata.stencildata.isRoiCandidated[dim3]))
//                            {
//                                if ((((structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] <= structdata.stencildata.positionA[0][dim2][dim3]) && (structdata.stencildata.positionA[0][dim2][dim3] <= candidatePositionX))
//                                    ||((candidatePositionX <= structdata.stencildata.positionA[0][dim2][dim3]) && (structdata.stencildata.positionA[0][dim2][dim3] <= structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx])))
//                                   &&(((structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] <= structdata.stencildata.positionA[1][dim2][dim3]) && (structdata.stencildata.positionA[1][dim2][dim3] <= candidatePositionY))
//                                    ||((candidatePositionY <= structdata.stencildata.positionA[1][dim2][dim3]) && (structdata.stencildata.positionA[1][dim2][dim3] <= structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx]))))
//                                {
//                                    isMeaningfulRoi = false;
//                                    break;
//                                }
//                            }
//                        }
//                    }
//                }
//                // get point
//                if (isMeaningfulRoi)
//                {
//                    structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] = candidatePositionX;
//                    structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] = candidatePositionY;
//                    structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = true;
//                }
//                else
//                {
//                    structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = false;
//                }
//            }
//            structdata.stencildata.isFourRoiReady = Utils::andArray(structdata.stencildata.isRoiCandidated);
//            structdata.stencildata.isRoiDetermined = false;
//            structdata.stencildata.isReadyToDraw = true;
//        }
//        int x, y, width, height;
//        x = structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx];
//        y = structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx];
//        width = structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx];
//        height = structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx];
//        if (structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] > structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx])
//        {
//            x = structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx];
//            width = structdata.stencildata.positionA[0][0][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx];
//        }
//        if (structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] > structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx])
//        {
//            y = structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx];
//            height = structdata.stencildata.positionA[1][0][structdata.stencildata.updateTargetIdx] - structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx];
//        }
//        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].x = x;
//        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].y = y;
//        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].width = width;
//        structdata.stencildata.rectInfo[structdata.stencildata.updateTargetIdx].height = height;
//        structdata.stencildata.isMaskExist = false;
//    }
//    if (event->type() == QEvent::MouseButtonRelease)
//    {
//        structdata.mouseButtonStatus = Qt::NoButton;
//    }
//    return QMainWindow::eventFilter(obj, event);
//}


// refactorying
bool MainWindow::eventFilter(QObject* obj, QEvent* event) {
    if (obj != ui->mainGraphicsView->viewport()) return QMainWindow::eventFilter(obj, event);

    const auto mouseEvent = static_cast<QMouseEvent*>(event);
    switch (mouseEvent->button()) {
        case Qt::LeftButton:
            handleLeftMouseButtonEvent(mouseEvent);
           // break;
        case Qt::RightButton:
            handleRightMouseButtonEvent(mouseEvent);
           // break;
        default:
            break;
    }

    if (event->type() == QEvent::MouseButtonRelease) {
        structdata.mouseButtonStatus = Qt::NoButton;
    }
    return QMainWindow::eventFilter(obj, event);
}



void MainWindow::handleLeftMouseButtonEvent(QMouseEvent* mouseEvent) {
    // OCR 영역 처리 로직
    switch (mouseEvent->type()) {
        case QEvent::MouseButtonPress:
            startOCRBoxSelection(mouseEvent->pos());
            break;
        case QEvent::MouseMove:
            endOCRBoxSelection(mouseEvent->pos());
            calculateOCRBox(); // OCR 영역 최종 결정
            break;
        case QEvent::MouseButtonRelease:
            endOCRBoxSelection(mouseEvent->pos());
            calculateOCRBox(); // OCR 영역 최종 결정
            break;
        default:
            break;
    }
}


void MainWindow::startOCRBoxSelection(const QPoint& startPos) {
    // OCR 데이터 초기화 및 시작점 설정
    structdata.ocrdata.positionX1 = startPos.x();
    structdata.ocrdata.positionY1 = startPos.y();
    structdata.ocrdata.isRoiCandidated = true; // 후보로 지정
    structdata.ocrdata.isRoiDetermined = false; // 아직 최종 결정되지 않음
    structdata.ocrdata.isReadyToDraw = false;
    ui->depthDisplaylabel->setText(QString::fromStdString(""));
}


void MainWindow::endOCRBoxSelection(const QPoint& endPos) {
    // OCR 종료점 업데이트
    structdata.ocrdata.positionX2 = endPos.x();
    structdata.ocrdata.positionY2 = endPos.y();
    structdata.ocrdata.isRoiCandidated = true;
    structdata.ocrdata.isRoiDetermined = false;
    structdata.ocrdata.isReadyToDraw = true;
}


void MainWindow::calculateOCRBox() {
    // 시작점과 끝점 사이의 좌표를 기반으로 실제 x, y 위치와 너비, 높이를 계산
    int x = std::min(structdata.ocrdata.positionX1, structdata.ocrdata.positionX2);
    int y = std::min(structdata.ocrdata.positionY1, structdata.ocrdata.positionY2);
    int width = std::abs(structdata.ocrdata.positionX2 - structdata.ocrdata.positionX1);
    int height = std::abs(structdata.ocrdata.positionY2 - structdata.ocrdata.positionY1);
    // 계산된 위치와 크기를 OCR 사각박스 정보에 저장
    structdata.ocrdata.rectInfo.x = x;
    structdata.ocrdata.rectInfo.y = y;
    structdata.ocrdata.rectInfo.width = width;
    structdata.ocrdata.rectInfo.height = height;
}

void MainWindow::handleRightMouseButtonEvent(QMouseEvent* mouseEvent) {
    switch (mouseEvent->type()) {
        case QEvent::MouseButtonPress:
            qDebug() << "Mouse Press";
//            startStencilBoxSelectionOrRemoveStencilBox(mouseEvent->pos());
            break;
        case QEvent::MouseMove:
            qDebug() <<"Move";
//            endStencilBoxSelection(mouseEvent->pos());
            break;
        case QEvent::MouseButtonRelease:
            qDebug() << "release";
          //  updateStencilBoxEnd(x, y, mouseEvent->type() == QEvent::MouseButtonRelease);
            break;
        default:
        break;
    }
}


void MainWindow::startStencilBoxSelectionOrRemoveStencilBox(const QPoint& pos) {
    // 클릭한 위치가 기존 박스 내에 있는지 확인하고, 해당 박스가 있다면 취소
    int existingBoxIndex = findStencilBoxIndexAtPoint(pos);

    qDebug() << existingBoxIndex << "existingBoxIndex";
//    if (existingBoxIndex != -1) {
//        // 기존 박스 내에서 선택된 경우, 해당 박스를 취소
//        structdata.stencildata.isRoiCandidated[existingBoxIndex] = false;
//        structdata.stencildata.isReadyToDraw = false; // 그리기 상태 업데이트
//    } else {
        // 그려놓은 박스의 갯수가 4개 이하일 때만 새로 생성
        if (countActiveStencilBoxes() < dim3Size) {
            // 새 박스 시작
            startNewStencilBoxAt(pos);
//        }
    }
}

int MainWindow::findStencilBoxIndexAtPoint(const QPoint& point) {
    for (int i = 0; i < dim3Size; ++i) {
        if (structdata.stencildata.isRoiCandidated[i]) {
            // 박스 영역 안에 포인트가 있는지 직접 여기에서 확인
            bool withinBox = point.x() >= structdata.stencildata.positionA[0][0][i] &&
                             point.x() <= structdata.stencildata.positionA[0][1][i] &&
                             point.y() >= structdata.stencildata.positionA[1][0][i] &&
                             point.y() <= structdata.stencildata.positionA[1][1][i];

            if (withinBox) {
                return i; // 클릭한 위치가 박스 내에 있음
            }
        }
    }
    return -1; // 클릭한 위치에 박스 없음
}

int MainWindow::countActiveStencilBoxes() {
    return std::count(structdata.stencildata.isRoiCandidated.begin(), structdata.stencildata.isRoiCandidated.end(), true);
}

void MainWindow::startNewStencilBoxAt(const QPoint& startPos) {
    // 새로운 박스 시작 로직
    for (int i = 0; i < dim3Size; ++i) {
        if (!structdata.stencildata.isRoiCandidated[i]) {

            structdata.stencildata.positionA[0][0][i] = startPos.x();
            structdata.stencildata.positionA[1][0][i] = startPos.y();
            structdata.stencildata.positionA[0][1][i] = startPos.x()-10;
            structdata.stencildata.positionA[1][1][i] = startPos.y()-10;
            structdata.stencildata.isRoiCandidated[i] = true; // 박스 활성화
            structdata.stencildata.isReadyToDraw = true; // 그리기 준비 상태
            qDebug() << structdata.stencildata.positionA[0][0][i] << "x";
            qDebug() << structdata.stencildata.positionA[1][0][i] << "y";

            break; // 첫번째 비활성화된 박스를 활성화하고 반복 종료


        }
    }
}


void MainWindow::endStencilBoxSelection(const QPoint& endPos) {
    qDebug() << "This is in endStencilBox";
    for (int i = 0; i < dim3Size; ++i)
    {
        qDebug() << "This is in endStencilBox";
        if (!structdata.stencildata.isRoiCandidated[i])
        {
            qDebug() << "endStencilBoxSelection " << structdata.stencildata.isRoiCandidated[i];
            structdata.stencildata.positionA[0][1][structdata.stencildata.updateTargetIdx] = endPos.x();
            structdata.stencildata.positionA[1][1][structdata.stencildata.updateTargetIdx] = endPos.y();
            structdata.stencildata.isRoiCandidated[structdata.stencildata.updateTargetIdx] = true;
            structdata.stencildata.isRoiDetermined = false;
            structdata.stencildata.isReadyToDraw = true;
        }
    }
}







//    // 스텐실 영역 처리 로직
//    switch (mouseEvent->type()) {
//        case QEvent::MouseButtonPress:
//            initStencilBoxSelection(mouseEvent->pos());
//            break;
//        case QEvent::MouseMove:
//            finalizeStencilBoxSelection(mouseEvent->pos());
//            calculateStencilBox();
//            break;
//        case QEvent::MouseButtonRelease:
//            finalizeStencilBoxSelection(mouseEvent->pos());
//            calculateStencilBox();
//            break;
//        default:
//            break;
//    }

//}


//void MainWindow::initStencilBoxSelection(const QPoint& startPos) {
//    int targetIdx = determineStencilTarget(startPos);
//    // 스텐실 데이터 초기화 및 시작점 설정
//    structdata.stencildata.positionA[0][0][targetIdx] = startPos.x();
//    structdata.stencildata.positionA[1][0][targetIdx] = startPos.y();
//    // 다른 상태 초기화
//    structdata.stencildata.isRoiCandidated[targetIdx] = true;
//    structdata.stencildata.isRoiDetermined = false;
//}

//void MainWindow::finalizeStencilBoxSelection(const QPoint& endPos) {
//    // 스텐실 영역 설정 및 상태 업데이트
//    int targetIdx = structdata.stencildata.updateTargetIdx;
//    structdata.stencildata.positionA[0][1][targetIdx] = endPos.x();
//    structdata.stencildata.positionA[1][1][targetIdx] = endPos.y();
//    structdata.stencildata.isReadyToDraw = true;
//}

//void MainWindow::calculateStencilBox() {
//    int targetIdx = structdata.stencildata.updateTargetIdx;
//    // 최종 영역 계산 및 저장
//    int x = std::min(structdata.stencildata.positionA[0][0][targetIdx], structdata.stencildata.positionA[0][1][targetIdx]);
//    int y = std::min(structdata.stencildata.positionA[1][0][targetIdx], structdata.stencildata.positionA[1][1][targetIdx]);
//    int width = std::abs(structdata.stencildata.positionA[0][1][targetIdx] - structdata.stencildata.positionA[0][0][targetIdx]);
//    int height = std::abs(structdata.stencildata.positionA[1][1][targetIdx] - structdata.stencildata.positionA[1][0][targetIdx]);

//    structdata.stencildata.rectInfo[targetIdx] = {x, y, width, height};
//    structdata.stencildata.isRoiDetermined = true; // 최종 결정
//}

//int MainWindow::determineStencilTarget(const QPoint& pos) {
//    // 기존 영역 내에서 선택이 이루어지는지 확인하고 적절한 targetIdx 결정 로직
//    // (예시 로직은 생략됨, 실제 구현에 따라 달라질 수 있음)
//    return Utils::findFirstFalse(structdata.stencildata.isRoiCandidated); // 단순 예시
//}


//void MainWindow::initializeStencilData(const QPoint& startPos) {
//    // 스텐실 데이터 초기화 및 시작점 설정
//    int targetIdx = findTargetStencilArea(startPos.x(), startPos.y());
//    structdata.stencildata.updateTargetIdx = targetIdx;
//    structdata.stencildata.positionA[0][0][targetIdx] = startPos.x();
//    structdata.stencildata.positionA[1][0][targetIdx] = startPos.y();
//    structdata.stencildata.isRoiCandidated[targetIdx] = false;
//}

//void MainWindow::updateStencilEndPoint(const QPoint& endPos) {
//    // 스텐실 종료점 업데이트
//    if (!structdata.stencildata.isFourRoiReady) {
//        int targetIdx = structdata.stencildata.updateTargetIdx;
//        structdata.stencildata.positionA[0][1][targetIdx] = endPos.x();
//        structdata.stencildata.positionA[1][1][targetIdx] = endPos.y();
//        structdata.stencildata.isRoiCandidated[targetIdx] = true;
//    }
//}

//void MainWindow::finalizeStencilArea(const QPoint& endPos) {
//    // 스텐실 영역 최종 결정
//    int targetIdx = structdata.stencildata.updateTargetIdx;
//    // 영역 유효성 검사 및 최종화 로직
//    validateAndFinalizeStencilArea(targetIdx, endPos);
//}






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
    /// draw US image
    // update drawing index
    int nextDrawingIndex = (structdata.currentDrawingIndex + 1) % ((int)structdata.pngFileNames.size());
    // load new image, then draw
    cv::Mat cvImage = cv::imread(structdata.pngFileNames[nextDrawingIndex]);
    cv::Mat cvOrgImage = cvImage;
    cv::resize(cvImage, cvImage, cv::Size(structdata.scaledOrgImg.cols, structdata.scaledOrgImg.rows));
    cv::cvtColor(cvImage, cvImage, cv::COLOR_BGR2RGB);
    if ((structdata.stencildata.isMaskExist) && (structdata.stencildata.isStencilApplied))
    {
        cv::Mat maskedCvImage;
        cv::multiply(cvImage, structdata.scaledOrgImgMask, maskedCvImage);
        QImage qtImage(maskedCvImage.data, maskedCvImage.cols, maskedCvImage.rows, maskedCvImage.step, QImage::Format_RGB888);
        delete pixmapItem;
        pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
    }
    else
    {
        QImage qtImage(cvImage.data, cvImage.cols, cvImage.rows, cvImage.step, QImage::Format_RGB888);
        delete pixmapItem;
        pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
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
            QGraphicsRectItem* rectItem = new QGraphicsRectItem(structdata.ocrdata.positionX1,
                                                                structdata.ocrdata.positionY1,
                                                                structdata.ocrdata.positionX2-structdata.ocrdata.positionX1,
                                                                structdata.ocrdata.positionY2-structdata.ocrdata.positionY1);
            if (!structdata.ocrdata.isRoiDetermined)
            {
                rectItem->setPen(QPen(Qt::red, 1));
            }
            else
            {
                rectItem->setPen(QPen(Qt::green, 2));
            }
            ui->mainGraphicsView->scene()->addItem(rectItem);  // structdata.ocrdata.rectItems.push_back(rectItem);
        }
        // draw rect rubber band for stencil
        if (structdata.stencildata.isReadyToDraw)
        {
            for (int dim3 = 0; dim3 < dim3Size; ++dim3)
            {
                if (structdata.stencildata.isRoiCandidated[dim3])
                {
                    QGraphicsRectItem* rectItem = new QGraphicsRectItem(structdata.stencildata.positionA[0][0][dim3],
                                                                        structdata.stencildata.positionA[1][0][dim3],
                                                                        structdata.stencildata.positionA[0][1][dim3]-structdata.stencildata.positionA[0][0][dim3],
                                                                        structdata.stencildata.positionA[1][1][dim3]-structdata.stencildata.positionA[1][0][dim3]);
                    if (!structdata.stencildata.isRoiDetermined)
                    {
                        rectItem->setPen(QPen(Qt::yellow, 1));
                    }
                    else
                    {
                        rectItem->setPen(QPen(Qt::blue, 2));
                    }
                    ui->mainGraphicsView->scene()->addItem(rectItem);
                }
            }
        }
    }

    /// OCR inference
    if (structdata.ocrdata.isRoiDetermined)
    {
        // get subROI
        cv::Mat subRoi_img = structdata.orgImg(structdata.ocrdata.recoveredRectInfo);
        // conduct Tesseract OCR
        std::string outText = runOCR(subRoi_img);
        // write the result at textbox
        ui->depthDisplaylabel->setText(QString::fromStdString(outText));
        ui->depthDisplaylabel->setAlignment(Qt::AlignVCenter | Qt::AlignHCenter);
    }
}

//if (std::count(structdata.stencildata.isRoiCandidated.begin(), structdata.stencildata.isRoiCandidated.end(), true) < 4)
