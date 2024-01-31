#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QGraphicsView>
#include <QDebug>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>


UserData userdata;

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // get window info
    QSize currentWindowSize = ui->graphicsView->size();
    int currentWidth = currentWindowSize.width();
    int currentHeight = currentWindowSize.height();

    // set image path
    QString imagePath = "/home/ko/Documents/tesseract_OCR_C_QT/Tesseract_OCR_C/resource/image_00001.png";

    // load image through cv
    cv::Mat cvImage = cv::imread(imagePath.toStdString());
    cv::Mat cvOrgImage = cvImage.clone();
    cv::resize(cvImage, cvImage, cv::Size(currentWidth, currentHeight));
    QImage qtImage(cvImage.data, cvImage.cols, cvImage.rows, cvImage.step, QImage::Format_RGB888);


    // init a status of userdata
    userdata.orgImg = cvOrgImage.clone();
    userdata.scaledOrgImg = cvImage.clone();
    userdata.scaledupdatedImg = cvImage.clone();
    userdata.isRoiDetermined = false;
    userdata.isThereCandidate = false;
    userdata.widthFactor = (float)cvOrgImage.cols / (float)cvImage.cols;
    userdata.heightFactor = (float)cvOrgImage.rows / (float)cvImage.rows;

    // gen QGraphicsScene, then connect QGraphicsView
    QGraphicsScene* scene = new QGraphicsScene(this);
    ui->graphicsView->setScene(scene);
    QGraphicsPixmapItem* pixmapItem = new QGraphicsPixmapItem(QPixmap::fromImage(qtImage));
    scene->addItem(pixmapItem);

    // anti-alias
    ui->graphicsView->setRenderHint(QPainter::Antialiasing);
    ui->graphicsView->setRenderHint(QPainter::SmoothPixmapTransform);

    // remove scrollbar
    ui->graphicsView->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    ui->graphicsView->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);

    // event filter, mouse events callback
    ui->graphicsView->viewport()->installEventFilter(this); // ui->graphicsView->viewport()->setMouseTracking(true);

    // push button event callback
    connect(ui->pushButton, &QPushButton::clicked, this, &MainWindow::onPushButtonClicked);

}

MainWindow::~MainWindow()
{
    delete ui;
}

bool MainWindow::eventFilter(QObject* obj, QEvent* event)
{
    if (obj == ui->graphicsView->viewport() && event->type() == QEvent::MouseMove)
    {
        // Clear all existing QGraphicsRectItem items in the scene
        for (QGraphicsRectItem* rectItem : userdata.rectItems) {
            ui->graphicsView->scene()->removeItem(rectItem);
            delete rectItem;
        }
        userdata.rectItems.clear();

        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        int x = mouseEvent->position().x();
        int y = mouseEvent->position().y();

        int width = x - userdata.rectInfo.x;
        int height = y - userdata.rectInfo.y;
        if ((width < 0) && (height < 0))
        {
            width = width * (-1);
            height = height * (-1);
            QGraphicsRectItem* rectItem = new QGraphicsRectItem(userdata.rectInfo.x - width, userdata.rectInfo.y - height, width, height);
            rectItem->setPen(QPen(Qt::red, 1));
            ui->graphicsView->scene()->addItem(rectItem);
            userdata.rectItems.push_back(rectItem);
        }
        else if ((width < 0) && (height >= 0))
        {
            width = width * (-1);
            QGraphicsRectItem* rectItem = new QGraphicsRectItem(userdata.rectInfo.x - width, userdata.rectInfo.y, width, height);
            rectItem->setPen(QPen(Qt::red, 1));
            ui->graphicsView->scene()->addItem(rectItem);
            userdata.rectItems.push_back(rectItem);
        }
        else if ((width >= 0) && (height < 0))
        {
            height = height * (-1);
            QGraphicsRectItem* rectItem = new QGraphicsRectItem(userdata.rectInfo.x, userdata.rectInfo.y - height, width, height);
            rectItem->setPen(QPen(Qt::red, 1));
            ui->graphicsView->scene()->addItem(rectItem);
            userdata.rectItems.push_back(rectItem);
        }
        else
        {
            QGraphicsRectItem* rectItem = new QGraphicsRectItem(userdata.rectInfo.x, userdata.rectInfo.y, width, height);
            rectItem->setPen(QPen(Qt::red, 1));
            ui->graphicsView->scene()->addItem(rectItem);
            userdata.rectItems.push_back(rectItem);
        }
    }
    QGraphicsRectItem* rectItem;
    if (obj == ui->graphicsView->viewport() && event->type() == QEvent::MouseButtonPress)
    {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        userdata.rectInfo.x = mouseEvent->position().x();
        userdata.rectInfo.y = mouseEvent->position().y();
        // Clear all existing QGraphicsRectItem items in the scene
        for (QGraphicsRectItem* rectItem : userdata.rectItems) {
            ui->graphicsView->scene()->removeItem(rectItem);
            delete rectItem;
        }
        userdata.rectItems.clear();
    }
    else if (obj == ui->graphicsView->viewport() && event->type() == QEvent::MouseButtonRelease)
    {
        QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
        int width = mouseEvent->position().x() - userdata.rectInfo.x;
        int height = mouseEvent->position().y() - userdata.rectInfo.y;
        if (width < 0)
        {
            width = width * (-1);
            userdata.rectInfo.x = mouseEvent->position().x();
        }
        if (height < 0)
        {
            height = height * (-1);
            userdata.rectInfo.y = mouseEvent->position().y();
        }
        userdata.rectInfo.width = width;
        userdata.rectInfo.height = height;
        // draw rect
        QGraphicsRectItem* rectItem = new QGraphicsRectItem(userdata.rectInfo.x, userdata.rectInfo.y, userdata.rectInfo.width, userdata.rectInfo.height);
        rectItem->setPen(QPen(Qt::red, 1));
        ui->graphicsView->scene()->addItem(rectItem);
        userdata.rectItems.push_back(rectItem);
        userdata.isThereCandidate = true;
        userdata.isRoiDetermined = false;
    }
    return QMainWindow::eventFilter(obj, event);
}


void MainWindow::onPushButtonClicked()
{
    // Clear all existing QGraphicsRectItem items in the scene
    for (QGraphicsRectItem* rectItem : userdata.rectItems) {
        ui->graphicsView->scene()->removeItem(rectItem);
        delete rectItem;
    }
    userdata.rectItems.clear();

    QGraphicsRectItem* rectItem = new QGraphicsRectItem(userdata.rectInfo.x, userdata.rectInfo.y, userdata.rectInfo.width, userdata.rectInfo.height);
    rectItem->setPen(QPen(Qt::green, 2));
    ui->graphicsView->scene()->addItem(rectItem);
    userdata.rectItems.push_back(rectItem);
    userdata.isRoiDetermined = true;

    if (userdata.isThereCandidate && userdata.isRoiDetermined)
    {
        cv::Rect recoveredRectInfo;
        recoveredRectInfo.x = (int)((float)userdata.rectInfo.x * userdata.widthFactor + 0.5f);
        recoveredRectInfo.y = (int)((float)userdata.rectInfo.y * userdata.heightFactor + 0.5f);
        recoveredRectInfo.width = (int)((float)userdata.rectInfo.width * userdata.widthFactor + 0.5f);
        recoveredRectInfo.height = (int)((float)userdata.rectInfo.height * userdata.heightFactor + 0.5f);

        std::cout << "x = " << recoveredRectInfo.x << ", y = " << recoveredRectInfo.y << std::endl;
        std::cout << "width = " << recoveredRectInfo.width << ", height = " << recoveredRectInfo.height << std::endl;

        cv::Mat subRoi_img = userdata.orgImg(recoveredRectInfo);

        // // inference of an updatedImg using tesseract
        // // init Tesseract OCR
        // tesseract::TessBaseAPI* ocr_api = new tesseract::TessBaseAPI();
        // if (ocr_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY) != 0)
        // {
        //     std::cerr << "Could not initialize Tesseract." << std::endl;
        // }
        // ocr_api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
        // ocr_api->SetVariable("tessedit_char_whitelist", "0123456789.");
        // ocr_api->SetImage(subRoi_img.data, subRoi_img.cols, subRoi_img.rows, 3, subRoi_img.step);
        // ocr_api->SetSourceResolution(250); // updatedImg resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
        // std::string outText;
        // outText = std::string(ocr_api->GetUTF8Text());
    }
}
