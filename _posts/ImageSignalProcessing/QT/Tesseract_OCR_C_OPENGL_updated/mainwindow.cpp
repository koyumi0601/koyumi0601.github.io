#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

#include <QGraphicsPixmapItem>
#include <QPixmap>
#include <QGraphicsView>
#include <QDebug>

#include <QtWidgets>
#include <QWidget>
#include <QOpenGLWidget>
#include <QImage>

UserData userdata;

// set image path of an example image to run this system

//QString imagePath = "/home/artixmed/project_build/tesseract_OCR_C_QT/Tesseract_OCR_C/resource/image_00001.png";
QString imagePath = "/home/ko/Documents/Tesseract_OCR_C_OPENGL/resource/image_00001.png";

int currentWidth = 1024;
int currentHeight = 608;


class OpenGLWidget : public QOpenGLWidget
{
public:
    explicit OpenGLWidget(QWidget* parent = nullptr) : QOpenGLWidget(parent)
    {
        setMouseTracking(true); // get moving event of mouse point
    }
    void setImage(const QImage& image)
    {
        qDebug() << "paintGL::setImage called";
        imageToDraw = image;
        update();  // Trigger a repaint
    }
protected:
    void initializeGL() override
    {
        // OpenGL initialization
    }
    void resizeGL(int w, int h) override
    {
        // OpenGL resize (if needed)
    }
    void paintGL() override
    {
        qDebug() << "paintGL called";
        QPainter painter(this);
        painter.drawImage(0, 0, imageToDraw);
        // Drawing (if needed)
    }
    void mousePressEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::LeftButton) // push down of mouse left button
        {
            userdata.leftPressed = true;

            QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
            userdata.rectInfo.x = mouseEvent->pos().x();
            userdata.rectInfo.y = mouseEvent->pos().y();

            QPainter painter(&imageToDraw);
            painter.drawImage(0, 0, userdata.qScaledOrgImg);
            update();

            qDebug() << "Left button pressed at" << event->pos();
            std::cout << "x=" <<userdata.rectInfo.x << ", y=" <<userdata.rectInfo.y << "\n" << std::endl;

        }
    }
    void mouseMoveEvent(QMouseEvent* event) override
    {
        if (userdata.leftPressed)
        {
//            setImage(userdata.qScaledOrgImg);
//            imageToDraw = userdata.qScaledOrgImg;
//            update();
            qDebug() << "Mouse dragging with Left button down at" << event->pos();
        }
    }
    void mouseReleaseEvent(QMouseEvent* event) override
    {
        if (event->button() == Qt::LeftButton) // release up of mouse left button
        {
            userdata.leftPressed = false;

            QMouseEvent* mouseEvent = static_cast<QMouseEvent*>(event);
            int width = mouseEvent->pos().x() - userdata.rectInfo.x;
            int height = mouseEvent->pos().y() - userdata.rectInfo.y;
            if (width < 0)
            {
                width = width * (-1);
                userdata.rectInfo.x = mouseEvent->pos().x();
            }
            if (height < 0)
            {
                height = height * (-1);
                userdata.rectInfo.y = mouseEvent->pos().y();
            }
            userdata.rectInfo.width = width;
            userdata.rectInfo.height = height;

            // remove previous rec, then draw rect
            QPainter painter(&imageToDraw);




            QPen pen(Qt::red);
            pen.setWidth(1);
            painter.setPen(pen);
            painter.drawRect(userdata.rectInfo.x, userdata.rectInfo.y, userdata.rectInfo.width, userdata.rectInfo.height);



            //setImage(imageToDraw);

            update();

            userdata.isThereCandidate = true;
            userdata.isRoiDetermined = false;
            userdata.prevRectInfo = userdata.rectInfo;

            qDebug() << "Left button released at" << event->pos();
        }
    }
    void paintEvent(QPaintEvent* event) override
    {
        Q_UNUSED(event);
        QPainter painter(this);
        painter.drawImage(0, 0, imageToDraw);
    }
private:
    QImage imageToDraw;
    //UserData userdata;
};



MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QImage qtImage(imagePath);
    QImage qtScaledImage = qtImage.scaled(currentWidth, currentHeight, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    // Initialize OpenGLWidget and set the image
    OpenGLWidget* openGLWidget = new OpenGLWidget(this);
    openGLWidget->setImage(qtScaledImage);

    // Set up layout
    QVBoxLayout* layout = new QVBoxLayout(ui->centralwidget);
    layout->addWidget(openGLWidget);
    ui->centralwidget->setLayout(layout);
    openGLWidget->setFixedSize(qtScaledImage.width(), qtScaledImage.height());


    // init a status of userdata
    userdata.qOrgImg = qtImage;
    userdata.qScaledOrgImg = qtScaledImage;
    userdata.isRoiDetermined = false;
    userdata.isThereCandidate = false;
    userdata.widthFactor = (float)qtImage.width() / (float)qtScaledImage.width();
    userdata.heightFactor = (float)qtImage.height() / (float)qtScaledImage.height();
    userdata.leftPressed = false;

    // push button event callback
    connect(ui->pushButton, &QPushButton::clicked, this, &MainWindow::onPushButtonClicked);

}
MainWindow::~MainWindow()
{
    delete ui;
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

//        // inference of an updatedImg using tesseract
//        // init Tesseract OCR
//        tesseract::TessBaseAPI* ocr_api = new tesseract::TessBaseAPI();
//        if (ocr_api->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY) != 0)
//        {
//            std::cerr << "Could not initialize Tesseract." << std::endl;
//        }
//        ocr_api->SetPageSegMode(tesseract::PSM_SINGLE_LINE);
//        ocr_api->SetVariable("tessedit_char_whitelist", "0123456789.");
//        ocr_api->SetImage(subRoi_img.data, subRoi_img.cols, subRoi_img.rows, 3, subRoi_img.step);
//        ocr_api->SetSourceResolution(250); // updatedImg resolution (warning: invalid resolution 0 dpi. Using 70 instead. Estimatin resolution as 197)
//        std::string outText;
//        outText = std::string(ocr_api->GetUTF8Text());
    }
}
