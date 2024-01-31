#include "mainwindow.h"
#include "./ui_mainwindow.h"
#include <QOpenGLWidget>
#include <QMouseEvent>
#include <QImage>
#include <QOpenGLTexture>
#include <opencv2/opencv.hpp>

class OpenGLCanvas : public QOpenGLWidget
{
public:
    OpenGLCanvas(QWidget *parent = nullptr)
        : QOpenGLWidget(parent),
          texture(nullptr)
    {
    }

    ~OpenGLCanvas()
    {
        makeCurrent(); // Ensure the correct context is current
        delete texture;
        doneCurrent();
    }

    void initializeGL() override
    {
        initializeOpenGLFunctions();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Load image and create OpenGL texture
        loadImage();
        createTexture();
    }

    void resizeGL(int w, int h) override
    {
        glViewport(0, 0, w, h);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, w, h, 0, -1, 1);
        glMatrixMode(GL_MODELVIEW);
    }

    void paintGL() override
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Draw image texture
        if (texture)
        {
            texture->bind();
            glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0);
            glVertex2i(0, 0);
            glTexCoord2f(1, 0);
            glVertex2i(width(), 0);
            glTexCoord2f(1, 1);
            glVertex2i(width(), height());
            glTexCoord2f(0, 1);
            glVertex2i(0, height());
            glEnd();
            texture->release();
        }

        // Draw rectangles
        for (const RectInfo &rectInfo : userdata.rectInfos)
        {
            drawRect(rectInfo.x, rectInfo.y, rectInfo.width, rectInfo.height, rectInfo.color);
        }
    }

protected:
    void mousePressEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton)
        {
            rectInfo.x = event->pos().x();
            rectInfo.y = event->pos().y();
            rectInfo.width = 0;
            rectInfo.height = 0;
            rectInfo.color = Qt::red;
            update();
        }
    }

    void mouseMoveEvent(QMouseEvent *event) override
    {
        if (event->buttons() & Qt::LeftButton)
        {
            rectInfo.width = event->pos().x() - rectInfo.x;
            rectInfo.height = event->pos().y() - rectInfo.y;
            update();
        }
    }

    void mouseReleaseEvent(QMouseEvent *event) override
    {
        if (event->button() == Qt::LeftButton)
        {
            userdata.rectInfos.push_back(rectInfo);
            update();
        }
    }

private:
    void drawRect(int x, int y, int width, int height, const QColor &color)
    {
        glColor4f(color.redF(), color.greenF(), color.blueF(), color.alphaF());
        glBegin(GL_QUADS);
        glVertex2i(x, y);
        glVertex2i(x + width, y);
        glVertex2i(x + width, y + height);
        glVertex2i(x, y + height);
        glEnd();
    }

    void loadImage()
    {
        QString imagePath = "/home/artixmed/project_build/tesseract_OCR_C_QT/Tesseract_OCR_C/resource/image_00001.png";
        cv::Mat cvImage = cv::imread(imagePath.toStdString());
        cv::cvtColor(cvImage, cvImage, cv::COLOR_BGR2RGB);
        QImage qtImage(cvImage.data, cvImage.cols, cvImage.rows, cvImage.step, QImage::Format_RGB888);
        image = qtImage;
    }

    void createTexture()
    {
        makeCurrent(); // Ensure the correct context is current
        if (texture)
            delete texture;

        texture = new QOpenGLTexture(image);
        doneCurrent();
    }

private:
    RectInfo rectInfo;
    QImage image;
    QOpenGLTexture *texture;
};

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent),
      ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    // Set up your OpenGLCanvas
    OpenGLCanvas *openGLCanvas = new OpenGLCanvas(this);
    setCentralWidget(openGLCanvas);

    // ...

    // Event filter, mouse events callback
    openGLCanvas->installEventFilter(this);

    // Connect push button event callback
    connect(ui->pushButton, &QPushButton::clicked, this, &MainWindow::onPushButtonClicked);
}

// ...
