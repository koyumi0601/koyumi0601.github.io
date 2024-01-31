#include <QImage>
#include <QString>
#include <QtWidgets/QApplication>
#include <QtOpenGL/QGLWidget>
#include <QtGui/QOpenGLFunctions>
#include <GL/gl.h> // OpenGL 헤더
#include <QtDebug>

class OpenGLImageWidget : public QGLWidget, protected QOpenGLFunctions
{
public:
    OpenGLImageWidget(QWidget *parent = nullptr) : QGLWidget(parent) {}

protected:
    void initializeGL() override
    {
        initializeOpenGLFunctions();
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    }

    void paintGL() override
    {
        // 이미지 파일 경로 설정
        QString imagePath = "/home/ko/Documents/Practice_class/resource/image_00001.png";

        // QImage를 사용하여 이미지 로드
        QImage loadedImage(imagePath);

        if (loadedImage.isNull())
        {
            // 이미지 로드 실패 시 처리
            qDebug() << "Failed to load the image.";
            return;
        }
        else
        {
            qDebug() << "Success to load the image.";
        }

        // 텍스처를 사용하여 이미지를 렌더링하는 코드
        GLuint textureID = bindImageToTexture(loadedImage);

        // 텍스처를 이용하여 이미지를 렌더링
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glBegin(GL_QUADS);
        glTexCoord2d(0.0, 0.0);
        glVertex2d(-1.0, -1.0);
        glTexCoord2d(1.0, 0.0);
        glVertex2d(1.0, -1.0);
        glTexCoord2d(1.0, 1.0);
        glVertex2d(1.0, 1.0);
        glTexCoord2d(0.0, 1.0);
        glVertex2d(-1.0, 1.0);
        glEnd();
        glDisable(GL_TEXTURE_2D);
    }

private:
    GLuint bindImageToTexture(const QImage &image)
    {
        GLuint textureID;
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);

        // 이미지 데이터를 텍스처로 업로드
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width(), image.height(), 0, GL_BGRA, GL_UNSIGNED_BYTE, image.bits());

        // 텍스처 파라미터 설정
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        return textureID;
    }
};

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    OpenGLImageWidget w;
    w.show();

    return a.exec();
}