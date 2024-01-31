#include "mainwindow.h"

#include <QApplication>
#include <QtGui/QGuiApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow mainwindow;
    mainwindow.show();
    return app.exec();
}
