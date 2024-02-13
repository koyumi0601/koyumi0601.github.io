#include "mainwindow.h"
#include <QApplication>
#include <QCoreApplication>

int fixedWindowWidth = 1024;
int fixedWindowHeight = 674;

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow mainwindow(QCoreApplication::arguments());
    mainwindow.setFixedSize(fixedWindowWidth, fixedWindowHeight);
    mainwindow.show();
    return app.exec();
}
