#include <QCoreApplication>
#include <QDebug>
#include <QLabel>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    qDebug() << "Hello, Qt on Linux!";

    return a.exec();
}
