/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.12.12
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QGraphicsView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralwidget;
    QPushButton *ocrRoiPushButton;
    QLabel *depthDisplaylabel;
    QGraphicsView *mainGraphicsView;
    QPushButton *confirmStencilRoiPushButton;
    QPushButton *generateStencilRoiPushButton;
    QComboBox *probeTypeComboBox;
    QCheckBox *applyStencilCheckBox;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1024, 682);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        ocrRoiPushButton = new QPushButton(centralwidget);
        ocrRoiPushButton->setObjectName(QString::fromUtf8("ocrRoiPushButton"));
        ocrRoiPushButton->setGeometry(QRect(760, 610, 151, 41));
        ocrRoiPushButton->setStyleSheet(QString::fromUtf8("background-color: rgb(186, 189, 182);"));
        depthDisplaylabel = new QLabel(centralwidget);
        depthDisplaylabel->setObjectName(QString::fromUtf8("depthDisplaylabel"));
        depthDisplaylabel->setGeometry(QRect(920, 610, 91, 41));
        depthDisplaylabel->setStyleSheet(QString::fromUtf8("border-color: rgb(0, 0, 0);"));
        depthDisplaylabel->setFrameShape(QFrame::Box);
        mainGraphicsView = new QGraphicsView(centralwidget);
        mainGraphicsView->setObjectName(QString::fromUtf8("mainGraphicsView"));
        mainGraphicsView->setGeometry(QRect(0, 0, 1024, 576));
        confirmStencilRoiPushButton = new QPushButton(centralwidget);
        confirmStencilRoiPushButton->setObjectName(QString::fromUtf8("confirmStencilRoiPushButton"));
        confirmStencilRoiPushButton->setGeometry(QRect(220, 610, 151, 41));
        confirmStencilRoiPushButton->setStyleSheet(QString::fromUtf8("background-color: rgb(186, 189, 182);"));
        generateStencilRoiPushButton = new QPushButton(centralwidget);
        generateStencilRoiPushButton->setObjectName(QString::fromUtf8("generateStencilRoiPushButton"));
        generateStencilRoiPushButton->setGeometry(QRect(380, 610, 151, 41));
        generateStencilRoiPushButton->setStyleSheet(QString::fromUtf8("background-color: rgb(186, 189, 182);"));
        probeTypeComboBox = new QComboBox(centralwidget);
        probeTypeComboBox->addItem(QString());
        probeTypeComboBox->addItem(QString());
        probeTypeComboBox->setObjectName(QString::fromUtf8("probeTypeComboBox"));
        probeTypeComboBox->setGeometry(QRect(10, 610, 91, 41));
        applyStencilCheckBox = new QCheckBox(centralwidget);
        applyStencilCheckBox->setObjectName(QString::fromUtf8("applyStencilCheckBox"));
        applyStencilCheckBox->setGeometry(QRect(110, 610, 111, 41));
        applyStencilCheckBox->setChecked(true);
        MainWindow->setCentralWidget(centralwidget);
        statusbar = new QStatusBar(MainWindow);
        statusbar->setObjectName(QString::fromUtf8("statusbar"));
        MainWindow->setStatusBar(statusbar);

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        ocrRoiPushButton->setText(QApplication::translate("MainWindow", "Confirm OCR ROI", nullptr));
        depthDisplaylabel->setText(QString());
        confirmStencilRoiPushButton->setText(QApplication::translate("MainWindow", "Confirm Stencil ROI", nullptr));
        generateStencilRoiPushButton->setText(QApplication::translate("MainWindow", "Generate Stencil ROI", nullptr));
        probeTypeComboBox->setItemText(0, QApplication::translate("MainWindow", "Convex", nullptr));
        probeTypeComboBox->setItemText(1, QApplication::translate("MainWindow", "Linear", nullptr));

        applyStencilCheckBox->setText(QApplication::translate("MainWindow", "Apply stencil", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
