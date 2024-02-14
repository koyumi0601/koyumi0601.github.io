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
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTableWidget>
#include <QtWidgets/QTextEdit>
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
    QTableWidget *monitoringTableWidget;
    QPushButton *pushItemPushButton;
    QPushButton *deleteItemPushButton;
    QTextEdit *systemNameTextEditBox;
    QTextEdit *probeNameTextEditBox;
    QLabel *systemNameTextBoxLabel;
    QLabel *probeNameTextBoxLabel;
    QStatusBar *statusbar;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QString::fromUtf8("MainWindow"));
        MainWindow->resize(1299, 706);
        centralwidget = new QWidget(MainWindow);
        centralwidget->setObjectName(QString::fromUtf8("centralwidget"));
        ocrRoiPushButton = new QPushButton(centralwidget);
        ocrRoiPushButton->setObjectName(QString::fromUtf8("ocrRoiPushButton"));
        ocrRoiPushButton->setGeometry(QRect(1040, 620, 151, 41));
        ocrRoiPushButton->setStyleSheet(QString::fromUtf8("background-color: rgb(186, 189, 182);"));
        depthDisplaylabel = new QLabel(centralwidget);
        depthDisplaylabel->setObjectName(QString::fromUtf8("depthDisplaylabel"));
        depthDisplaylabel->setGeometry(QRect(1200, 620, 91, 41));
        depthDisplaylabel->setStyleSheet(QString::fromUtf8("border-color: rgb(0, 0, 0);"));
        depthDisplaylabel->setFrameShape(QFrame::Box);
        mainGraphicsView = new QGraphicsView(centralwidget);
        mainGraphicsView->setObjectName(QString::fromUtf8("mainGraphicsView"));
        mainGraphicsView->setGeometry(QRect(276, 0, 1024, 576));
        confirmStencilRoiPushButton = new QPushButton(centralwidget);
        confirmStencilRoiPushButton->setObjectName(QString::fromUtf8("confirmStencilRoiPushButton"));
        confirmStencilRoiPushButton->setGeometry(QRect(490, 620, 151, 41));
        confirmStencilRoiPushButton->setStyleSheet(QString::fromUtf8("background-color: rgb(186, 189, 182);"));
        generateStencilRoiPushButton = new QPushButton(centralwidget);
        generateStencilRoiPushButton->setObjectName(QString::fromUtf8("generateStencilRoiPushButton"));
        generateStencilRoiPushButton->setGeometry(QRect(650, 620, 151, 41));
        generateStencilRoiPushButton->setStyleSheet(QString::fromUtf8("background-color: rgb(186, 189, 182);"));
        probeTypeComboBox = new QComboBox(centralwidget);
        probeTypeComboBox->addItem(QString());
        probeTypeComboBox->addItem(QString());
        probeTypeComboBox->setObjectName(QString::fromUtf8("probeTypeComboBox"));
        probeTypeComboBox->setGeometry(QRect(280, 620, 91, 41));
        applyStencilCheckBox = new QCheckBox(centralwidget);
        applyStencilCheckBox->setObjectName(QString::fromUtf8("applyStencilCheckBox"));
        applyStencilCheckBox->setGeometry(QRect(380, 620, 111, 41));
        applyStencilCheckBox->setChecked(true);
        monitoringTableWidget = new QTableWidget(centralwidget);
        if (monitoringTableWidget->columnCount() < 3)
            monitoringTableWidget->setColumnCount(3);
        QFont font;
        font.setPointSize(8);
        QTableWidgetItem *__qtablewidgetitem = new QTableWidgetItem();
        __qtablewidgetitem->setFont(font);
        monitoringTableWidget->setHorizontalHeaderItem(0, __qtablewidgetitem);
        QTableWidgetItem *__qtablewidgetitem1 = new QTableWidgetItem();
        __qtablewidgetitem1->setFont(font);
        monitoringTableWidget->setHorizontalHeaderItem(1, __qtablewidgetitem1);
        QTableWidgetItem *__qtablewidgetitem2 = new QTableWidgetItem();
        __qtablewidgetitem2->setFont(font);
        monitoringTableWidget->setHorizontalHeaderItem(2, __qtablewidgetitem2);
        monitoringTableWidget->setObjectName(QString::fromUtf8("monitoringTableWidget"));
        monitoringTableWidget->setGeometry(QRect(0, 10, 211, 567));
        pushItemPushButton = new QPushButton(centralwidget);
        pushItemPushButton->setObjectName(QString::fromUtf8("pushItemPushButton"));
        pushItemPushButton->setGeometry(QRect(217, 230, 51, 51));
        deleteItemPushButton = new QPushButton(centralwidget);
        deleteItemPushButton->setObjectName(QString::fromUtf8("deleteItemPushButton"));
        deleteItemPushButton->setGeometry(QRect(217, 290, 51, 51));
        systemNameTextEditBox = new QTextEdit(centralwidget);
        systemNameTextEditBox->setObjectName(QString::fromUtf8("systemNameTextEditBox"));
        systemNameTextEditBox->setGeometry(QRect(110, 590, 131, 41));
        probeNameTextEditBox = new QTextEdit(centralwidget);
        probeNameTextEditBox->setObjectName(QString::fromUtf8("probeNameTextEditBox"));
        probeNameTextEditBox->setGeometry(QRect(110, 640, 131, 41));
        systemNameTextBoxLabel = new QLabel(centralwidget);
        systemNameTextBoxLabel->setObjectName(QString::fromUtf8("systemNameTextBoxLabel"));
        systemNameTextBoxLabel->setGeometry(QRect(10, 600, 91, 17));
        probeNameTextBoxLabel = new QLabel(centralwidget);
        probeNameTextBoxLabel->setObjectName(QString::fromUtf8("probeNameTextBoxLabel"));
        probeNameTextBoxLabel->setGeometry(QRect(10, 650, 91, 17));
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
        QTableWidgetItem *___qtablewidgetitem = monitoringTableWidget->horizontalHeaderItem(0);
        ___qtablewidgetitem->setText(QApplication::translate("MainWindow", "system", nullptr));
        QTableWidgetItem *___qtablewidgetitem1 = monitoringTableWidget->horizontalHeaderItem(1);
        ___qtablewidgetitem1->setText(QApplication::translate("MainWindow", "probe", nullptr));
        QTableWidgetItem *___qtablewidgetitem2 = monitoringTableWidget->horizontalHeaderItem(2);
        ___qtablewidgetitem2->setText(QApplication::translate("MainWindow", "depth", nullptr));
        pushItemPushButton->setText(QApplication::translate("MainWindow", "<<", nullptr));
        deleteItemPushButton->setText(QApplication::translate("MainWindow", "X", nullptr));
        systemNameTextBoxLabel->setText(QApplication::translate("MainWindow", "system name", nullptr));
        probeNameTextBoxLabel->setText(QApplication::translate("MainWindow", "probe name", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
