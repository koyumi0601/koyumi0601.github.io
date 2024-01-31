#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QMouseEvent>
#include <QGraphicsRectItem>
#include <opencv2/opencv.hpp>

struct UserData {
    cv::Mat orgImg;                           // no-change image
    cv::Mat scaledOrgImg;                     // could be changed such as adding Rect
    cv::Mat scaledupdatedImg;                 // could be changed such as adding Rect
    cv::Rect rectInfo;                        // left-top and right-bottom coordinates of selected sub-ROI
    std::vector<QGraphicsRectItem*> rectItems;
    bool isRoiDetermined;                     // whether determine roi or not
    bool isThereCandidate;                    // whether exist the candidate roi or not
    float widthFactor;
    float heightFactor;
};


QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    bool eventFilter(QObject* obj, QEvent* event) override;
    void onPushButtonClicked();

private:
    Ui::MainWindow *ui;
    UserData userdata;
};
#endif // MAINWINDOW_H
