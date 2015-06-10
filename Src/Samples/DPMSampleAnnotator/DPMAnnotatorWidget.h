#ifndef DPMANNOTATORWIDGET_H
#define DPMANNOTATORWIDGET_H

#include<QWidget>
#include<QPushButton>
#include<QImage>
#include<QLabel>
#include<QMap>
#include<QMultiMap>
#include<QColor>
#include<QPen>
#include<QRadioButton>
#include<QGroupBox>
#include<QSplitter>
#include<QScrollArea>
#include<QTableWidget>
#include<QTableWidgetItem>
#include<QHeaderView>
#include<opencv2/opencv.hpp>

#include<RobotSDK.h>
namespace RobotSDK_Module
{

class RobotSDK_EXPORT DPMAnnotatorCheckBox : public QWidget
{
    Q_OBJECT
public:
    DPMAnnotatorCheckBox(QString attribute, int id=0, QWidget * parent=NULL);
protected:
    QHBoxLayout * layout;
public:
    int getSelection();
};

class RobotSDK_EXPORT DPMAnnotatorWidget : public QWidget
{
    Q_OBJECT
public:
    DPMAnnotatorWidget(QWidget *parent = 0);
    ~DPMAnnotatorWidget();
    void setAttributes(QString category, QString attribute);
    void clearAttributes();
    void showSample(cv::Mat image, int frameid, QString category, int id, cv::Rect rect, QString attributes);
    QString getAttributes();
protected:
    QLabel * imageviewer;
    QLabel * sampleinfo;
    QLabel * sampleviewer;
    QTableWidget * attributeslayout;
    bool annotateflag;
protected:
    QMultiMap<QString, QString> attributesmap;
public slots:
    void slotNext();
    void slotSkip();
signals:
    void signalNext();
};

}

#endif // DPMANNOTATORWIDGET_H
