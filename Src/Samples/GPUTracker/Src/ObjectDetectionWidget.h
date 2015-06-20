#ifndef OBJECTDETECTIONWIDGET_H
#define OBJECTDETECTIONWIDGET_H

#include<QGraphicsView>
#include<QGraphicsScene>
#include<QGraphicsPixmapItem>
#include<QGraphicsItem>

#include<opencv2/opencv.hpp>

#include<RobotSDK.h>

namespace RobotSDK_Module
{

#define TrackedStateNum 5
typedef union{struct{double x,y,theta,width,length;};double data[TrackedStateNum];} TrackedState;

class TrackedObjectParticlePack
{
public:
    static const int statenum;
    static int particlenum;
    static TrackedState min;
    static TrackedState max;
    static TrackedState sigma;
public:
    TrackedObjectPack();
public:
    QTime timestamp;
    cv::Mat egotransform;
    QVector<int> ids;
    QVector<int> deltamsec;
    QVector<TrackedState> objects;
public:
    void addObject(QTime timeStamp, cv::Mat egoTransform, int id, TrackedObject object);
    void addObjectPack(TrackedObjectPack objectPack);
    void updateEgoTransform(QTime timeStamp, cv::Mat egoTransform);
};

class QGraphicsTrackedObjectItem : public QGraphicsItem
{
public:
    QGraphicsTrackedRectItem(TrackedObject * trackedObject, QGraphicsItem * parent=0);
public:
    TrackedObject * trackedobject;
};

class ObjectDetectionWidget : public QGraphicsView
{
public:
    ObjectDetectionWidget(QWidget * parent=0);
public:
    QGraphicsScene * scene;
    QGraphicsPixmapItem * map;
    int idcount;
public:
    void setMap(QImage & image);

};

}

#endif // OBJECTDETECTIONWIDGET_H
