#ifndef VEHICLEDETECTORWIDGET_H
#define VEHICLEDETECTORWIDGET_H

#include<QGraphicsView>
#include<QGraphicsScene>
#include<QGraphicsPixmapItem>
#include<QGraphicsLineItem>
#include<QImage>
#include<QPixmap>
#include<QMouseEvent>
#include<QVector>
#include<QLineF>

class VehicleDetectorWidget : public QGraphicsView
{
public:
    VehicleDetectorWidget(QWidget * parent=0);
protected:
    QGraphicsScene * scene;
    QGraphicsPixmapItem * pixmap;
    bool startpointflag;
    QPointF startpoint;
protected:
    void mousePressEvent(QMouseEvent *event);
public:
    void setPixmap(QImage & image);
    QVector<QLineF> getDetection();
    void clear();
};

#endif // VEHICLEDETECTORWIDGET_H
