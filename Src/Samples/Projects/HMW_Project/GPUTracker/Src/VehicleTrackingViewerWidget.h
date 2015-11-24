#ifndef VEHICLETRACKINGVIEWERWIDGET_H
#define VEHICLETRACKINGVIEWERWIDGET_H

#include<QGraphicsView>
#include<QGraphicsScene>
#include<QGraphicsPixmapItem>
#include<QGraphicsRectItem>
#include<QGraphicsTextItem>
#include<QImage>
#include<QPixmap>
#include<QMouseEvent>
#include<QVector>
#include<QRectF>

class VehicleTrackingViewerWidget : public QGraphicsView
{
public:
    VehicleTrackingViewerWidget(QWidget * parent=0);
protected:
    QGraphicsScene * scene;
    QGraphicsPixmapItem * pixmap;
    int center;
public:
    void setPixmap(QImage & image);
    void addTrackingResult(float x, float y, float theta, float width, float height);
    void clear();
};

#endif // VEHICLETRACKINGVIEWERWIDGET_H
