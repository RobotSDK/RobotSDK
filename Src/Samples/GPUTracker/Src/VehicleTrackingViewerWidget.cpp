#include "VehicleTrackingViewerWidget.h"

VehicleTrackingViewerWidget::VehicleTrackingViewerWidget(QWidget * parent)
    : QGraphicsView(parent)
{
    scene=new QGraphicsScene;
    this->setScene(scene);
}

void VehicleTrackingViewerWidget::setPixmap(QImage &image)
{
    scene->clear();
    pixmap=new QGraphicsPixmapItem(QPixmap::fromImage(image));
    pixmap->setZValue(0);
    pixmap->setPos(0,0);
    scene->addItem(pixmap);
    center=image.width()/2;
}

void VehicleTrackingViewerWidget::addTrackingResult(float x, float y, float theta, float width, float height)
{
    QRectF rect;
    rect.setX(-width/2);rect.setY(-height/2);
    rect.setWidth(width);rect.setHeight(height);

    QGraphicsRectItem * rectitem=new QGraphicsRectItem(rect,pixmap);
    rectitem->setPos(center-y,center-x);
    float PI=3.14159265359;
    rectitem->setRotation(-theta*180/PI);
    rectitem->setZValue(1);
    rectitem->setPen(QPen(Qt::yellow,3));
    rectitem->setBrush(Qt::NoBrush);
}

void VehicleTrackingViewerWidget::clear()
{
    scene->clear();
}
