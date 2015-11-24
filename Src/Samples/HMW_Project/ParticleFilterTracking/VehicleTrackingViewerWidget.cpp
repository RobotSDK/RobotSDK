#include "VehicleTrackingViewerWidget.h"

VehicleTrackingViewerWidget::VehicleTrackingViewerWidget(QWidget * parent)
    : QGraphicsView(parent)
{
    scene=new QGraphicsScene;
    this->setScene(scene);
    presel=-1;
}

void VehicleTrackingViewerWidget::setPixmap(QImage &image)
{
    scene->clear();
    pixmap=new QGraphicsPixmapItem(QPixmap::fromImage(image));
    pixmap->setZValue(0);
    pixmap->setPos(0,0);
    scene->addItem(pixmap);
    presel=-1;
    rects.clear();
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
    rectitem->setPen(QPen(Qt::black,3));
    rectitem->setBrush(Qt::NoBrush);
    //rectitem->hide();
    rects.push_back(rectitem);
}

void VehicleTrackingViewerWidget::clear()
{
    scene->clear();
}

void VehicleTrackingViewerWidget::slotShowRect(int id)
{
    if(id<0||id>=rects.size())
    {
        return;
    }
    if(presel>=0)
    {
        rects[presel]->hide();
    }
    rects[id]->show();
    presel=id;
}
